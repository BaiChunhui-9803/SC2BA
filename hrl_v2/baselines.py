"""Matched-protocol PyMARL (QMIX) and on-policy (R-MAPPO) baselines for fair comparison.

Directly imports and runs the unmodified local algorithm implementations:
  - PyMARL QMIX:    D:\\白春辉\\实验平台\\pymarl (RNNAgent + QMixer, Double-Q, RMSprop)
  - on-policy MAPPO: D:\\白春辉\\实验平台\\on-policy (R_MAPPOPolicy + R_MAPPO + SharedReplayBuffer)

Both baselines execute on the exact same BaseMicroEnv (SC2BAUnitEnv / LegacyMMUnitEnv),
same step budgets, same opponent regimes, same optional curriculum warmup, and the exact
same frozen multi-opponent evaluation protocol (with SHA-256 immutability verification)
as DyIM-CHRL.
"""

from collections import Counter, defaultdict, deque
import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import RMSprop

from bch.plan import Layout
from bch.records import json_line, write_json

from .config import DEFAULT_LEGACY_MAP_ROOT, DEFAULT_SC2BA_PLAN, HRLv2Config, LEGACY_MM_SCENARIOS
from .env_adapter import BaseMicroEnv, StepBundle
from .eval_protocol import ALGO_OPPONENTS
from .trainer import resolve_device


DEFAULT_PYMARL_ROOT = Path(r"D:\白春辉\实验平台\pymarl")
DEFAULT_ONPOLICY_ROOT = Path(r"D:\白春辉\实验平台\on-policy")


def _ensure_sys_path(paths: Sequence[Path]) -> None:
    for p in paths:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _tensor_dict_fingerprint(named_tensors: Sequence[Tuple[str, torch.Tensor]]) -> str:
    h = sha256()
    for name, tensor in sorted(named_tensors, key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class _CompatBox:
    """Minimal gym.spaces.Box-compatible descriptor accepted by on-policy."""

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = tuple(shape)


_CompatBox.__name__ = "Box"


class _CompatDiscrete:
    """Minimal gym.spaces.Discrete-compatible descriptor accepted by on-policy."""

    def __init__(self, n: int):
        self.n = int(n)


_CompatDiscrete.__name__ = "Discrete"


def _make_spaces(obs_dim: int, state_dim: int, n_actions: int):
    try:
        from gym.spaces import Box, Discrete

        obs_sp = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        cent_sp = Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        act_sp = Discrete(n_actions)
        return obs_sp, cent_sp, act_sp
    except Exception:
        return _CompatBox((obs_dim,)), _CompatBox((state_dim,)), _CompatDiscrete(n_actions)


@dataclass
class BaselineConfig:
    algo: str = "qmix"  # qmix | mappo
    scenario: str = "3m"
    regime: str = "pool-all9"
    seed: int = 101
    step_mul: int = 8
    device: str = "auto"

    # Training & evaluation budgets (exact match to HRLv2Config)
    train_steps: int = 40000
    eval_every: int = 8000
    eval_init: bool = True
    val_repeats: int = 2
    test_repeats: int = 5
    legacy_eval_episodes: int = 30
    legacy_final_episodes: int = 120
    legacy_episode_limit: Optional[int] = None

    # Optional smooth curriculum & resume
    curriculum_warmup_frac: float = 0.0
    curriculum_bot_start_prob: float = 0.85
    resume_from: Optional[str] = None

    # PyMARL QMIX hyperparameters (aligned with pymarl/src/config/algs/qmix.yaml)
    qmix_rnn_hidden_dim: int = 64
    qmix_mixing_embed_dim: int = 32
    qmix_hypernet_layers: int = 2
    qmix_hypernet_embed: int = 64
    qmix_lr: float = 5e-4
    qmix_gamma: float = 0.99
    qmix_buffer_size: int = 5000
    qmix_batch_size: int = 32
    qmix_target_update_interval: int = 200
    qmix_epsilon_start: float = 1.0
    qmix_epsilon_finish: float = 0.05
    qmix_epsilon_anneal_steps: int = 50000
    qmix_grad_norm_clip: float = 10.0
    qmix_double_q: bool = True

    # on-policy MAPPO hyperparameters (aligned with on-policy SMAC defaults)
    mappo_hidden_size: int = 64
    mappo_layer_N: int = 1
    mappo_lr: float = 5e-4
    mappo_critic_lr: float = 5e-4
    mappo_gamma: float = 0.99
    mappo_gae_lambda: float = 0.95
    mappo_clip_param: float = 0.2
    mappo_ppo_epoch: int = 5
    mappo_num_mini_batch: int = 1
    mappo_entropy_coef: float = 0.01
    mappo_data_chunk_length: int = 10
    mappo_rollout_steps: int = 1020  # divisible by data_chunk_length=10, ~1024 matching DyIM-CHRL

    # External repo & plan paths (read-only)
    pymarl_root: str = str(DEFAULT_PYMARL_ROOT)
    onpolicy_root: str = str(DEFAULT_ONPOLICY_ROOT)
    plan_path: str = str(DEFAULT_SC2BA_PLAN)
    legacy_map_root: str = str(DEFAULT_LEGACY_MAP_ROOT)
    legacy_fast_reset: bool = True

    def is_legacy_mm(self) -> bool:
        return self.scenario in LEGACY_MM_SCENARIOS

    def to_env_cfg(self) -> HRLv2Config:
        return HRLv2Config(
            scenario=self.scenario,
            regime=self.regime,
            seed=self.seed,
            step_mul=self.step_mul,
            device=self.device,
            plan_path=self.plan_path,
            legacy_map_root=self.legacy_map_root,
            legacy_fast_reset=self.legacy_fast_reset,
            legacy_episode_limit=self.legacy_episode_limit,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _sample_curriculum_opponent(
    rng: np.random.Generator,
    train_opponents: Sequence[str],
    total_steps: int,
    train_steps: int,
    warmup_frac: float,
    bot_start_prob: float,
) -> Tuple[str, Optional[float]]:
    if warmup_frac <= 0.0 or "bot" not in train_opponents or len(train_opponents) <= 1:
        return train_opponents[int(rng.integers(len(train_opponents)))], None
    warmup_steps = max(1.0, float(warmup_frac) * float(train_steps))
    alpha = min(1.0, float(total_steps) / warmup_steps)
    uniform_p = 1.0 / float(len(train_opponents))
    start_p = float(np.clip(bot_start_prob, uniform_p, 0.98))
    p_bot = start_p * (1.0 - alpha) + uniform_p * alpha
    if float(rng.random()) < p_bot:
        return "bot", p_bot
    non_bot = [o for o in train_opponents if o != "bot"]
    return non_bot[int(rng.integers(len(non_bot)))], p_bot


def _aggregate_frozen_eval(
    rows: List[dict],
    by_opp: Dict[str, Counter],
    by_layout_opp: Dict[Tuple[str, str], Counter],
    hp_margins: List[float],
    infer_ms: List[float],
    layouts: Sequence[Layout],
    opponents: Sequence[str],
    step_tag: int,
    algo: str,
) -> dict:
    per_opponent = {}
    for opp in opponents:
        c = by_opp[opp]
        n = sum(c.values())
        per_opponent[opp] = {
            "episodes": n,
            "win": c["win"],
            "loss": c["loss"],
            "draw": c["draw"],
            "timeout": c["timeout"],
            "win_rate": c["win"] / n if n else 0.0,
        }

    all9_wr = float(np.mean([per_opponent[o]["win_rate"] for o in opponents])) if opponents else 0.0
    algos_present = [o for o in ALGO_OPPONENTS if o in per_opponent]
    algos8_wr = (
        float(np.mean([per_opponent[o]["win_rate"] for o in algos_present]))
        if algos_present else all9_wr
    )
    worst3_wr = (
        float(np.mean(sorted([per_opponent[o]["win_rate"] for o in algos_present])[:3]))
        if len(algos_present) >= 3 else algos8_wr
    )
    bot_wr = per_opponent["bot"]["win_rate"] if "bot" in per_opponent else None

    by_layout = {}
    for layout in layouts:
        wrs = []
        for opp in opponents:
            c = by_layout_opp[(layout.id, opp)]
            n = sum(c.values())
            if n:
                wrs.append(c["win"] / n)
        by_layout[layout.id] = float(np.mean(wrs)) if wrs else 0.0

    return {
        "checkpoint_steps": step_tag,
        "algorithm": algo,
        "episodes": len(rows),
        "all9_win_rate": all9_wr,
        "algos8_win_rate": algos8_wr,
        "worst3_algos_win_rate": worst3_wr,
        "bot_win_rate": bot_wr,
        "mean_hp_margin": float(np.mean(hp_margins)) if hp_margins else 0.0,
        "by_layout_win_rate": by_layout,
        "active_prototypes": None,
        "mean_clusters": None,
        "encode_ms_p50": float(np.median(infer_ms)) if infer_ms else 0.0,
        "per_opponent": per_opponent,
    }


# ==============================================================================
# 1. PyMARL QMIX (imports actual RNNAgent + QMixer from D:\白春辉\实验平台\pymarl)
# ==============================================================================

class PyMARLQMIXModel:
    """Thin wrapper around PyMARL's RNNAgent + QMixer for train, eval, and PK."""

    def __init__(
        self,
        cfg: BaselineConfig,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        device: torch.device,
    ):
        pymarl_root = Path(cfg.pymarl_root)
        _ensure_sys_path([pymarl_root, pymarl_root / "src"])
        from modules.agents.rnn_agent import RNNAgent
        from modules.mixers.qmix import QMixer

        self.cfg = cfg
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.device = device
        self.input_shape = self.obs_dim + self.n_actions + self.n_agents

        agent_args = SimpleNamespace(
            rnn_hidden_dim=cfg.qmix_rnn_hidden_dim,
            n_actions=self.n_actions,
        )
        mixer_args = SimpleNamespace(
            n_agents=self.n_agents,
            state_shape=(self.state_dim,),
            mixing_embed_dim=cfg.qmix_mixing_embed_dim,
            hypernet_layers=cfg.qmix_hypernet_layers,
            hypernet_embed=cfg.qmix_hypernet_embed,
        )
        self.agent = RNNAgent(self.input_shape, agent_args).to(device)
        self.target_agent = copy.deepcopy(self.agent).to(device)
        self.mixer = QMixer(mixer_args).to(device)
        self.target_mixer = copy.deepcopy(self.mixer).to(device)
        self.agent_ids = torch.eye(self.n_agents, dtype=torch.float32, device=device)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).contiguous()

    def build_inputs(self, obs_t: torch.Tensor, last_act_oh: torch.Tensor) -> torch.Tensor:
        """Concatenate [obs, last_action_onehot, agent_id_onehot] -> [B*N, input_shape]."""
        bsz = obs_t.shape[0]
        aid = self.agent_ids.unsqueeze(0).expand(bsz, -1, -1)
        return torch.cat([obs_t, last_act_oh, aid], dim=-1).reshape(bsz * self.n_agents, self.input_shape)

    def forward_step(
        self,
        obs_t: torch.Tensor,
        last_act_oh: torch.Tensor,
        hidden: torch.Tensor,
        *,
        use_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = obs_t.shape[0]
        net = self.target_agent if use_target else self.agent
        x = self.build_inputs(obs_t, last_act_oh)
        q, h_next = net(x, hidden.reshape(bsz * self.n_agents, -1))
        return (
            q.reshape(bsz, self.n_agents, self.n_actions),
            h_next.reshape(bsz, self.n_agents, -1),
        )

    @torch.inference_mode()
    def select_actions(
        self,
        obs_np: np.ndarray,
        avail_np: np.ndarray,
        last_act_oh_np: np.ndarray,
        hidden: torch.Tensor,
        *,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        la_t = torch.from_numpy(last_act_oh_np).unsqueeze(0).to(self.device)
        q_vals, h_next = self.forward_step(obs_t, la_t, hidden, use_target=False)
        q_np = q_vals[0].cpu().numpy()
        q_np[avail_np <= 0.5] = -1e9
        greedy = np.argmax(q_np, axis=-1).astype(int).tolist()
        if epsilon <= 0.0 or rng is None:
            return greedy, h_next
        actions = []
        for i in range(self.n_agents):
            if float(rng.random()) < epsilon:
                legal = np.flatnonzero(avail_np[i] > 0.5)
                actions.append(int(legal[int(rng.integers(len(legal)))]) if len(legal) > 0 else 0)
            else:
                actions.append(greedy[i])
        return actions, h_next

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def fingerprint(self) -> str:
        items = [(f"agent.{k}", v) for k, v in self.agent.state_dict().items()]
        items += [(f"mixer.{k}", v) for k, v in self.mixer.state_dict().items()]
        return _tensor_dict_fingerprint(items)

    def parameters(self):
        return list(self.agent.parameters()) + list(self.mixer.parameters())

    def eval_mode(self) -> None:
        self.agent.eval()
        self.mixer.eval()

    def train_mode(self) -> None:
        self.agent.train()
        self.mixer.train()


class PyMARLQMIXTrainer:
    """Episode-replay Double-Q QMIX trainer using PyMARL's RNNAgent & QMixer."""

    def __init__(
        self,
        cfg: BaselineConfig,
        env: BaseMicroEnv,
        *,
        train_opponents: Sequence[str],
        eval_opponents: Sequence[str],
        out_dir: Path,
        eval_env: Optional[BaseMicroEnv] = None,
    ):
        self.cfg = cfg
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.train_opponents = tuple(train_opponents)
        self.eval_opponents = tuple(eval_opponents)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        self.model = PyMARLQMIXModel(
            cfg, env.n_agents, env.obs_dim, env.state_dim, env.n_actions, self.device
        )
        self.optimizer = RMSprop(
            params=self.model.parameters(), lr=cfg.qmix_lr, alpha=0.99, eps=1e-5
        )
        self.buffer: deque = deque(maxlen=cfg.qmix_buffer_size)

        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0
        self.last_target_update_ep = 0
        self.learning_curve: List[dict] = []

    def _current_epsilon(self) -> float:
        frac = min(1.0, float(self.total_steps) / max(float(self.cfg.qmix_epsilon_anneal_steps), 1.0))
        return float(self.cfg.qmix_epsilon_start + frac * (self.cfg.qmix_epsilon_finish - self.cfg.qmix_epsilon_start))

    def _collect_episode(self) -> Tuple[dict, bool, Optional[float]]:
        layout = self.env.train_layouts[int(self.rng.integers(len(self.env.train_layouts)))]
        opponent, p_bot = _sample_curriculum_opponent(
            self.rng, self.train_opponents, self.total_steps, self.cfg.train_steps,
            self.cfg.curriculum_warmup_frac, self.cfg.curriculum_bot_start_prob,
        )
        bundle = self.env.reset_round(layout, opponent)
        hidden = self.model.init_hidden(1)
        last_act_oh = np.zeros((self.env.n_agents, self.env.n_actions), dtype=np.float32)

        obs_list = [bundle.unit_obs.copy()]
        state_list = [bundle.global_state.copy()]
        avail_list = [bundle.avail_actions.copy()]
        last_act_list = [last_act_oh.copy()]
        act_list = []
        rew_list = []
        term_list = []

        self.model.eval_mode()
        while not bundle.done:
            eps = self._current_epsilon()
            actions, hidden = self.model.select_actions(
                bundle.unit_obs, bundle.avail_actions, last_act_oh, hidden,
                epsilon=eps, rng=self.rng,
            )
            next_bundle = self.env.step_unit_actions(actions)
            self.total_steps += 1

            act_arr = np.asarray(actions, dtype=np.int64)
            next_last_act_oh = np.eye(self.env.n_actions, dtype=np.float32)[act_arr]
            # True termination (battle won/lost/draw) vs time-limit truncation
            is_timeout = bool(next_bundle.done and next_bundle.observation.friendly and next_bundle.observation.enemy)
            terminated = bool(next_bundle.done and not is_timeout)

            act_list.append(act_arr)
            rew_list.append(float(next_bundle.reward))
            term_list.append(1.0 if terminated else 0.0)

            obs_list.append(next_bundle.unit_obs.copy())
            state_list.append(next_bundle.global_state.copy())
            avail_list.append(next_bundle.avail_actions.copy())
            last_act_list.append(next_last_act_oh.copy())

            last_act_oh = next_last_act_oh
            bundle = next_bundle

        self.total_episodes += 1
        ep = {
            "T": len(act_list),
            "obs": np.stack(obs_list, axis=0),             # [T+1, N, obs_dim]
            "state": np.stack(state_list, axis=0),         # [T+1, state_dim]
            "avail": np.stack(avail_list, axis=0),         # [T+1, N, A]
            "last_act": np.stack(last_act_list, axis=0),   # [T+1, N, A]
            "actions": np.stack(act_list, axis=0),         # [T, N]
            "rewards": np.asarray(rew_list, dtype=np.float32),     # [T]
            "terminated": np.asarray(term_list, dtype=np.float32), # [T]
        }
        won = bool(bundle.info.get("red_battle_won"))
        return ep, won, p_bot

    def _train_step(self) -> Optional[dict]:
        if len(self.buffer) < self.cfg.qmix_batch_size:
            return None
        self.model.train_mode()
        idxs = self.rng.choice(len(self.buffer), size=self.cfg.qmix_batch_size, replace=False)
        batch_eps = [self.buffer[int(i)] for i in idxs]
        bsz = len(batch_eps)
        max_t = max(ep["T"] for ep in batch_eps)
        n_ag = self.env.n_agents
        n_ac = self.env.n_actions

        obs_b = np.zeros((bsz, max_t + 1, n_ag, self.env.obs_dim), dtype=np.float32)
        state_b = np.zeros((bsz, max_t + 1, self.env.state_dim), dtype=np.float32)
        avail_b = np.zeros((bsz, max_t + 1, n_ag, n_ac), dtype=np.float32)
        la_b = np.zeros((bsz, max_t + 1, n_ag, n_ac), dtype=np.float32)
        act_b = np.zeros((bsz, max_t, n_ag), dtype=np.int64)
        rew_b = np.zeros((bsz, max_t, 1), dtype=np.float32)
        term_b = np.zeros((bsz, max_t, 1), dtype=np.float32)
        mask_b = np.zeros((bsz, max_t, 1), dtype=np.float32)

        for b, ep in enumerate(batch_eps):
            t_len = ep["T"]
            obs_b[b, : t_len + 1] = ep["obs"]
            state_b[b, : t_len + 1] = ep["state"]
            avail_b[b, : t_len + 1] = ep["avail"]
            la_b[b, : t_len + 1] = ep["last_act"]
            act_b[b, :t_len] = ep["actions"]
            rew_b[b, :t_len, 0] = ep["rewards"]
            term_b[b, :t_len, 0] = ep["terminated"]
            mask_b[b, :t_len, 0] = 1.0

        dev = self.device
        obs_t = torch.from_numpy(obs_b).to(dev)
        state_t = torch.from_numpy(state_b).to(dev)
        avail_t = torch.from_numpy(avail_b).to(dev)
        la_t = torch.from_numpy(la_b).to(dev)
        act_t = torch.from_numpy(act_b).to(dev)
        rew_t = torch.from_numpy(rew_b).to(dev)
        term_t = torch.from_numpy(term_b).to(dev)
        mask_t = torch.from_numpy(mask_b).to(dev)

        # Unroll RNNAgent
        mac_out = []
        h = self.model.init_hidden(bsz)
        for t in range(max_t + 1):
            q_t, h = self.model.forward_step(obs_t[:, t], la_t[:, t], h, use_target=False)
            mac_out.append(q_t)
        mac_out = torch.stack(mac_out, dim=1)  # [B, T+1, N, A]
        chosen_q = torch.gather(mac_out[:, :-1], dim=3, index=act_t.unsqueeze(-1)).squeeze(-1)  # [B, T, N]

        with torch.no_grad():
            target_out = []
            th_h = self.model.init_hidden(bsz)
            for t in range(max_t + 1):
                tq_t, th_h = self.model.forward_step(obs_t[:, t], la_t[:, t], th_h, use_target=True)
                target_out.append(tq_t)
            target_out = torch.stack(target_out[1:], dim=1)  # [B, T, N, A]
            target_out[avail_t[:, 1:] <= 0.5] = -9999999.0

            if self.cfg.qmix_double_q:
                live_detach = mac_out[:, 1:].detach().clone()
                live_detach[avail_t[:, 1:] <= 0.5] = -9999999.0
                best_act = live_detach.max(dim=3, keepdim=True)[1]
                target_max_q = torch.gather(target_out, dim=3, index=best_act).squeeze(-1)
            else:
                target_max_q = target_out.max(dim=3)[0]

        q_tot = self.model.mixer(chosen_q, state_t[:, :-1])
        with torch.no_grad():
            target_q_tot = self.model.target_mixer(target_max_q, state_t[:, 1:])
            targets = rew_t + self.cfg.qmix_gamma * (1.0 - term_t) * target_q_tot

        td_error = (q_tot - targets) * mask_t
        denom = mask_t.sum().clamp_min(1.0)
        loss = (td_error ** 2).sum() / denom

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.qmix_grad_norm_clip))
        self.optimizer.step()
        self.update_count += 1

        if (self.total_episodes - self.last_target_update_ep) >= self.cfg.qmix_target_update_interval:
            self.model.update_targets()
            self.last_target_update_ep = self.total_episodes

        return {
            "update": self.update_count,
            "steps": self.total_steps,
            "episodes": self.total_episodes,
            "epsilon": round(self._current_epsilon(), 4),
            "loss": round(float(loss.item()), 5),
            "grad_norm": round(grad_norm, 4),
            "q_tot_mean": round(float((q_tot * mask_t).sum().item() / denom.item()), 4),
        }

    def evaluate_frozen(self, step_tag: int, *, final: bool = False) -> Tuple[dict, List[dict]]:
        if self.cfg.is_legacy_mm():
            layouts = self.eval_env.test_layouts
            opps = ("bot",)
            reps = self.cfg.legacy_final_episodes if final else self.cfg.legacy_eval_episodes
        else:
            if final:
                std = [l for l in self.eval_env.train_layouts if l.id == "standard"]
                layouts = tuple(std) + tuple(self.eval_env.test_layouts)
                reps = self.cfg.test_repeats
            else:
                layouts = self.eval_env.val_layouts
                reps = self.cfg.val_repeats
            opps = self.eval_opponents

        self.model.eval_mode()
        fp_before = self.model.fingerprint()
        rows: List[dict] = []
        by_opp = defaultdict(Counter)
        by_layout_opp = defaultdict(Counter)
        hp_margins: List[float] = []
        infer_ms: List[float] = []

        for layout in layouts:
            for opp in opps:
                for rep in range(reps):
                    bundle = self.eval_env.reset_round(layout, opp)
                    first_obs = bundle.observation
                    hidden = self.model.init_hidden(1)
                    last_act_oh = np.zeros((self.eval_env.n_agents, self.eval_env.n_actions), dtype=np.float32)
                    ep_rew = 0.0
                    ep_ms = []
                    while not bundle.done:
                        t0 = perf_counter()
                        actions, hidden = self.model.select_actions(
                            bundle.unit_obs, bundle.avail_actions, last_act_oh, hidden, epsilon=0.0
                        )
                        ep_ms.append((perf_counter() - t0) * 1000.0)
                        last_act_oh = np.eye(self.eval_env.n_actions, dtype=np.float32)[np.asarray(actions, dtype=np.int64)]
                        bundle = self.eval_env.step_unit_actions(actions)
                        ep_rew += bundle.reward
                    audit = self.eval_env.episode_details(first_obs, bundle.observation, bundle.info)
                    by_opp[opp][audit["outcome"]] += 1
                    by_layout_opp[(layout.id, opp)][audit["outcome"]] += 1
                    hp_margins.append(float(audit.get("hp_margin", 0.0)))
                    p50 = float(np.median(ep_ms)) if ep_ms else 0.0
                    infer_ms.append(p50)
                    rows.append({
                        "checkpoint_steps": step_tag,
                        "layout": layout.id,
                        "split": layout.split,
                        "opponent": opp,
                        "repeat": rep,
                        "outcome": audit["outcome"],
                        "steps": audit["steps"],
                        "final_hp": audit["final_hp"],
                        "hp_margin": audit.get("hp_margin", 0.0),
                        "episode_reward": float(ep_rew),
                        "encode_ms_p50": p50,
                    })

        if self.model.fingerprint() != fp_before:
            raise RuntimeError("PyMARL QMIX frozen evaluation mutated model weights!")

        summary = _aggregate_frozen_eval(
            rows, by_opp, by_layout_opp, hp_margins, infer_ms, layouts, opps, step_tag, "pymarl-qmix"
        )
        return summary, rows

    def save_checkpoint(self, path: Path) -> None:
        torch.save({
            "format": "pymarl-qmix-v2",
            "algorithm": "qmix",
            "config": self.cfg.to_dict(),
            "n_agents": self.env.n_agents,
            "obs_dim": self.env.obs_dim,
            "state_dim": self.env.state_dim,
            "n_actions": self.env.n_actions,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "update_count": self.update_count,
            "last_target_update_ep": self.last_target_update_ep,
            "agent_state_dict": self.model.agent.state_dict(),
            "target_agent_state_dict": self.model.target_agent.state_dict(),
            "mixer_state_dict": self.model.mixer.state_dict(),
            "target_mixer_state_dict": self.model.target_mixer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "learning_curve": self.learning_curve,
        }, path)

    def load_checkpoint(self, path: Path, *, load_optimizer: bool = True, load_history: bool = True) -> None:
        ckpt_path = Path(path)
        payload = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        self.model.agent.load_state_dict(payload["agent_state_dict"])
        self.model.target_agent.load_state_dict(payload.get("target_agent_state_dict", payload["agent_state_dict"]))
        self.model.mixer.load_state_dict(payload["mixer_state_dict"])
        self.model.target_mixer.load_state_dict(payload.get("target_mixer_state_dict", payload["mixer_state_dict"]))
        if load_optimizer and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.total_steps = int(payload.get("total_steps", 0))
        self.total_episodes = int(payload.get("total_episodes", 0))
        self.update_count = int(payload.get("update_count", 0))
        self.last_target_update_ep = int(payload.get("last_target_update_ep", self.total_episodes))
        if load_history and isinstance(payload.get("learning_curve"), list):
            self.learning_curve = list(payload["learning_curve"])

    def train(self) -> dict:
        t0 = perf_counter()
        cfg = self.cfg
        log_path = self.out_dir / "train_log.jsonl"
        val_eps_path = self.out_dir / "val_episodes.jsonl"

        resumed = self.total_steps > 0
        if resumed:
            next_eval_at = ((self.total_steps // cfg.eval_every) + 1) * cfg.eval_every
            if self.learning_curve:
                write_json(self.out_dir / "learning_curve.json", self.learning_curve)
        else:
            next_eval_at = 0 if cfg.eval_init else cfg.eval_every
            if cfg.eval_init:
                summary, rows = self.evaluate_frozen(0, final=False)
                self.learning_curve.append(summary)
                with val_eps_path.open("a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json_line(row))
                write_json(self.out_dir / "learning_curve.json", self.learning_curve)
                next_eval_at = cfg.eval_every

        recent_wins: deque = deque(maxlen=50)
        with log_path.open("a", encoding="utf-8") as log_f:
            while self.total_steps < cfg.train_steps:
                ep, won, p_bot = self._collect_episode()
                self.buffer.append(ep)
                recent_wins.append(1.0 if won else 0.0)
                metrics = self._train_step()
                if metrics is not None and (self.total_episodes % 10 == 0 or self.total_steps >= cfg.train_steps):
                    log_f.write(json_line({
                        **metrics,
                        "train_win_rate_50": round(float(np.mean(recent_wins)), 4),
                        "curriculum_bot_prob": p_bot,
                    }))
                    log_f.flush()

                if self.total_steps >= next_eval_at and next_eval_at <= cfg.train_steps:
                    summary, rows = self.evaluate_frozen(self.total_steps, final=False)
                    self.learning_curve.append(summary)
                    with val_eps_path.open("a", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json_line(row))
                    write_json(self.out_dir / "learning_curve.json", self.learning_curve)
                    self.save_checkpoint(self.out_dir / "checkpoint_latest.pt")
                    print(
                        f"[pymarl-qmix {cfg.scenario}/{cfg.regime}] step={self.total_steps}/{cfg.train_steps} "
                        f"all9={summary['all9_win_rate']:.3f} bot={summary['bot_win_rate']} eps={self._current_epsilon():.3f}",
                        flush=True,
                    )
                    next_eval_at += cfg.eval_every

        self.save_checkpoint(self.out_dir / "checkpoint_final.pt")
        final_summary, final_rows = self.evaluate_frozen(self.total_steps, final=True)
        with (self.out_dir / "final_eval_episodes.jsonl").open("w", encoding="utf-8") as f:
            for row in final_rows:
                f.write(json_line(row))

        elapsed = perf_counter() - t0
        report = {
            "format": "baseline-report-v2",
            "algorithm": "pymarl-qmix",
            "scenario": cfg.scenario,
            "backend": "legacy_mm" if cfg.is_legacy_mm() else "sc2ba",
            "regime": cfg.regime,
            "seed": cfg.seed,
            "train_steps": self.total_steps,
            "train_episodes": self.total_episodes,
            "updates": self.update_count,
            "resumed_from": cfg.resume_from,
            "curriculum_warmup_frac": cfg.curriculum_warmup_frac,
            "wall_seconds": round(elapsed, 2),
            "config": cfg.to_dict(),
            "final_eval": final_summary,
            "learning_curve_points": len(self.learning_curve),
        }
        write_json(self.out_dir / "report.json", report)
        write_json(self.out_dir / "done.json", {
            "algorithm": "pymarl-qmix",
            "scenario": cfg.scenario,
            "regime": cfg.regime,
            "seed": cfg.seed,
            "train_steps": self.total_steps,
            "resumed_from": cfg.resume_from,
            "all9_win_rate": final_summary["all9_win_rate"],
            "algos8_win_rate": final_summary["algos8_win_rate"],
            "worst3_algos_win_rate": final_summary["worst3_algos_win_rate"],
            "bot_win_rate": final_summary["bot_win_rate"],
            "mean_hp_margin": final_summary["mean_hp_margin"],
            "wall_seconds": round(elapsed, 2),
        })
        return report


# ==============================================================================
# 2. on-policy R-MAPPO (imports actual R_MAPPOPolicy + R_MAPPO + SharedReplayBuffer)
# ==============================================================================

def _build_onpolicy_args(cfg: BaselineConfig, episode_length: int, data_chunk_length: int):
    _ensure_sys_path([Path(cfg.onpolicy_root)])
    from onpolicy.config import get_config

    parser = get_config()
    args = parser.parse_args([])
    args.algorithm_name = "rmappo"
    args.seed = cfg.seed
    args.n_rollout_threads = 1
    args.n_eval_rollout_threads = 1
    args.episode_length = int(episode_length)
    args.hidden_size = int(cfg.mappo_hidden_size)
    args.layer_N = int(cfg.mappo_layer_N)
    args.use_recurrent_policy = True
    args.use_naive_recurrent_policy = False
    args.recurrent_N = 1
    args.data_chunk_length = int(data_chunk_length)
    args.lr = float(cfg.mappo_lr)
    args.critic_lr = float(cfg.mappo_critic_lr)
    args.ppo_epoch = int(cfg.mappo_ppo_epoch)
    args.num_mini_batch = int(cfg.mappo_num_mini_batch)
    args.clip_param = float(cfg.mappo_clip_param)
    args.entropy_coef = float(cfg.mappo_entropy_coef)
    args.gamma = float(cfg.mappo_gamma)
    args.gae_lambda = float(cfg.mappo_gae_lambda)
    args.use_valuenorm = True
    args.use_popart = False
    args.use_feature_normalization = True
    args.use_orthogonal = True
    args.use_ReLU = True
    args.gain = 0.01
    args.use_proper_time_limits = False
    args.use_value_active_masks = True
    args.use_policy_active_masks = True
    return args


class OnPolicyMAPPOTrainer:
    """Matched-protocol trainer wrapping on-policy's R_MAPPOPolicy + R_MAPPO."""

    def __init__(
        self,
        cfg: BaselineConfig,
        env: BaseMicroEnv,
        *,
        train_opponents: Sequence[str],
        eval_opponents: Sequence[str],
        out_dir: Path,
        eval_env: Optional[BaseMicroEnv] = None,
    ):
        _ensure_sys_path([Path(cfg.onpolicy_root)])
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
        from onpolicy.utils.shared_buffer import SharedReplayBuffer

        self._SharedReplayBuffer = SharedReplayBuffer
        self.cfg = cfg
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.train_opponents = tuple(train_opponents)
        self.eval_opponents = tuple(eval_opponents)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        self.obs_space, self.cent_obs_space, self.act_space = _make_spaces(
            env.obs_dim, env.state_dim, env.n_actions
        )
        self.op_args = _build_onpolicy_args(
            cfg, episode_length=cfg.mappo_rollout_steps, data_chunk_length=cfg.mappo_data_chunk_length
        )
        self.policy = R_MAPPOPolicy(
            self.op_args, self.obs_space, self.cent_obs_space, self.act_space, device=self.device
        )
        self.trainer = R_MAPPO(self.op_args, self.policy, device=self.device)

        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0
        self.learning_curve: List[dict] = []

        # Live episode cursor
        self._cur_bundle: Optional[StepBundle] = None
        self._cur_rnn_actor: Optional[np.ndarray] = None
        self._cur_rnn_critic: Optional[np.ndarray] = None
        self._cur_masks: Optional[np.ndarray] = None

    def _fingerprint(self) -> str:
        items = [(f"actor.{k}", v) for k, v in self.policy.actor.state_dict().items()]
        items += [(f"critic.{k}", v) for k, v in self.policy.critic.state_dict().items()]
        return _tensor_dict_fingerprint(items)

    def _start_new_episode(self) -> Optional[float]:
        layout = self.env.train_layouts[int(self.rng.integers(len(self.env.train_layouts)))]
        opponent, p_bot = _sample_curriculum_opponent(
            self.rng, self.train_opponents, self.total_steps, self.cfg.train_steps,
            self.cfg.curriculum_warmup_frac, self.cfg.curriculum_bot_start_prob,
        )
        self._cur_bundle = self.env.reset_round(layout, opponent)
        self._cur_rnn_actor = np.zeros(
            (1, self.env.n_agents, self.op_args.recurrent_N, self.op_args.hidden_size), dtype=np.float32
        )
        self._cur_rnn_critic = np.zeros_like(self._cur_rnn_actor)
        self._cur_masks = np.ones((1, self.env.n_agents, 1), dtype=np.float32)
        return p_bot

    @staticmethod
    def _tile_cent_obs(global_state: np.ndarray, n_agents: int) -> np.ndarray:
        return np.broadcast_to(global_state[None, None, :], (1, n_agents, global_state.shape[0])).copy()

    @staticmethod
    def _alive_active_mask(bundle: StepBundle) -> np.ndarray:
        # unit_extra[:, 3] == 1.0 iff agent is alive
        return bundle.unit_extra[:, 3].astype(np.float32).reshape(1, -1, 1)

    @staticmethod
    def _choose_chunk_len(rollout_len: int, preferred: int) -> int:
        for cand in (preferred, 10, 8, 5, 4, 2, 1):
            if cand <= rollout_len and rollout_len % cand == 0:
                return cand
        return 1

    def _collect_and_update(self, batch_steps: int) -> dict:
        chunk_len = self._choose_chunk_len(batch_steps, self.cfg.mappo_data_chunk_length)
        buf_args = copy.copy(self.op_args)
        buf_args.episode_length = batch_steps
        buf_args.data_chunk_length = chunk_len
        self.trainer.data_chunk_length = chunk_len

        buffer = self._SharedReplayBuffer(
            buf_args, self.env.n_agents, self.obs_space, self.cent_obs_space, self.act_space
        )
        p_bot = None
        if self._cur_bundle is None or self._cur_bundle.done:
            p_bot = self._start_new_episode()

        assert self._cur_bundle is not None
        buffer.share_obs[0] = self._tile_cent_obs(self._cur_bundle.global_state, self.env.n_agents)
        buffer.obs[0] = self._cur_bundle.unit_obs[None, :, :].copy()
        buffer.available_actions[0] = self._cur_bundle.avail_actions[None, :, :].copy()
        buffer.rnn_states[0] = self._cur_rnn_actor.copy()
        buffer.rnn_states_critic[0] = self._cur_rnn_critic.copy()
        buffer.masks[0] = self._cur_masks.copy()
        buffer.active_masks[0] = self._alive_active_mask(self._cur_bundle)

        self.trainer.prep_rollout()
        ep_wins = 0
        ep_finished = 0
        n_ag = self.env.n_agents

        for step in range(batch_steps):
            with torch.no_grad():
                val_t, act_t, lp_t, rnn_a_t, rnn_c_t = self.policy.get_actions(
                    buffer.share_obs[step].reshape(n_ag, -1),
                    buffer.obs[step].reshape(n_ag, -1),
                    buffer.rnn_states[step].reshape(n_ag, self.op_args.recurrent_N, -1),
                    buffer.rnn_states_critic[step].reshape(n_ag, self.op_args.recurrent_N, -1),
                    buffer.masks[step].reshape(n_ag, 1),
                    buffer.available_actions[step].reshape(n_ag, -1),
                    deterministic=False,
                )

            values = val_t.cpu().numpy().reshape(1, n_ag, 1)
            actions = act_t.cpu().numpy().reshape(1, n_ag, 1)
            action_lps = lp_t.cpu().numpy().reshape(1, n_ag, 1)
            rnn_a = rnn_a_t.cpu().numpy().reshape(1, n_ag, self.op_args.recurrent_N, -1)
            rnn_c = rnn_c_t.cpu().numpy().reshape(1, n_ag, self.op_args.recurrent_N, -1)

            act_list = actions[0, :, 0].astype(int).tolist()
            next_bundle = self.env.step_unit_actions(act_list)
            self.total_steps += 1
            rewards = np.full((1, n_ag, 1), float(next_bundle.reward), dtype=np.float32)

            if next_bundle.done:
                self.total_episodes += 1
                ep_finished += 1
                if next_bundle.info.get("red_battle_won"):
                    ep_wins += 1
                p_bot = self._start_new_episode()
                assert self._cur_bundle is not None
                masks = np.zeros((1, n_ag, 1), dtype=np.float32)
                rnn_a = np.zeros_like(rnn_a)
                rnn_c = np.zeros_like(rnn_c)
                next_cent = self._tile_cent_obs(self._cur_bundle.global_state, n_ag)
                next_obs = self._cur_bundle.unit_obs[None, :, :].copy()
                next_avail = self._cur_bundle.avail_actions[None, :, :].copy()
                next_active = self._alive_active_mask(self._cur_bundle)
            else:
                self._cur_bundle = next_bundle
                masks = np.ones((1, n_ag, 1), dtype=np.float32)
                next_cent = self._tile_cent_obs(next_bundle.global_state, n_ag)
                next_obs = next_bundle.unit_obs[None, :, :].copy()
                next_avail = next_bundle.avail_actions[None, :, :].copy()
                next_active = self._alive_active_mask(next_bundle)

            buffer.insert(
                next_cent, next_obs, rnn_a, rnn_c, actions, action_lps, values, rewards, masks,
                active_masks=next_active, available_actions=next_avail,
            )

        self._cur_rnn_actor = buffer.rnn_states[-1].copy()
        self._cur_rnn_critic = buffer.rnn_states_critic[-1].copy()
        self._cur_masks = buffer.masks[-1].copy()

        # Bootstrap final value & run on-policy R_MAPPO update
        with torch.no_grad():
            next_v_t = self.policy.get_values(
                buffer.share_obs[-1].reshape(n_ag, -1),
                buffer.rnn_states_critic[-1].reshape(n_ag, self.op_args.recurrent_N, -1),
                buffer.masks[-1].reshape(n_ag, 1),
            )
            next_v = next_v_t.cpu().numpy().reshape(1, n_ag, 1)

        buffer.compute_returns(next_v, self.trainer.value_normalizer)
        self.trainer.prep_training()
        train_info = self.trainer.train(buffer)
        self.update_count += 1

        return {
            "update": self.update_count,
            "steps": self.total_steps,
            "episodes_finished": ep_finished,
            "train_win_rate": (ep_wins / ep_finished) if ep_finished else None,
            "curriculum_bot_prob": p_bot,
            "policy_loss": round(float(train_info["policy_loss"]), 5),
            "value_loss": round(float(train_info["value_loss"]), 5),
            "dist_entropy": round(float(train_info["dist_entropy"]), 4),
        }

    def evaluate_frozen(self, step_tag: int, *, final: bool = False) -> Tuple[dict, List[dict]]:
        if self.cfg.is_legacy_mm():
            layouts = self.eval_env.test_layouts
            opps = ("bot",)
            reps = self.cfg.legacy_final_episodes if final else self.cfg.legacy_eval_episodes
        else:
            if final:
                std = [l for l in self.eval_env.train_layouts if l.id == "standard"]
                layouts = tuple(std) + tuple(self.eval_env.test_layouts)
                reps = self.cfg.test_repeats
            else:
                layouts = self.eval_env.val_layouts
                reps = self.cfg.val_repeats
            opps = self.eval_opponents

        self.trainer.prep_rollout()
        fp_before = self._fingerprint()
        rows: List[dict] = []
        by_opp = defaultdict(Counter)
        by_layout_opp = defaultdict(Counter)
        hp_margins: List[float] = []
        infer_ms: List[float] = []
        n_ag = self.eval_env.n_agents

        for layout in layouts:
            for opp in opps:
                for rep in range(reps):
                    bundle = self.eval_env.reset_round(layout, opp)
                    first_obs = bundle.observation
                    rnn_a = np.zeros((n_ag, self.op_args.recurrent_N, self.op_args.hidden_size), dtype=np.float32)
                    masks = np.ones((n_ag, 1), dtype=np.float32)
                    ep_rew = 0.0
                    ep_ms = []
                    with torch.inference_mode():
                        while not bundle.done:
                            t0 = perf_counter()
                            act_t, rnn_a_t = self.policy.act(
                                bundle.unit_obs, rnn_a, masks, bundle.avail_actions, deterministic=True
                            )
                            ep_ms.append((perf_counter() - t0) * 1000.0)
                            rnn_a = rnn_a_t.cpu().numpy()
                            actions = act_t.cpu().numpy().reshape(-1).astype(int).tolist()
                            bundle = self.eval_env.step_unit_actions(actions)
                            ep_rew += bundle.reward
                    audit = self.eval_env.episode_details(first_obs, bundle.observation, bundle.info)
                    by_opp[opp][audit["outcome"]] += 1
                    by_layout_opp[(layout.id, opp)][audit["outcome"]] += 1
                    hp_margins.append(float(audit.get("hp_margin", 0.0)))
                    p50 = float(np.median(ep_ms)) if ep_ms else 0.0
                    infer_ms.append(p50)
                    rows.append({
                        "checkpoint_steps": step_tag,
                        "layout": layout.id,
                        "split": layout.split,
                        "opponent": opp,
                        "repeat": rep,
                        "outcome": audit["outcome"],
                        "steps": audit["steps"],
                        "final_hp": audit["final_hp"],
                        "hp_margin": audit.get("hp_margin", 0.0),
                        "episode_reward": float(ep_rew),
                        "encode_ms_p50": p50,
                    })

        if self._fingerprint() != fp_before:
            raise RuntimeError("on-policy MAPPO frozen evaluation mutated model weights!")

        summary = _aggregate_frozen_eval(
            rows, by_opp, by_layout_opp, hp_margins, infer_ms, layouts, opps, step_tag, "onpolicy-mappo"
        )
        return summary, rows

    def save_checkpoint(self, path: Path) -> None:
        vn_state = self.trainer.value_normalizer.state_dict() if self.trainer.value_normalizer is not None else None
        torch.save({
            "format": "onpolicy-mappo-v2",
            "algorithm": "mappo",
            "config": self.cfg.to_dict(),
            "n_agents": self.env.n_agents,
            "obs_dim": self.env.obs_dim,
            "state_dim": self.env.state_dim,
            "n_actions": self.env.n_actions,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "update_count": self.update_count,
            "actor_state_dict": self.policy.actor.state_dict(),
            "critic_state_dict": self.policy.critic.state_dict(),
            "actor_optimizer_state_dict": self.policy.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.policy.critic_optimizer.state_dict(),
            "value_normalizer_state_dict": vn_state,
            "learning_curve": self.learning_curve,
        }, path)

    def load_checkpoint(self, path: Path, *, load_optimizer: bool = True, load_history: bool = True) -> None:
        ckpt_path = Path(path)
        payload = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        self.policy.actor.load_state_dict(payload["actor_state_dict"])
        self.policy.critic.load_state_dict(payload["critic_state_dict"])
        if load_optimizer:
            if "actor_optimizer_state_dict" in payload:
                self.policy.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in payload:
                self.policy.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        if payload.get("value_normalizer_state_dict") is not None and self.trainer.value_normalizer is not None:
            self.trainer.value_normalizer.load_state_dict(payload["value_normalizer_state_dict"])
        self.total_steps = int(payload.get("total_steps", 0))
        self.total_episodes = int(payload.get("total_episodes", 0))
        self.update_count = int(payload.get("update_count", 0))
        if load_history and isinstance(payload.get("learning_curve"), list):
            self.learning_curve = list(payload["learning_curve"])

    def train(self) -> dict:
        t0 = perf_counter()
        cfg = self.cfg
        log_path = self.out_dir / "train_log.jsonl"
        val_eps_path = self.out_dir / "val_episodes.jsonl"

        resumed = self.total_steps > 0
        if resumed:
            next_eval_at = ((self.total_steps // cfg.eval_every) + 1) * cfg.eval_every
            if self.learning_curve:
                write_json(self.out_dir / "learning_curve.json", self.learning_curve)
            self._cur_bundle = None
        else:
            next_eval_at = 0 if cfg.eval_init else cfg.eval_every
            if cfg.eval_init:
                summary, rows = self.evaluate_frozen(0, final=False)
                self.learning_curve.append(summary)
                with val_eps_path.open("a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json_line(row))
                write_json(self.out_dir / "learning_curve.json", self.learning_curve)
                next_eval_at = cfg.eval_every
                self._cur_bundle = None

        with log_path.open("a", encoding="utf-8") as log_f:
            while self.total_steps < cfg.train_steps:
                remaining = cfg.train_steps - self.total_steps
                batch_steps = min(cfg.mappo_rollout_steps, remaining)
                metrics = self._collect_and_update(batch_steps)
                log_f.write(json_line(metrics))
                log_f.flush()

                if self.total_steps >= next_eval_at and next_eval_at <= cfg.train_steps:
                    summary, rows = self.evaluate_frozen(self.total_steps, final=False)
                    self.learning_curve.append(summary)
                    with val_eps_path.open("a", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json_line(row))
                    write_json(self.out_dir / "learning_curve.json", self.learning_curve)
                    self.save_checkpoint(self.out_dir / "checkpoint_latest.pt")
                    print(
                        f"[onpolicy-mappo {cfg.scenario}/{cfg.regime}] step={self.total_steps}/{cfg.train_steps} "
                        f"all9={summary['all9_win_rate']:.3f} bot={summary['bot_win_rate']}",
                        flush=True,
                    )
                    next_eval_at += cfg.eval_every
                    if self.eval_env is self.env:
                        self._cur_bundle = None

        self.save_checkpoint(self.out_dir / "checkpoint_final.pt")
        final_summary, final_rows = self.evaluate_frozen(self.total_steps, final=True)
        with (self.out_dir / "final_eval_episodes.jsonl").open("w", encoding="utf-8") as f:
            for row in final_rows:
                f.write(json_line(row))

        elapsed = perf_counter() - t0
        report = {
            "format": "baseline-report-v2",
            "algorithm": "onpolicy-mappo",
            "scenario": cfg.scenario,
            "backend": "legacy_mm" if cfg.is_legacy_mm() else "sc2ba",
            "regime": cfg.regime,
            "seed": cfg.seed,
            "train_steps": self.total_steps,
            "train_episodes": self.total_episodes,
            "updates": self.update_count,
            "resumed_from": cfg.resume_from,
            "curriculum_warmup_frac": cfg.curriculum_warmup_frac,
            "wall_seconds": round(elapsed, 2),
            "config": cfg.to_dict(),
            "final_eval": final_summary,
            "learning_curve_points": len(self.learning_curve),
        }
        write_json(self.out_dir / "report.json", report)
        write_json(self.out_dir / "done.json", {
            "algorithm": "onpolicy-mappo",
            "scenario": cfg.scenario,
            "regime": cfg.regime,
            "seed": cfg.seed,
            "train_steps": self.total_steps,
            "resumed_from": cfg.resume_from,
            "all9_win_rate": final_summary["all9_win_rate"],
            "algos8_win_rate": final_summary["algos8_win_rate"],
            "worst3_algos_win_rate": final_summary["worst3_algos_win_rate"],
            "bot_win_rate": final_summary["bot_win_rate"],
            "mean_hp_margin": final_summary["mean_hp_margin"],
            "wall_seconds": round(elapsed, 2),
        })
        return report
