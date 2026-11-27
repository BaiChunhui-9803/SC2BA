"""Two-timescale Semi-MDP PPO trainer for DyIM-CHRL.

Co-trains:
  - Upper online IM-CNN + EMA-VQ dynamic prototype pool (Split/Merge/Prune) +
    granularity head + extensible cluster guidance head + upper centralized critic
    at macro interval tau.
  - Lower cluster-conditioned FiLM GRU-Actor + lower centralized CTDE critic at every
    environment step over native atomic unit actions (zero manual macros).
"""

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from bch.records import json_line, write_json

from .config import HRLv2Config
from .env_adapter import BaseMicroEnv, StepBundle
from .eval_protocol import evaluate_frozen
from .influence_map import build_tactical_influence_map
from .lower_policy import DyIMCHRLPolicy
from .tactical_cluster import TacticalClusterer


@dataclass
class LowerStepRecord:
    unit_obs: np.ndarray            # [N, obs_dim]
    unit_extra: np.ndarray          # [N, 4]
    unit_rel_cluster: np.ndarray    # [N, 2]
    unit_cluster_feat: np.ndarray   # [N, D_fc]
    unit_guidance: np.ndarray       # [N, D_g]
    rnn_hidden: np.ndarray          # [N, H]
    avail_actions: np.ndarray       # [N, A]
    global_state: np.ndarray        # [state_dim]
    actions: np.ndarray             # [N] int64
    log_probs: np.ndarray           # [N] float32
    values: np.ndarray              # [N] float32
    alive_mask: np.ndarray          # [N] float32
    reward: float
    done: bool
    next_values: np.ndarray         # [N] float32 (0 if done else bootstrap)


@dataclass
class UpperMacroRecord:
    im_grid: np.ndarray             # [C, H, W]
    global_state: np.ndarray        # [state_dim]
    alive_count: int
    gran_idx: int
    cluster_feats: np.ndarray       # [M_max, D_c]
    cluster_norm_coords: np.ndarray # [M_max, 2]
    cluster_mask: np.ndarray        # [M_max]
    raw_guidance: np.ndarray        # [M_max] or [M_max, D_g]
    old_log_prob: float             # log p(gran) + log p(guidance)
    old_value: float
    macro_return: float             # discounted sum over tau steps
    span: int                       # actual number of micro steps executed in this macro option
    done: bool
    next_value: float               # V_high at next macro state (0 if done)


def _compute_lower_gae(
    records: List[LowerStepRecord],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-agent GAE over collected lower steps; returns (advantages [T, N], returns [T, N])."""
    t_len = len(records)
    n_agents = records[0].actions.shape[0]
    adv = np.zeros((t_len, n_agents), dtype=np.float32)
    last_gae = np.zeros((n_agents,), dtype=np.float32)
    for t in range(t_len - 1, -1, -1):
        rec = records[t]
        non_term = 0.0 if rec.done else 1.0
        next_v = rec.next_values if (t == t_len - 1 or rec.done) else records[t + 1].values
        delta = rec.reward + gamma * non_term * next_v - rec.values
        last_gae = delta + gamma * gae_lambda * non_term * last_gae
        adv[t] = last_gae * rec.alive_mask
    returns = adv + np.stack([r.values for r in records], axis=0)
    return adv, returns


def _compute_upper_gae(
    records: List[UpperMacroRecord],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-MDP GAE over macro transitions with variable option span; returns (adv, returns, td_errors)."""
    m_len = len(records)
    adv = np.zeros((m_len,), dtype=np.float32)
    td_errors = np.zeros((m_len,), dtype=np.float32)
    last_gae = 0.0
    for i in range(m_len - 1, -1, -1):
        rec = records[i]
        non_term = 0.0 if rec.done else 1.0
        disc = gamma ** max(rec.span, 1)
        next_v = rec.next_value if (i == m_len - 1 or rec.done) else records[i + 1].old_value
        delta = rec.macro_return + disc * non_term * next_v - rec.old_value
        td_errors[i] = delta
        last_gae = delta + disc * gae_lambda * non_term * last_gae
        adv[i] = last_gae
    returns = adv + np.asarray([r.old_value for r in records], dtype=np.float32)
    return adv, returns, td_errors


def resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


class DyIMCHRLTrainer:
    """End-to-end online trainer with dynamic codebook logging and frozen evaluation."""

    def __init__(
        self,
        cfg: HRLv2Config,
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

        self.policy = DyIMCHRLPolicy(
            cfg, obs_dim=env.obs_dim, state_dim=env.state_dim, n_actions=env.n_actions
        ).to(self.device)

        self.opt_upper = torch.optim.Adam(self.policy.upper_parameters(), lr=cfg.lr_upper, eps=1e-5)
        self.opt_actor = torch.optim.Adam(self.policy.actor_parameters(), lr=cfg.lr_actor, eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.policy.critic_parameters(), lr=cfg.lr_critic, eps=1e-5)

        self.clusterer = TacticalClusterer(cfg)
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0
        self.learning_curve: List[dict] = []

        # Live episode cursor for continuous rollout collection across PPO updates
        self._cur_bundle: Optional[StepBundle] = None
        self._cur_hidden: Optional[torch.Tensor] = None
        self._cur_ep_step = 0
        self._cur_layout_idx = 0
        self._cur_opp_idx = 0
        self._cur_macro_rec: Optional[UpperMacroRecord] = None
        self._cur_spatial: Optional[torch.Tensor] = None
        self._cur_guidance: Optional[torch.Tensor] = None
        self._cur_req_clusters = 1

    def _current_bot_sampling_prob(self) -> Optional[float]:
        """Compute smooth bot-anchored curriculum probability if enabled for multi-opponent pool."""
        if (
            self.cfg.curriculum_warmup_frac <= 0.0
            or "bot" not in self.train_opponents
            or len(self.train_opponents) <= 1
        ):
            return None
        warmup_steps = max(1.0, float(self.cfg.curriculum_warmup_frac) * float(self.cfg.train_steps))
        alpha = min(1.0, float(self.total_steps) / warmup_steps)
        uniform_p = 1.0 / float(len(self.train_opponents))
        start_p = float(np.clip(self.cfg.curriculum_bot_start_prob, uniform_p, 0.98))
        return start_p * (1.0 - alpha) + uniform_p * alpha

    def _sample_train_match(self):
        layout = self.env.train_layouts[int(self.rng.integers(len(self.env.train_layouts)))]
        p_bot = self._current_bot_sampling_prob()
        if p_bot is None:
            opponent = self.train_opponents[int(self.rng.integers(len(self.train_opponents)))]
            return layout, opponent
        non_bot = [o for o in self.train_opponents if o != "bot"]
        if float(self.rng.random()) < p_bot:
            opponent = "bot"
        else:
            opponent = non_bot[int(self.rng.integers(len(non_bot)))]
        return layout, opponent

    def _start_new_episode(self) -> None:
        layout, opponent = self._sample_train_match()
        self.clusterer.reset()
        self._cur_bundle = self.env.reset_round(layout, opponent)
        self._cur_hidden = self.policy.init_lower_hidden(1, self.env.n_agents, self.device)
        self._cur_ep_step = 0
        self._cur_macro_rec = None
        self._cur_spatial = None
        self._cur_guidance = None
        self._cur_req_clusters = 1

    @torch.no_grad()
    def _collect_rollout(self, step_budget: int) -> Tuple[List[LowerStepRecord], List[UpperMacroRecord], dict]:
        self.policy.eval()
        lower_buf: List[LowerStepRecord] = []
        upper_buf: List[UpperMacroRecord] = []
        ep_wins = 0
        ep_finished = 0

        if self._cur_bundle is None or self._cur_bundle.done:
            self._start_new_episode()

        for _ in range(step_budget):
            bundle = self._cur_bundle
            assert bundle is not None and self._cur_hidden is not None
            im_grid = build_tactical_influence_map(
                bundle.observation, grid_size=self.cfg.im_size, channels=self.cfg.im_channels
            )
            is_macro = (self._cur_ep_step % self.cfg.macro_interval == 0) or (self._cur_macro_rec is None)

            if is_macro:
                im_t = torch.from_numpy(im_grid.tensor).unsqueeze(0).to(self.device)
                gstate_t = torch.from_numpy(bundle.global_state).unsqueeze(0).to(self.device)
                abs_out = self.policy.encode_upper(im_t)
                v_high = float(self.policy.value_upper(abs_out.z_quant, gstate_t).item())

                # Close previous macro option if one is open in the same episode
                if self._cur_macro_rec is not None:
                    self._cur_macro_rec.next_value = v_high
                    self._cur_macro_rec.done = False
                    upper_buf.append(self._cur_macro_rec)

                alive_n = len(bundle.observation.friendly)
                alive_t = torch.tensor([alive_n], dtype=torch.long, device=self.device)
                gran_idx_t, gran_lp_t, _ = self.policy.sample_granularity(abs_out.z_quant, alive_t)
                gran_idx = int(gran_idx_t.item())
                self._cur_req_clusters = gran_idx + 1

                assign = self.clusterer.assign(
                    bundle.observation,
                    bundle.agent_tags,
                    im_grid,
                    requested_clusters=self._cur_req_clusters,
                    recluster=True,
                )
                c_feats_t = torch.from_numpy(assign.cluster_feats).unsqueeze(0).to(self.device)
                c_coords_t = torch.from_numpy(assign.cluster_norm_coords).unsqueeze(0).to(self.device)
                c_mask_t = torch.from_numpy(assign.cluster_mask).unsqueeze(0).to(self.device)
                fused_cf = self.policy.fuse_cluster_features(abs_out.spatial_feat, c_feats_t, c_coords_t, c_mask_t)
                g_out = self.policy.guidance_head.sample(abs_out.z_quant, fused_cf, c_mask_t)

                self._cur_spatial = abs_out.spatial_feat
                self._cur_guidance = g_out.cluster_guidance
                self._cur_macro_rec = UpperMacroRecord(
                    im_grid=im_grid.tensor.copy(),
                    global_state=bundle.global_state.copy(),
                    alive_count=alive_n,
                    gran_idx=gran_idx,
                    cluster_feats=assign.cluster_feats.copy(),
                    cluster_norm_coords=assign.cluster_norm_coords.copy(),
                    cluster_mask=assign.cluster_mask.copy(),
                    raw_guidance=g_out.raw_action[0].detach().cpu().numpy().copy(),
                    old_log_prob=float((gran_lp_t + g_out.log_prob).item()),
                    old_value=v_high,
                    macro_return=0.0,
                    span=0,
                    done=False,
                    next_value=0.0,
                )
            else:
                assign = self.clusterer.assign(
                    bundle.observation,
                    bundle.agent_tags,
                    im_grid,
                    requested_clusters=self._cur_req_clusters,
                    recluster=False,
                )
                c_feats_t = torch.from_numpy(assign.cluster_feats).unsqueeze(0).to(self.device)
                c_coords_t = torch.from_numpy(assign.cluster_norm_coords).unsqueeze(0).to(self.device)
                c_mask_t = torch.from_numpy(assign.cluster_mask).unsqueeze(0).to(self.device)
                fused_cf = self.policy.fuse_cluster_features(self._cur_spatial, c_feats_t, c_coords_t, c_mask_t)

            # Lower step decision
            u_obs_t = torch.from_numpy(bundle.unit_obs).unsqueeze(0).to(self.device)
            u_ext_t = torch.from_numpy(bundle.unit_extra).unsqueeze(0).to(self.device)
            u_rel_t = torch.from_numpy(assign.unit_rel_cluster).unsqueeze(0).to(self.device)
            u2c_t = torch.from_numpy(assign.unit_to_cluster).unsqueeze(0).to(self.device)
            avail_t = torch.from_numpy(bundle.avail_actions).unsqueeze(0).to(self.device)
            gstate_t = torch.from_numpy(bundle.global_state).unsqueeze(0).to(self.device)

            u_cf_t, u_gd_t = self.policy.gather_unit_cluster_tensors(fused_cf, self._cur_guidance, u2c_t)
            h_in_np = self._cur_hidden[0].cpu().numpy().copy()

            actions_t, lp_t, _, val_t, h_next = self.policy.act_lower(
                u_obs_t,
                u_ext_t,
                u_rel_t,
                fused_cf,
                self._cur_guidance,
                u2c_t,
                self._cur_hidden,
                avail_t,
                gstate_t,
                deterministic=False,
            )

            next_bundle = self.env.step_unit_actions(actions_t[0].cpu().tolist())
            intrinsic = self.policy.guidance_head.intrinsic_reward(
                bundle.observation, next_bundle.observation, assign, self._cur_macro_rec.raw_guidance
            )
            step_reward = float(next_bundle.reward) + float(intrinsic)
            alive_mask = (assign.unit_to_cluster >= 0).astype(np.float32)

            # Accumulate macro return
            assert self._cur_macro_rec is not None
            self._cur_macro_rec.macro_return += (self.cfg.gamma ** self._cur_macro_rec.span) * float(next_bundle.reward)
            self._cur_macro_rec.span += 1

            lower_buf.append(
                LowerStepRecord(
                    unit_obs=bundle.unit_obs.copy(),
                    unit_extra=bundle.unit_extra.copy(),
                    unit_rel_cluster=assign.unit_rel_cluster.copy(),
                    unit_cluster_feat=u_cf_t[0].cpu().numpy().copy(),
                    unit_guidance=u_gd_t[0].cpu().numpy().copy(),
                    rnn_hidden=h_in_np,
                    avail_actions=bundle.avail_actions.copy(),
                    global_state=bundle.global_state.copy(),
                    actions=actions_t[0].cpu().numpy().astype(np.int64),
                    log_probs=lp_t[0].cpu().numpy().astype(np.float32),
                    values=val_t[0].cpu().numpy().astype(np.float32),
                    alive_mask=alive_mask,
                    reward=step_reward,
                    done=next_bundle.done,
                    next_values=np.zeros((self.env.n_agents,), dtype=np.float32),
                )
            )

            self.total_steps += 1
            self._cur_ep_step += 1
            self._cur_bundle = next_bundle
            self._cur_hidden = h_next

            if next_bundle.done:
                self._cur_macro_rec.done = True
                self._cur_macro_rec.next_value = 0.0
                upper_buf.append(self._cur_macro_rec)
                self._cur_macro_rec = None
                self.total_episodes += 1
                ep_finished += 1
                if next_bundle.info.get("red_battle_won"):
                    ep_wins += 1
                self._start_new_episode()

        # Bootstrap last lower & open macro transitions if rollout ended mid-episode
        if lower_buf and not lower_buf[-1].done:
            b = self._cur_bundle
            assert b is not None and self._cur_hidden is not None
            im_grid = build_tactical_influence_map(
                b.observation, grid_size=self.cfg.im_size, channels=self.cfg.im_channels
            )
            assign = self.clusterer.assign(
                b.observation,
                b.agent_tags,
                im_grid,
                requested_clusters=self._cur_req_clusters,
                recluster=False,
            )
            c_feats_t = torch.from_numpy(assign.cluster_feats).unsqueeze(0).to(self.device)
            c_coords_t = torch.from_numpy(assign.cluster_norm_coords).unsqueeze(0).to(self.device)
            c_mask_t = torch.from_numpy(assign.cluster_mask).unsqueeze(0).to(self.device)
            fused_cf = self.policy.fuse_cluster_features(self._cur_spatial, c_feats_t, c_coords_t, c_mask_t)
            u_obs_t = torch.from_numpy(b.unit_obs).unsqueeze(0).to(self.device)
            u_ext_t = torch.from_numpy(b.unit_extra).unsqueeze(0).to(self.device)
            u_rel_t = torch.from_numpy(assign.unit_rel_cluster).unsqueeze(0).to(self.device)
            u2c_t = torch.from_numpy(assign.unit_to_cluster).unsqueeze(0).to(self.device)
            avail_t = torch.from_numpy(b.avail_actions).unsqueeze(0).to(self.device)
            gstate_t = torch.from_numpy(b.global_state).unsqueeze(0).to(self.device)
            _, _, _, next_v_low, _ = self.policy.act_lower(
                u_obs_t, u_ext_t, u_rel_t, fused_cf, self._cur_guidance, u2c_t,
                self._cur_hidden, avail_t, gstate_t, deterministic=True,
            )
            lower_buf[-1].next_values = next_v_low[0].cpu().numpy().astype(np.float32)

        if self._cur_macro_rec is not None and self._cur_macro_rec.span > 0:
            b = self._cur_bundle
            assert b is not None
            im_grid = build_tactical_influence_map(
                b.observation, grid_size=self.cfg.im_size, channels=self.cfg.im_channels
            )
            im_t = torch.from_numpy(im_grid.tensor).unsqueeze(0).to(self.device)
            gstate_t = torch.from_numpy(b.global_state).unsqueeze(0).to(self.device)
            abs_out = self.policy.encode_upper(im_t)
            self._cur_macro_rec.next_value = float(self.policy.value_upper(abs_out.z_quant, gstate_t).item())
            self._cur_macro_rec.done = False
            upper_buf.append(self._cur_macro_rec)
            # Clear open macro record so next rollout starts a clean macro step
            self._cur_macro_rec = None

        stats = {
            "episodes_finished": ep_finished,
            "train_win_rate": (ep_wins / ep_finished) if ep_finished else None,
            "lower_samples": len(lower_buf),
            "upper_samples": len(upper_buf),
            "curriculum_bot_prob": self._current_bot_sampling_prob(),
        }
        return lower_buf, upper_buf, stats

    def _update_ppo(
        self,
        lower_buf: List[LowerStepRecord],
        upper_buf: List[UpperMacroRecord],
    ) -> Tuple[dict, List[dict]]:
        self.policy.train()
        self.update_count += 1
        cfg = self.cfg
        dev = self.device

        # ---------------- 1. Lower Cluster-Conditioned MAPPO Update ----------------
        low_adv, low_ret = _compute_lower_gae(lower_buf, cfg.gamma, cfg.gae_lambda)
        u_obs = torch.from_numpy(np.stack([r.unit_obs for r in lower_buf])).to(dev)
        u_ext = torch.from_numpy(np.stack([r.unit_extra for r in lower_buf])).to(dev)
        u_rel = torch.from_numpy(np.stack([r.unit_rel_cluster for r in lower_buf])).to(dev)
        u_cf = torch.from_numpy(np.stack([r.unit_cluster_feat for r in lower_buf])).to(dev)
        u_gd = torch.from_numpy(np.stack([r.unit_guidance for r in lower_buf])).to(dev)
        u_hid = torch.from_numpy(np.stack([r.rnn_hidden for r in lower_buf])).to(dev)
        u_avail = torch.from_numpy(np.stack([r.avail_actions for r in lower_buf])).to(dev)
        g_state = torch.from_numpy(np.stack([r.global_state for r in lower_buf])).to(dev)
        acts = torch.from_numpy(np.stack([r.actions for r in lower_buf])).to(dev)
        old_lp = torch.from_numpy(np.stack([r.log_probs for r in lower_buf])).to(dev)
        alive = torch.from_numpy(np.stack([r.alive_mask for r in lower_buf])).to(dev)
        adv_t = torch.from_numpy(low_adv).to(dev)
        ret_t = torch.from_numpy(low_ret).to(dev)

        alive_bool = alive > 0.5
        if alive_bool.any():
            a_mean = adv_t[alive_bool].mean()
            a_std = adv_t[alive_bool].std(unbiased=False).clamp_min(1e-6)
            adv_norm = ((adv_t - a_mean) / a_std) * alive
        else:
            adv_norm = adv_t

        t_len = len(lower_buf)
        mb = min(cfg.minibatch_size, max(1, t_len))
        low_policy_losses, low_value_losses, low_entropies = [], [], []

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(t_len, device=dev)
            for start in range(0, t_len, mb):
                idx = perm[start : start + mb]
                lp, ent, val = self.policy.evaluate_lower(
                    u_obs[idx], u_ext[idx], u_rel[idx], u_cf[idx], u_gd[idx],
                    u_hid[idx], u_avail[idx], g_state[idx], acts[idx],
                )
                mask = alive[idx]
                denom = mask.sum().clamp_min(1.0)
                ratio = torch.exp((lp - old_lp[idx]).clamp(-10.0, 10.0))
                s1 = ratio * adv_norm[idx]
                s2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_norm[idx]
                p_loss = -(torch.minimum(s1, s2) * mask).sum() / denom
                v_loss = (0.5 * (val - ret_t[idx]).pow(2) * mask).sum() / denom
                e_bonus = (ent * mask).sum() / denom
                loss = p_loss + cfg.value_coef * v_loss - cfg.entropy_coef * e_bonus

                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters(), cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters(), cfg.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

                low_policy_losses.append(float(p_loss.item()))
                low_value_losses.append(float(v_loss.item()))
                low_entropies.append(float(e_bonus.item()))

        # ---------------- 2. Upper Online IM-CNN + Dynamic VQ + Guidance Update ----------------
        up_policy_losses, up_value_losses, commit_losses, bisim_losses, rpred_losses = [], [], [], [], []
        sm_events: List[dict] = []

        if upper_buf:
            up_adv, up_ret, up_td = _compute_upper_gae(upper_buf, cfg.gamma, cfg.gae_lambda)
            im_b = torch.from_numpy(np.stack([r.im_grid for r in upper_buf])).to(dev)
            gs_b = torch.from_numpy(np.stack([r.global_state for r in upper_buf])).to(dev)
            alive_n_b = torch.tensor([r.alive_count for r in upper_buf], dtype=torch.long, device=dev)
            gran_b = torch.tensor([r.gran_idx for r in upper_buf], dtype=torch.long, device=dev)
            cf_b = torch.from_numpy(np.stack([r.cluster_feats for r in upper_buf])).to(dev)
            cc_b = torch.from_numpy(np.stack([r.cluster_norm_coords for r in upper_buf])).to(dev)
            cm_b = torch.from_numpy(np.stack([r.cluster_mask for r in upper_buf])).to(dev)
            rg_b = torch.from_numpy(np.stack([r.raw_guidance for r in upper_buf])).to(dev)
            old_ulp = torch.tensor([r.old_log_prob for r in upper_buf], dtype=torch.float32, device=dev)
            m_ret_b = torch.tensor([r.macro_return for r in upper_buf], dtype=torch.float32, device=dev)
            next_v_b = torch.tensor([r.next_value for r in upper_buf], dtype=torch.float32, device=dev)
            u_adv_t = torch.from_numpy(up_adv).to(dev)
            u_ret_t = torch.from_numpy(up_ret).to(dev)
            u_td_t = torch.from_numpy(up_td).to(dev)

            # Online warm initialization of the dynamic codebook on the very first batch (zero offline data)
            if not bool(self.policy.abstractor.codebook.initialized.item()):
                with torch.no_grad():
                    z_init, _ = self.policy.abstractor.backbone(im_b)
                    self.policy.abstractor.codebook.warm_initialize(z_init)

            u_adv_norm = (u_adv_t - u_adv_t.mean()) / u_adv_t.std(unbiased=False).clamp_min(1e-6)
            u_len = len(upper_buf)
            u_mb = min(max(32, cfg.minibatch_size // 2), max(1, u_len))

            for _ in range(cfg.ppo_epochs):
                perm = torch.randperm(u_len, device=dev)
                for start in range(0, u_len, u_mb):
                    idx = perm[start : start + u_mb]
                    abs_out = self.policy.encode_upper(im_b[idx])
                    gran_lp, gran_ent = self.policy.evaluate_granularity(
                        abs_out.z_quant, alive_n_b[idx], gran_b[idx]
                    )
                    fused_cf = self.policy.fuse_cluster_features(
                        abs_out.spatial_feat, cf_b[idx], cc_b[idx], cm_b[idx]
                    )
                    g_eval = self.policy.guidance_head.evaluate(
                        abs_out.z_quant, fused_cf, cm_b[idx], rg_b[idx]
                    )
                    new_ulp = gran_lp + g_eval.log_prob
                    ratio = torch.exp((new_ulp - old_ulp[idx]).clamp(-10.0, 10.0))
                    s1 = ratio * u_adv_norm[idx]
                    s2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * u_adv_norm[idx]
                    up_p_loss = -torch.minimum(s1, s2).mean()

                    v_pred = self.policy.value_upper(abs_out.z_quant, gs_b[idx])
                    up_v_loss = 0.5 * F.mse_loss(v_pred, u_ret_t[idx])

                    gran_oh = F.one_hot(gran_b[idx], num_classes=cfg.max_clusters).float()
                    pooled_gd = (g_eval.cluster_guidance * cm_b[idx].unsqueeze(-1)).sum(dim=1) / cm_b[idx].sum(
                        dim=1, keepdim=True
                    ).clamp_min(1.0)
                    rpred_loss, bisim_loss = self.policy.abstractor.auxiliary_losses(
                        abs_out.z_cont, abs_out.z_quant, gran_oh, pooled_gd, m_ret_b[idx], next_v_b[idx]
                    )
                    up_ent = (gran_ent + g_eval.entropy).mean()

                    total_up_loss = (
                        up_p_loss
                        + cfg.value_coef * up_v_loss
                        + cfg.commitment_beta * abs_out.commit_loss
                        + cfg.reward_pred_coef * rpred_loss
                        + cfg.bisim_coef * bisim_loss
                        + cfg.skill_diversity_coef * g_eval.diversity_loss
                        - cfg.skill_entropy_coef * up_ent
                    )

                    self.opt_upper.zero_grad(set_to_none=True)
                    self.opt_critic.zero_grad(set_to_none=True)
                    total_up_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.upper_parameters(), cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters(), cfg.max_grad_norm)
                    self.opt_upper.step()
                    self.opt_critic.step()

                    up_policy_losses.append(float(up_p_loss.item()))
                    up_value_losses.append(float(up_v_loss.item()))
                    commit_losses.append(float(abs_out.commit_loss.item()))
                    bisim_losses.append(float(bisim_loss.item()))
                    rpred_losses.append(float(rpred_loss.item()))

            # Post-update EMA prototype refresh + TD/return heterogeneity tracking + Split/Merge
            with torch.no_grad():
                abs_all = self.policy.encode_upper(im_b)
                v_all = self.policy.value_upper(abs_all.z_quant, gs_b)
                self.policy.abstractor.codebook.observe_online_stats(
                    abs_all.z_cont, abs_all.proto_id, u_td_t, m_ret_b, v_all
                )
                if self.update_count % cfg.split_merge_every == 0:
                    sm_events = self.policy.abstractor.codebook.maybe_split_and_merge(self.update_count)

        metrics = {
            "update": self.update_count,
            "steps": self.total_steps,
            "low_policy_loss": round(float(np.mean(low_policy_losses)), 5) if low_policy_losses else 0.0,
            "low_value_loss": round(float(np.mean(low_value_losses)), 5) if low_value_losses else 0.0,
            "low_entropy": round(float(np.mean(low_entropies)), 4) if low_entropies else 0.0,
            "up_policy_loss": round(float(np.mean(up_policy_losses)), 5) if up_policy_losses else 0.0,
            "up_value_loss": round(float(np.mean(up_value_losses)), 5) if up_value_losses else 0.0,
            "commit_loss": round(float(np.mean(commit_losses)), 5) if commit_losses else 0.0,
            "bisim_loss": round(float(np.mean(bisim_losses)), 5) if bisim_losses else 0.0,
            "reward_pred_loss": round(float(np.mean(rpred_losses)), 5) if rpred_losses else 0.0,
            **self.policy.abstractor.codebook.diagnostics(),
            "split_merge_events": sm_events,
        }
        return metrics, sm_events

    def save_checkpoint(self, path: Path) -> None:
        torch.save({
            "format": "dyim-chrl-v2",
            "config": self.cfg.to_dict(),
            "obs_dim": self.env.obs_dim,
            "state_dim": self.env.state_dim,
            "n_actions": self.env.n_actions,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "update_count": self.update_count,
            "policy_state_dict": self.policy.state_dict(),
            "opt_upper_state_dict": self.opt_upper.state_dict(),
            "opt_actor_state_dict": self.opt_actor.state_dict(),
            "opt_critic_state_dict": self.opt_critic.state_dict(),
            "learning_curve": self.learning_curve,
        }, path)

    def load_checkpoint(
        self,
        path: Path,
        *,
        load_optimizers: bool = True,
        load_history: bool = True,
    ) -> None:
        ckpt_path = Path(path)
        payload = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        self.policy.load_state_dict(payload["policy_state_dict"])
        self.total_steps = int(payload.get("total_steps", 0))
        self.total_episodes = int(payload.get("total_episodes", 0))
        self.update_count = int(payload.get("update_count", 0))

        if load_optimizers:
            if "opt_upper_state_dict" in payload:
                self.opt_upper.load_state_dict(payload["opt_upper_state_dict"])
            if "opt_actor_state_dict" in payload:
                self.opt_actor.load_state_dict(payload["opt_actor_state_dict"])
            if "opt_critic_state_dict" in payload:
                self.opt_critic.load_state_dict(payload["opt_critic_state_dict"])

        if load_history:
            if isinstance(payload.get("learning_curve"), list) and payload["learning_curve"]:
                self.learning_curve = list(payload["learning_curve"])
            else:
                sibling_lc = ckpt_path.parent / "learning_curve.json"
                if sibling_lc.exists():
                    try:
                        loaded_lc = json.loads(sibling_lc.read_text(encoding="utf-8"))
                        if isinstance(loaded_lc, list):
                            self.learning_curve = loaded_lc
                    except Exception:
                        pass
            # Copy prior codebook evolution log when resuming into a fresh output directory
            src_evo = ckpt_path.parent / "codebook_evolution.jsonl"
            dst_evo = self.out_dir / "codebook_evolution.jsonl"
            if src_evo.exists() and src_evo.resolve() != dst_evo.resolve() and not dst_evo.exists():
                dst_evo.write_text(src_evo.read_text(encoding="utf-8"), encoding="utf-8")

    def _run_validation_checkpoint(self, step_tag: int, *, final: bool = False) -> Tuple[dict, List[dict]]:
        if self.cfg.is_legacy_mm():
            layouts = self.eval_env.test_layouts
            opps = ("bot",)
            reps = self.cfg.legacy_final_episodes if final else self.cfg.legacy_eval_episodes
        else:
            if final:
                # Match v10/v11 final eval: standard + near layouts across all 9 opponents
                std = [l for l in self.eval_env.train_layouts if l.id == "standard"]
                layouts = tuple(std) + tuple(self.eval_env.test_layouts)
                reps = self.cfg.test_repeats
            else:
                layouts = self.eval_env.val_layouts
                reps = self.cfg.val_repeats
            opps = self.eval_opponents
        return evaluate_frozen(
            self.eval_env, self.policy, self.cfg, self.device,
            layouts=layouts, opponents=opps, repeats=reps, step_tag=step_tag,
        )

    def train(self) -> dict:
        t0 = perf_counter()
        cfg = self.cfg
        evo_path = self.out_dir / "codebook_evolution.jsonl"
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
                summary, rows = self._run_validation_checkpoint(0, final=False)
                self.learning_curve.append(summary)
                with val_eps_path.open("a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json_line(row))
                write_json(self.out_dir / "learning_curve.json", self.learning_curve)
                next_eval_at = cfg.eval_every
                # Ensure training resumes cleanly after eval on shared env
                self._cur_bundle = None

        with evo_path.open("a", encoding="utf-8") as evo_f:
            while self.total_steps < cfg.train_steps:
                remaining = cfg.train_steps - self.total_steps
                batch_steps = min(cfg.rollout_steps, remaining)
                lower_buf, upper_buf, roll_stats = self._collect_rollout(batch_steps)
                upd_metrics, _ = self._update_ppo(lower_buf, upper_buf)
                evo_f.write(json_line({**upd_metrics, **roll_stats}))
                evo_f.flush()

                if self.total_steps >= next_eval_at and next_eval_at <= cfg.train_steps:
                    summary, rows = self._run_validation_checkpoint(self.total_steps, final=False)
                    self.learning_curve.append(summary)
                    with val_eps_path.open("a", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json_line(row))
                    write_json(self.out_dir / "learning_curve.json", self.learning_curve)
                    self.save_checkpoint(self.out_dir / "checkpoint_latest.pt")
                    print(
                        f"[hrl_v2 {cfg.scenario}/{cfg.regime}] step={self.total_steps}/{cfg.train_steps} "
                        f"all9={summary['all9_win_rate']:.3f} bot={summary['bot_win_rate']} "
                        f"K_t={summary['active_prototypes']} clu={summary['mean_clusters']:.2f}",
                        flush=True,
                    )
                    next_eval_at += cfg.eval_every
                    if self.eval_env is self.env:
                        self._cur_bundle = None

        self.save_checkpoint(self.out_dir / "checkpoint_final.pt")
        final_summary, final_rows = self._run_validation_checkpoint(self.total_steps, final=True)
        with (self.out_dir / "final_eval_episodes.jsonl").open("w", encoding="utf-8") as f:
            for row in final_rows:
                f.write(json_line(row))

        elapsed = perf_counter() - t0
        report = {
            "format": "dyim-chrl-report-v2",
            "algorithm": "dyim-chrl",
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
            "algorithm": "dyim-chrl",
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
            "active_prototypes": final_summary["active_prototypes"],
            "wall_seconds": round(elapsed, 2),
        })
        return report
