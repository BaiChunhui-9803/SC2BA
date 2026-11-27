"""Dual-Agent Head-to-Head PK Arena for SC2BA without needing APyMARL.

Uses SC2BA's native dual-Agent mode:
    StarCraft2BAEnv(map_name=scenario, players=[Agent(a_race, RED), Agent(b_race, BLUE)], random_enemy=False)

Supports reciprocal Red-vs-Blue matches between ANY combination of:
  - ``dyim:<path/to/checkpoint_final.pt>``  (DyIM-CHRL, with mirrored Observation for Camp.BLUE)
  - ``qmix:<path/to/checkpoint_final.pt>``  (PyMARL QMIX checkpoint trained via hrl_v2.run_baseline)
  - ``mappo:<path/to/checkpoint_final.pt>`` (on-policy MAPPO checkpoint trained via hrl_v2.run_baseline)
  - ``sc2ba:<policy_name>``                 (SC2BA built-in frozen pool: qmix/vdn/dop/fop/qplex/coma/iql/qtran)
"""

from abc import ABC, abstractmethod
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
from io import StringIO
import itertools
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bch.observation import Observation, Unit
from bch.plan import Layout, read_plan, scenario_seeds
from bch.records import json_line, write_json

from .baselines import BaselineConfig, PyMARLQMIXModel, _build_onpolicy_args, _ensure_sys_path, _make_spaces
from .config import DEFAULT_SC2BA_PLAN, HRLv2Config
from .env_adapter import _build_unit_extra
from .influence_map import build_tactical_influence_map
from .lower_policy import DyIMCHRLPolicy
from .tactical_cluster import TacticalClusterer
from .trainer import resolve_device


@dataclass
class CampView:
    """Canonical per-camp observation bundle (Camp.BLUE already mirrored to Red perspective)."""

    camp_name: str  # "RED" | "BLUE"
    observation: Observation
    unit_obs: np.ndarray
    unit_extra: np.ndarray
    global_state: np.ndarray
    avail_actions: np.ndarray
    agent_tags: Tuple[Optional[int], ...]


def mirror_observation_for_blue(
    obs_red_view: Observation,
    *,
    center_x: float,
    center_y: float,
) -> Observation:
    """Mirror world coordinates around (center_x, center_y) and swap friendly/enemy for Camp.BLUE.

    Ensures a Blue-side DyIM-CHRL agent sees the exact same canonical left-to-right battlefield
    geometry in its 4-channel Influence Map and TacticalClusterer as SC2BA's mirrored get_obs(BLUE).
    """

    def flip_unit(u: Unit) -> Unit:
        return Unit(
            tag=u.tag,
            x=float(2.0 * center_x - u.x),
            y=float(2.0 * center_y - u.y),
            health=u.health,
            health_max=u.health_max,
            radius=u.radius,
            cooldown=u.cooldown,
            shield=u.shield,
            shield_max=u.shield_max,
            unit_type=u.unit_type,
            attack_range=u.attack_range,
            is_medivac=u.is_medivac,
            energy=u.energy,
        )

    return Observation(
        friendly=tuple(flip_unit(u) for u in obs_red_view.enemy),
        enemy=tuple(flip_unit(u) for u in obs_red_view.friendly),
        step=obs_red_view.step,
        game_loop=obs_red_view.game_loop,
        map_height=obs_red_view.map_height,
        influence_scale=obs_red_view.influence_scale,
    )


class PKAgentAdapter(ABC):
    """Unified deterministic inference interface for any contender in the PK arena."""

    label: str
    kind: str

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def select_actions(self, view: CampView, step_idx: int) -> List[int]:
        ...

    def on_step_result(self, actions: List[int], reward: float, done: bool, step_idx: int) -> None:
        pass


class DyIMCHRLPKAdapter(PKAgentAdapter):
    def __init__(self, label: str, ckpt_path: Path, device: torch.device):
        self.label = label
        self.kind = "dyim"
        self.device = device
        payload = torch.load(ckpt_path, weights_only=False, map_location=device)
        self.cfg = HRLv2Config.from_dict(payload["config"])
        self.policy = DyIMCHRLPolicy(
            self.cfg,
            obs_dim=int(payload["obs_dim"]),
            state_dim=int(payload["state_dim"]),
            n_actions=int(payload["n_actions"]),
        ).to(device)
        self.policy.load_state_dict(payload["policy_state_dict"])
        self.policy.eval()

        self.clusterer = TacticalClusterer(self.cfg)
        self._hidden: Optional[torch.Tensor] = None
        self._cached_spatial: Optional[torch.Tensor] = None
        self._cached_guidance: Optional[torch.Tensor] = None
        self._req_clusters = 1

    def reset(self) -> None:
        self.clusterer.reset()
        self._hidden = None
        self._cached_spatial = None
        self._cached_guidance = None
        self._req_clusters = 1

    @torch.inference_mode()
    def select_actions(self, view: CampView, step_idx: int) -> List[int]:
        n_agents = view.unit_obs.shape[0]
        if self._hidden is None:
            self._hidden = self.policy.init_lower_hidden(1, n_agents, self.device)

        im_grid = build_tactical_influence_map(
            view.observation, grid_size=self.cfg.im_size, channels=self.cfg.im_channels
        )
        is_macro = (step_idx % self.cfg.macro_interval == 0) or (self._cached_spatial is None)
        if is_macro:
            im_t = torch.from_numpy(im_grid.tensor).unsqueeze(0).to(self.device)
            abs_out = self.policy.encode_upper(im_t)
            alive_t = torch.tensor([len(view.observation.friendly)], dtype=torch.long, device=self.device)
            gran_idx, _, _ = self.policy.sample_granularity(abs_out.z_quant, alive_t, deterministic=True)
            self._req_clusters = int(gran_idx.item()) + 1
            assign = self.clusterer.assign(
                view.observation, view.agent_tags, im_grid,
                requested_clusters=self._req_clusters, recluster=True,
            )
            c_feats_t = torch.from_numpy(assign.cluster_feats).unsqueeze(0).to(self.device)
            c_coords_t = torch.from_numpy(assign.cluster_norm_coords).unsqueeze(0).to(self.device)
            c_mask_t = torch.from_numpy(assign.cluster_mask).unsqueeze(0).to(self.device)
            fused_cf = self.policy.fuse_cluster_features(abs_out.spatial_feat, c_feats_t, c_coords_t, c_mask_t)
            g_out = self.policy.guidance_head.sample(abs_out.z_quant, fused_cf, c_mask_t, deterministic=True)
            self._cached_spatial = abs_out.spatial_feat
            self._cached_guidance = g_out.cluster_guidance
        else:
            assign = self.clusterer.assign(
                view.observation, view.agent_tags, im_grid,
                requested_clusters=self._req_clusters, recluster=False,
            )
            c_feats_t = torch.from_numpy(assign.cluster_feats).unsqueeze(0).to(self.device)
            c_coords_t = torch.from_numpy(assign.cluster_norm_coords).unsqueeze(0).to(self.device)
            c_mask_t = torch.from_numpy(assign.cluster_mask).unsqueeze(0).to(self.device)
            fused_cf = self.policy.fuse_cluster_features(self._cached_spatial, c_feats_t, c_coords_t, c_mask_t)

        actions_t, _, _, _, self._hidden = self.policy.act_lower(
            torch.from_numpy(view.unit_obs).unsqueeze(0).to(self.device),
            torch.from_numpy(view.unit_extra).unsqueeze(0).to(self.device),
            torch.from_numpy(assign.unit_rel_cluster).unsqueeze(0).to(self.device),
            fused_cf,
            self._cached_guidance,
            torch.from_numpy(assign.unit_to_cluster).unsqueeze(0).to(self.device),
            self._hidden,
            torch.from_numpy(view.avail_actions).unsqueeze(0).to(self.device),
            torch.from_numpy(view.global_state).unsqueeze(0).to(self.device),
            deterministic=True,
        )
        return actions_t[0].cpu().tolist()


class PyMARLQMIXPKAdapter(PKAgentAdapter):
    def __init__(self, label: str, ckpt_path: Path, device: torch.device):
        self.label = label
        self.kind = "qmix"
        payload = torch.load(ckpt_path, weights_only=False, map_location=device)
        cfg = BaselineConfig(**{k: v for k, v in payload["config"].items() if k in BaselineConfig.__dataclass_fields__})
        self.model = PyMARLQMIXModel(
            cfg,
            n_agents=int(payload["n_agents"]),
            obs_dim=int(payload["obs_dim"]),
            state_dim=int(payload["state_dim"]),
            n_actions=int(payload["n_actions"]),
            device=device,
        )
        self.model.agent.load_state_dict(payload["agent_state_dict"])
        self.model.mixer.load_state_dict(payload["mixer_state_dict"])
        self.model.eval_mode()
        self._hidden: Optional[torch.Tensor] = None
        self._last_act_oh: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._hidden = self.model.init_hidden(1)
        self._last_act_oh = np.zeros((self.model.n_agents, self.model.n_actions), dtype=np.float32)

    @torch.inference_mode()
    def select_actions(self, view: CampView, step_idx: int) -> List[int]:
        if self._hidden is None or self._last_act_oh is None:
            self.reset()
        actions, self._hidden = self.model.select_actions(
            view.unit_obs, view.avail_actions, self._last_act_oh, self._hidden, epsilon=0.0
        )
        self._last_act_oh = np.eye(self.model.n_actions, dtype=np.float32)[np.asarray(actions, dtype=np.int64)]
        return actions


class OnPolicyMAPPOPKAdapter(PKAgentAdapter):
    def __init__(self, label: str, ckpt_path: Path, device: torch.device):
        self.label = label
        self.kind = "mappo"
        payload = torch.load(ckpt_path, weights_only=False, map_location=device)
        cfg = BaselineConfig(**{k: v for k, v in payload["config"].items() if k in BaselineConfig.__dataclass_fields__})
        _ensure_sys_path([Path(cfg.onpolicy_root)])
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

        self.n_agents = int(payload["n_agents"])
        obs_sp, cent_sp, act_sp = _make_spaces(
            int(payload["obs_dim"]), int(payload["state_dim"]), int(payload["n_actions"])
        )
        self.op_args = _build_onpolicy_args(cfg, episode_length=cfg.mappo_rollout_steps, data_chunk_length=10)
        self.policy = R_MAPPOPolicy(self.op_args, obs_sp, cent_sp, act_sp, device=device)
        self.policy.actor.load_state_dict(payload["actor_state_dict"])
        self.policy.critic.load_state_dict(payload["critic_state_dict"])
        self.policy.actor.eval()
        self.policy.critic.eval()
        self._rnn_a: Optional[np.ndarray] = None
        self._masks: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._rnn_a = np.zeros(
            (self.n_agents, self.op_args.recurrent_N, self.op_args.hidden_size), dtype=np.float32
        )
        self._masks = np.ones((self.n_agents, 1), dtype=np.float32)

    @torch.inference_mode()
    def select_actions(self, view: CampView, step_idx: int) -> List[int]:
        if self._rnn_a is None or self._masks is None:
            self.reset()
        act_t, rnn_a_t = self.policy.act(
            view.unit_obs, self._rnn_a, self._masks, view.avail_actions, deterministic=True
        )
        self._rnn_a = rnn_a_t.cpu().numpy()
        return act_t.cpu().numpy().reshape(-1).astype(int).tolist()


class SC2BAPoolPKAdapter(PKAgentAdapter):
    """Wraps SC2BA's built-in frozen EnemyControl models (qmix/vdn/dop/fop/qplex/coma/iql/qtran)."""

    def __init__(self, label: str, policy_name: str, scenario: str, env_info: dict):
        from sc2ba.enemycontrol.enemy_control import EnemyControl, enemy_name

        self.label = label
        self.kind = "sc2ba"
        if policy_name not in enemy_name or enemy_name[policy_name] == 0:
            raise ValueError(f"Unknown SC2BA built-in pool policy '{policy_name}'")
        self._code = int(enemy_name[policy_name])
        with redirect_stdout(StringIO()):
            self._ctrl = EnemyControl(env_info, scenario)

    def reset(self) -> None:
        self._ctrl.reset(self._code)

    @torch.inference_mode()
    def select_actions(self, view: CampView, step_idx: int) -> List[int]:
        self._ctrl.batch.update({
            "obs": view.unit_obs[None, ...],
            "avail_actions": view.avail_actions.astype(np.int32)[None, ...],
        }, ts=step_idx)
        return self._ctrl.mac.select_actions(self._ctrl.batch, t_ep=step_idx, test_mode=True)[0].tolist()

    def on_step_result(self, actions: List[int], reward: float, done: bool, step_idx: int) -> None:
        self._ctrl.update_action_reward(actions, reward, done, step_idx)


def build_pk_adapter(
    label: str,
    spec: str,
    *,
    scenario: str,
    device: torch.device,
    env_info: Optional[dict] = None,
) -> PKAgentAdapter:
    if ":" not in spec:
        raise ValueError(f"Agent spec must be '<kind>:<path_or_name>', got '{spec}'")
    kind, target = spec.split(":", 1)
    kind = kind.strip().lower()
    target = target.strip()
    if kind == "dyim":
        return DyIMCHRLPKAdapter(label, Path(target), device)
    if kind == "qmix":
        return PyMARLQMIXPKAdapter(label, Path(target), device)
    if kind == "mappo":
        return OnPolicyMAPPOPKAdapter(label, Path(target), device)
    if kind == "sc2ba":
        if env_info is None:
            raise ValueError("env_info required to instantiate sc2ba built-in pool adapter")
        return SC2BAPoolPKAdapter(label, target, scenario, env_info)
    raise ValueError(f"Unsupported agent kind '{kind}' (use dyim, qmix, mappo, or sc2ba)")


class SC2BADualPKArena:
    """Runs symmetric dual-Agent Red vs Blue matches natively inside SC2BA."""

    def __init__(self, scenario: str, seed: int, step_mul: int = 8, plan_path: Path = DEFAULT_SC2BA_PLAN):
        from absl import flags
        from sc2ba.env import Agent, Camp, StarCraft2BAEnv
        from sc2ba.env.starcraft2.maps import get_map_params
        from bch.arena import UNIT_ATTACK_RANGE

        self._Camp = Camp
        self._UNIT_ATTACK_RANGE = UNIT_ATTACK_RANGE
        self.scenario = scenario

        document, available, _, _ = read_plan(plan_path)
        if scenario not in available:
            raise ValueError(f"Scenario '{scenario}' not found in {plan_path}")
        sc = available[scenario]
        self.layouts = tuple(sc.layouts)
        engine_seed, _, _ = scenario_seeds(seed, document, sc.name)

        params = get_map_params(scenario)
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["hrl_v2_pk"])

        self._env = StarCraft2BAEnv(
            map_name=scenario,
            players=[Agent(params["a_race"], Camp.RED.name), Agent(params["b_race"], Camp.BLUE.name)],
            seed=engine_seed,
            step_mul=step_mul,
            random_enemy=False,
            window_size_x=320,
            window_size_y=240,
        )
        self._default_blue = self._env.blue_start_position
        self._default_mirror = self._env.mirror_create_blue_camp
        self._default_blue_units = dict(params["blue_units"])
        self._initial_scale = 16 * max(self._env.n_red_agents, self._env.n_blue_agents)
        self.env_info = self._env.get_env_info()

    def _read_observation_red(self) -> Tuple[Observation, Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        env = self._env

        def read_side(side):
            out = []
            for unit in sorted(side.values(), key=lambda item: item.tag):
                if unit.health <= 0:
                    continue
                utype = env.get_unit_type_id(unit) or 1
                out.append(
                    Unit(
                        int(unit.tag),
                        float(unit.pos.x),
                        float(unit.pos.y),
                        float(unit.health),
                        float(unit.health_max),
                        float(unit.radius),
                        float(unit.weapon_cooldown),
                        shield=float(getattr(unit, "shield", 0.0)),
                        shield_max=float(getattr(unit, "shield_max", 0.0)),
                        unit_type=int(utype),
                        attack_range=float(self._UNIT_ATTACK_RANGE.get(utype, 5.0)),
                        is_medivac=(utype == 3),
                        energy=float(getattr(unit, "energy", 0.0)),
                    )
                )
            return tuple(out)

        red_tags = tuple(
            int(env.agents[idx].tag) if idx in env.agents and env.agents[idx].health > 0 else None
            for idx in range(env.n_red_agents)
        )
        blue_tags = tuple(
            int(env.enemies[idx].tag) if idx in env.enemies and env.enemies[idx].health > 0 else None
            for idx in range(env.n_blue_agents)
        )
        obs_red = Observation(
            read_side(env.agents),
            read_side(env.enemies),
            int(env.episode_steps),
            int(env._obs.observation.game_loop),
            int(env.map_y),
            float(self._initial_scale),
        )
        return obs_red, red_tags, blue_tags

    def _build_views(self) -> Tuple[CampView, CampView]:
        env = self._env
        Camp = self._Camp
        obs_red, red_tags, blue_tags = self._read_observation_red()
        obs_blue = mirror_observation_for_blue(obs_red, center_x=float(env.center_x), center_y=float(env.center_y))

        red_by_tag = {u.tag: u for u in obs_red.friendly}
        blue_by_tag = {u.tag: u for u in obs_blue.friendly}

        red_view = CampView(
            camp_name="RED",
            observation=obs_red,
            unit_obs=np.asarray(env.get_obs(Camp.RED), dtype=np.float32),
            unit_extra=_build_unit_extra(env.n_red_agents, red_tags, red_by_tag),
            global_state=np.asarray(env.get_state(Camp.RED), dtype=np.float32),
            avail_actions=np.asarray(env.get_avail_actions(Camp.RED), dtype=np.float32),
            agent_tags=red_tags,
        )
        blue_view = CampView(
            camp_name="BLUE",
            observation=obs_blue,
            unit_obs=np.asarray(env.get_obs(Camp.BLUE), dtype=np.float32),
            unit_extra=_build_unit_extra(env.n_blue_agents, blue_tags, blue_by_tag),
            global_state=np.asarray(env.get_state(Camp.BLUE), dtype=np.float32),
            avail_actions=np.asarray(env.get_avail_actions(Camp.BLUE), dtype=np.float32),
            agent_tags=blue_tags,
        )
        return red_view, blue_view

    def mirror_audit(self) -> dict:
        env = self._env
        red_pts = [(float(u.pos.x), float(u.pos.y)) for u in env.agents.values() if u.health > 0]
        blue_pts = [(float(u.pos.x), float(u.pos.y)) for u in env.enemies.values() if u.health > 0]
        expected_blue = [(2.0 * env.center_x - x, 2.0 * env.center_y - y) for x, y in red_pts]
        # Nearest-neighbor mirror error (works for any unit count without factorial blowup)
        errs = [
            min(math.hypot(ex - bx, ey - by) for bx, by in blue_pts)
            for ex, ey in expected_blue
        ] if red_pts and blue_pts else [0.0]
        return {
            "center": [float(env.center_x), float(env.center_y)],
            "red_units": len(red_pts),
            "blue_units": len(blue_pts),
            "max_mirror_pos_error": round(float(max(errs)), 4),
        }

    def play_match(
        self,
        red_agent: PKAgentAdapter,
        blue_agent: PKAgentAdapter,
        layout: Layout,
        *,
        audit_mirror: bool = False,
    ) -> Tuple[dict, Optional[dict]]:
        env = self._env
        env.red_start_position = layout.red or env.red_start_position
        # For true symmetric PK, mirror Blue spawn when layout.blue is None
        env.blue_start_position = layout.blue or self._default_blue
        env.mirror_create_blue_camp = self._default_mirror and layout.blue is None
        env.blue_units_config = dict(self._default_blue_units)

        with redirect_stdout(StringIO()):
            env.reset()
        red_agent.reset()
        blue_agent.reset()
        audit_info = self.mirror_audit() if audit_mirror else None

        red_ret = 0.0
        blue_ret = 0.0
        step_idx = 0
        info = {}

        while step_idx < env.episode_limit:
            red_view, blue_view = self._build_views()
            red_acts = red_agent.select_actions(red_view, step_idx)
            blue_acts = blue_agent.select_actions(blue_view, step_idx)

            # Enforce strict action legality audit
            for i, a in enumerate(red_acts):
                if red_view.avail_actions[i, a] <= 0.5:
                    raise RuntimeError(f"Illegal RED action {a} from {red_agent.label} at step {step_idx}")
            for i, a in enumerate(blue_acts):
                if blue_view.avail_actions[i, a] <= 0.5:
                    raise RuntimeError(f"Illegal BLUE action {a} from {blue_agent.label} at step {step_idx}")

            (r_rew, b_rew), done, info = env.step(list(red_acts) + list(blue_acts))
            red_ret += float(r_rew)
            blue_ret += float(b_rew)
            red_agent.on_step_result(red_acts, float(r_rew), bool(done), step_idx)
            blue_agent.on_step_result(blue_acts, float(b_rew), bool(done), step_idx)
            step_idx += 1
            if done:
                break

        obs_red, _, _ = self._read_observation_red()
        red_hp = float(sum(u.effective_health for u in obs_red.friendly))
        blue_hp = float(sum(u.effective_health for u in obs_red.enemy))
        outcome = (
            "win" if info.get("red_battle_won")
            else "loss" if info.get("blue_battle_won")
            else "timeout" if (obs_red.friendly and obs_red.enemy)
            else "draw"
        )
        row = {
            "scenario": self.scenario,
            "layout": layout.id,
            "red": red_agent.label,
            "blue": blue_agent.label,
            "red_result": outcome,
            "steps": step_idx,
            "red_return": round(red_ret, 4),
            "blue_return": round(blue_ret, 4),
            "red_hp": round(red_hp, 2),
            "blue_hp": round(blue_hp, 2),
            "hp_margin": round(red_hp - blue_hp, 2),
        }
        return row, audit_info

    def close(self) -> None:
        self._env.close()


def aggregate_pk_results(
    labels: Sequence[str],
    matches: Sequence[dict],
) -> dict:
    """Aggregate reciprocal Red/Blue PK results into a symmetric win-rate matrix and ranking."""
    totals = Counter()
    hp_sum: Dict[Tuple[str, str], float] = Counter()
    hp_cnt: Dict[Tuple[str, str], int] = Counter()

    for m in matches:
        r, b, res = m["red"], m["blue"], m["red_result"]
        totals[(r, b, res)] += 1
        hp_sum[(r, b)] += float(m["hp_margin"])
        hp_cnt[(r, b)] += 1

    pairwise = []
    matrix = {a: {} for a in labels}
    for a in labels:
        for b in labels:
            if a == b:
                matrix[a][b] = None
                continue
            r_w = totals[(a, b, "win")]
            r_l = totals[(a, b, "loss")]
            r_d = totals[(a, b, "draw")] + totals[(a, b, "timeout")]
            b_w = totals[(b, a, "loss")]
            b_l = totals[(b, a, "win")]
            b_d = totals[(b, a, "draw")] + totals[(b, a, "timeout")]
            games = r_w + r_l + r_d + b_w + b_l + b_d
            wins = r_w + b_w
            losses = r_l + b_l
            draws = r_d + b_d
            wr = wins / games if games else 0.0
            mean_hp = (
                (hp_sum[(a, b)] - hp_sum[(b, a)]) / max(hp_cnt[(a, b)] + hp_cnt[(b, a)], 1)
            )
            matrix[a][b] = round(wr, 4)
            pairwise.append({
                "policy": a,
                "opponent": b,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": round(wr, 4),
                "red_wins": r_w,
                "red_games": r_w + r_l + r_d,
                "blue_wins": b_w,
                "blue_games": b_w + b_l + b_d,
                "mean_hp_margin": round(mean_hp, 2),
            })

    ranking = sorted(
        [
            {
                "policy": a,
                "mean_win_rate": round(
                    float(np.mean([matrix[a][b] for b in labels if b != a])) if len(labels) > 1 else 0.0, 4
                ),
            }
            for a in labels
        ],
        key=lambda x: (-x["mean_win_rate"], x["policy"]),
    )
    return {"labels": list(labels), "matrix": matrix, "pairwise": pairwise, "ranking": ranking}


def _isolate_temp_dir(tag: str) -> None:
    tmp_dir = Path(tempfile.gettempdir()) / f"sc2ba_hrl_v2_pk_{tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Run reciprocal head-to-head dual-agent PK in SC2BA.")
    p.add_argument("--scenario", required=True, help="SC2BA scenario name (e.g. 3m, 2s3z)")
    p.add_argument(
        "--agent",
        action="append",
        required=True,
        help="Contender spec '<label>=<kind>:<path_or_name>' (kinds: dyim, qmix, mappo, sc2ba). Pass >=2 times.",
    )
    p.add_argument("--layouts", default="standard,offset,diagonal,near", help="Comma-separated layout IDs to evaluate")
    p.add_argument("--repeats", type=int, default=5, help="Repeats per (layout, ordered red-vs-blue pair)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.agent) < 2:
        raise ValueError("Pass at least two --agent specifications for head-to-head PK.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _isolate_temp_dir(f"{args.scenario}_{out_dir.name}")

    device = resolve_device(args.device)
    t0 = perf_counter()
    arena = SC2BADualPKArena(args.scenario, seed=args.seed)
    try:
        wanted = {s.strip() for s in args.layouts.split(",") if s.strip()}
        layouts = [l for l in arena.layouts if l.id in wanted]
        if not layouts:
            layouts = list(arena.layouts)

        adapters: Dict[str, PKAgentAdapter] = {}
        specs_meta: Dict[str, str] = {}
        for raw in args.agent:
            if "=" not in raw:
                raise ValueError(f"Expected --agent label=kind:target, got '{raw}'")
            label, spec = raw.split("=", 1)
            label = label.strip()
            adapters[label] = build_pk_adapter(
                label, spec, scenario=args.scenario, device=device, env_info=arena.env_info
            )
            specs_meta[label] = spec

        labels = list(adapters.keys())
        ordered_pairs = list(itertools.permutations(labels, 2))
        matches: List[dict] = []
        mirror_audit = None
        matches_path = out_dir / "pk_matches.jsonl"

        with matches_path.open("w", encoding="utf-8") as f:
            for idx, (red_lbl, blue_lbl) in enumerate(ordered_pairs):
                for layout in layouts:
                    for rep in range(args.repeats):
                        row, audit = arena.play_match(
                            adapters[red_lbl],
                            adapters[blue_lbl],
                            layout,
                            audit_mirror=(mirror_audit is None),
                        )
                        if audit is not None:
                            mirror_audit = audit
                        row["repeat"] = rep
                        matches.append(row)
                        f.write(json_line(row))
                f.flush()
                print(
                    f"[PK {args.scenario}] completed {idx + 1}/{len(ordered_pairs)}: "
                    f"{red_lbl} (RED) vs {blue_lbl} (BLUE)",
                    flush=True,
                )
    finally:
        arena.close()

    agg = aggregate_pk_results(labels, matches)
    elapsed = round(perf_counter() - t0, 2)
    payload = {
        "format": "hrl-v2-pk-matrix-v1",
        "scenario": args.scenario,
        "seed": args.seed,
        "layouts": [l.id for l in layouts],
        "repeats_per_side": args.repeats,
        "total_matches": len(matches),
        "wall_seconds": elapsed,
        "mirror_audit": mirror_audit,
        "agents": specs_meta,
        **agg,
    }
    write_json(out_dir / "pk_matrix.json", payload)
    write_json(out_dir / "done.json", {
        "scenario": args.scenario,
        "total_matches": len(matches),
        "ranking": agg["ranking"],
        "wall_seconds": elapsed,
    })
    with (out_dir / "pk_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy", "opponent", "games", "wins", "losses", "draws",
                "win_rate", "red_wins", "red_games", "blue_wins", "blue_games", "mean_hp_margin",
            ],
        )
        writer.writeheader()
        writer.writerows(agg["pairwise"])

    print(f"[DONE PK {args.scenario}] {len(matches)} matches in {elapsed}s -> {out_dir}", flush=True)
    for r in agg["ranking"]:
        print(f"  {r['policy']:24s} mean_WR = {r['mean_win_rate'] * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
