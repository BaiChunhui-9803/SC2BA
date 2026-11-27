"""Unified atomic-action environment adapters for SC2BA and legacy MM4m/MM8 maps.

Guarantees 100% physical isolation from ``bch/`` (no mutations to bch code or runs),
while reusing the exact v10/v11 scenario plans, layouts, opponents, and eval semantics.
"""

from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bch.observation import Observation, Unit
from bch.plan import Layout, Scenario, read_plan, scenario_seeds

from .config import HRLv2Config, LEGACY_MM_SCENARIOS


@dataclass
class StepBundle:
    """Unified single-step data shared across SC2BA, legacy MM, and mock test envs."""

    observation: Observation          # Structured alive units for IM & clustering
    unit_obs: np.ndarray              # [N_agents, obs_dim] float32
    unit_extra: np.ndarray            # [N_agents, 4] float32 (cooldown_norm, hp_ratio, is_medivac, alive)
    global_state: np.ndarray          # [state_dim] float32
    avail_actions: np.ndarray         # [N_agents, n_actions] float32 (1=legal, 0=illegal)
    agent_tags: Tuple[Optional[int], ...]  # Slot -> SC2 unit tag (None if dead)
    reward: float
    done: bool
    info: dict


class BaseMicroEnv(ABC):
    """Unified environment interface for DyIM-CHRL training and frozen evaluation."""

    scenario_name: str
    n_agents: int
    n_enemies: int
    obs_dim: int
    state_dim: int
    n_actions: int
    episode_limit: int
    opponents: Tuple[str, ...]
    train_layouts: Tuple[Layout, ...]
    val_layouts: Tuple[Layout, ...]
    test_layouts: Tuple[Layout, ...]

    @abstractmethod
    def reset_round(self, layout: Optional[Layout], opponent: str) -> StepBundle:
        ...

    @abstractmethod
    def step_unit_actions(self, actions: Sequence[int]) -> StepBundle:
        ...

    @abstractmethod
    def episode_details(self, first: Observation, final: Observation, info: dict) -> dict:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _build_unit_extra(
    n_agents: int,
    agent_tags: Sequence[Optional[int]],
    friendly_by_tag: Dict[int, Unit],
) -> np.ndarray:
    """Per-agent micro cues explicitly exposed to the lower GRU actor."""
    extra = np.zeros((n_agents, 4), dtype=np.float32)
    for idx in range(n_agents):
        tag = agent_tags[idx]
        unit = friendly_by_tag.get(tag) if tag is not None else None
        if unit is None:
            continue
        extra[idx, 0] = float(np.clip(unit.cooldown / 35.0, 0.0, 1.0))
        extra[idx, 1] = float(unit.effective_health / max(unit.effective_health_max, 1.0))
        extra[idx, 2] = 1.0 if unit.is_medivac else 0.0
        extra[idx, 3] = 1.0
    return extra


class SC2BAUnitEnv(BaseMicroEnv):
    """SC2BA native atomic-action environment (zero manual macros, full opponent pool)."""

    def __init__(self, cfg: HRLv2Config, scenario: Scenario, engine_seed: int):
        from absl import flags
        from sc2ba.enemycontrol.enemy_control import enemy_name
        from sc2ba.env import Agent, Bot, Camp, StarCraft2BAEnv, difficulties
        from sc2ba.env.starcraft2.maps import get_map_params
        from bch.arena import UNIT_ATTACK_RANGE

        self._Camp = Camp
        self._UNIT_ATTACK_RANGE = UNIT_ATTACK_RANGE
        self._OPPONENTS = {"bot": 0, **{name: code for name, code in enemy_name.items() if code}}

        self.scenario_name = scenario.name
        self.opponents = scenario.opponents
        self.train_layouts = tuple(l for l in scenario.layouts if l.split == "train")
        self.val_layouts = tuple(l for l in scenario.layouts if l.split == "validation")
        self.test_layouts = tuple(l for l in scenario.layouts if l.split == "test")

        params = get_map_params(scenario.name)
        allowed_ids = params.get("blue_control_models", [0])
        for opp in scenario.opponents:
            if self._OPPONENTS.get(opp) not in allowed_ids:
                raise ValueError(f"Opponent {opp} unavailable in SC2BA scenario {scenario.name}.")

        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["hrl_v2"])

        players = [
            Agent(params["a_race"], Camp.RED.name),
            Bot(params["b_race"], Camp.BLUE.name, difficulties["7"]),
        ]
        use_models = any(self._OPPONENTS[name] for name in scenario.opponents)
        self._env = StarCraft2BAEnv(
            map_name=scenario.name,
            players=players,
            seed=engine_seed,
            step_mul=cfg.step_mul,
            random_enemy=use_models,
            window_size_x=320,
            window_size_y=240,
        )
        self._default_blue = self._env.blue_start_position
        self._default_mirror = self._env.mirror_create_blue_camp
        self._default_blue_units = dict(params["blue_units"])
        self._initial_scale = 16 * max(self._env.n_red_agents, self._env.n_blue_agents)
        self._last_end_loop = None

        self.n_agents = int(self._env.n_red_agents)
        self.n_enemies = int(self._env.n_blue_agents)
        self.obs_dim = int(self._env.get_obs_size(Camp.RED))
        self.state_dim = int(self._env.get_state_size(Camp.RED))
        self.n_actions = int(self._env.n_red_actions)
        self.episode_limit = int(self._env.episode_limit)

    def _observe_battle(self) -> Tuple[Observation, Tuple[Optional[int], ...]]:
        env = self._env

        def read_side(side):
            result = []
            for unit in sorted(side.values(), key=lambda item: item.tag):
                if unit.health <= 0:
                    continue
                utype = env.get_unit_type_id(unit) or 1
                result.append(
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
            return tuple(result)

        agent_tags = tuple(
            int(env.agents[idx].tag) if idx in env.agents and env.agents[idx].health > 0 else None
            for idx in range(self.n_agents)
        )
        obs = Observation(
            read_side(env.agents),
            read_side(env.enemies),
            int(env.episode_steps),
            int(env._obs.observation.game_loop),
            int(env.map_y),
            float(self._initial_scale),
        )
        return obs, agent_tags

    def _make_bundle(self, reward: float, done: bool, info: dict) -> StepBundle:
        env = self._env
        Camp = self._Camp
        obs, agent_tags = self._observe_battle()
        unit_obs = np.asarray(env.get_obs(Camp.RED), dtype=np.float32)
        global_state = np.asarray(env.get_state(Camp.RED), dtype=np.float32)
        avail = np.asarray(env.get_avail_actions(Camp.RED), dtype=np.float32)
        friendly_by_tag = {u.tag: u for u in obs.friendly}
        unit_extra = _build_unit_extra(self.n_agents, agent_tags, friendly_by_tag)
        return StepBundle(
            observation=obs,
            unit_obs=unit_obs,
            unit_extra=unit_extra,
            global_state=global_state,
            avail_actions=avail,
            agent_tags=agent_tags,
            reward=float(reward),
            done=bool(done),
            info=dict(info),
        )

    def reset_round(self, layout: Optional[Layout], opponent: str) -> StepBundle:
        env = self._env
        env.random_enemy_seq = [self._OPPONENTS[opponent]]
        if layout is not None:
            env.red_start_position = layout.red or env.red_start_position
            env.blue_start_position = layout.blue or self._default_blue
            env.mirror_create_blue_camp = self._default_mirror and layout.blue is None
        env.blue_units_config = dict(self._default_blue_units)
        restarts = env.force_restarts
        with redirect_stdout(StringIO()):
            env.reset()
        if env.force_restarts != restarts or (env.enemy_code or 0) != self._OPPONENTS[opponent]:
            raise RuntimeError("SC2BA reset did not produce the requested opponent match.")
        bundle = self._make_bundle(reward=0.0, done=False, info={})
        if self._last_end_loop is not None and bundle.observation.game_loop <= self._last_end_loop:
            raise RuntimeError("SC2 loaded a fresh game instead of fast-reusing the map.")
        return bundle

    @torch.inference_mode()
    def step_unit_actions(self, actions: Sequence[int]) -> StepBundle:
        env = self._env
        (red_reward, _blue_reward), done, info = env.step([int(a) for a in actions])
        if "dead_red" not in info and not done:
            raise RuntimeError("SC2BA did not return a normal step.")
        bundle = self._make_bundle(reward=float(red_reward), done=bool(done), info=info)
        if done:
            self._last_end_loop = bundle.observation.game_loop
        return bundle

    def episode_details(self, first: Observation, final: Observation, info: dict) -> dict:
        outcome = (
            "win" if info.get("red_battle_won")
            else "loss" if info.get("blue_battle_won")
            else "timeout" if final.friendly and final.enemy
            else "draw"
        )
        return {
            "outcome": outcome,
            "steps": int(final.step),
            "step_limit": int(self.episode_limit),
            "final_game_loop": int(final.game_loop),
            "final_hp": [
                float(sum(u.effective_health for u in side))
                for side in (final.friendly, final.enemy)
            ],
            "hp_margin": float(final.margin),
            "engine_restarts": int(self._env.force_restarts),
        }

    def close(self) -> None:
        self._env.close()


class LegacyMMUnitEnv(BaseMicroEnv):
    """Unified atomic-action adapter for legacy HRL-IMCBS maps (MM4m, MM8, etc.).

    Reuses HRL-IMCBS-test's ``FastResetSC2Env`` (280x faster episode reset) without
    modifying a single line in the legacy repo, and exposes the exact same SMAC-style
    per-unit atomic action space:
      0: no-op (dead only)
      1: stop
      2..5: move N / S / E / W (+/- move_amount)
      6..6+N_enemies-1: attack enemy slot j (masked by shoot_range & alive)
    """

    def __init__(self, cfg: HRLv2Config, engine_seed: int):
        import sys
        from absl import flags
        from pysc2.env import sc2_env
        from pysc2.lib import actions as pysc2_actions
        from pysc2.lib import features as pysc2_features
        from pysc2.lib import units as pysc2_units

        spec = LEGACY_MM_SCENARIOS[cfg.scenario]
        self._actions = pysc2_actions
        self._features = pysc2_features
        self._marine_type = int(pysc2_units.Terran.Marine)

        legacy_repo = Path(cfg.legacy_map_root).resolve().parent
        if str(legacy_repo) not in sys.path:
            sys.path.insert(0, str(legacy_repo))
        from pysc2_compat import install_pysc2_compat
        from sc2_fast_reset import FastResetSC2Env, ScenarioMap

        install_pysc2_compat()
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(["hrl_v2"])

        self.scenario_name = cfg.scenario
        self.n_agents = int(spec["n_units"])
        self.n_enemies = int(spec["n_units"])
        self.episode_limit = int(cfg.legacy_episode_limit) if cfg.legacy_episode_limit is not None else int(spec["limit"])
        self.n_actions = 6 + self.n_enemies
        self.move_amount = 2.0
        self.shoot_range = 6.0
        self.sight_range = 12.0
        self.map_size = 128.0
        self._initial_scale = 16.0 * self.n_agents
        self.opponents = ("bot",)

        dummy = Layout(id="standard", split="train", red=(0.0, 0.0), blue=None, red_units=(), blue_units=())
        self.train_layouts = (dummy,)
        self.val_layouts = (dummy,)
        self.test_layouts = (dummy,)

        # Match SMAC observation layout: 4 move + N_enemy * 5 + (N_agent - 1) * 5 + 2 own
        self.obs_dim = 4 + self.n_enemies * 5 + (self.n_agents - 1) * 5 + 2
        # Global state: N_agent * 5 + N_enemy * 4 + 1 timestep
        self.state_dim = self.n_agents * 5 + self.n_enemies * 4 + 1

        game_map = ScenarioMap(spec["map_name"], spec["filename"], fast_reset=cfg.legacy_fast_reset)
        self._env = FastResetSC2Env(
            map_name=game_map,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_hard),
            ],
            agent_interface_format=pysc2_features.AgentInterfaceFormat(
                action_space=pysc2_actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=128,
            ),
            step_mul=cfg.step_mul,
            game_steps_per_episode=self.episode_limit * cfg.step_mul,
            unit_type=self._marine_type,
            units_per_side=self.n_agents,
            fixed_layout=False,
            disable_fog=True,
            random_seed=engine_seed,
            visualize=False,
        )
        self._slot_ally_tags: List[int] = []
        self._slot_enemy_tags: List[int] = []
        self._prev_ally_hp: Dict[int, float] = {}
        self._prev_enemy_hp: Dict[int, float] = {}
        self._ep_steps = 0
        self._max_reward = self.n_enemies * (45.0 + 10.0) + 200.0
        self._last_ts = None

    def _extract_units(self, ts) -> Tuple[Dict[int, Unit], Dict[int, Unit], int]:
        raw_units = ts.observation.raw_units
        self_rel = self._features.PlayerRelative.SELF
        enemy_rel = self._features.PlayerRelative.ENEMY
        allies: Dict[int, Unit] = {}
        enemies: Dict[int, Unit] = {}
        for u in raw_units:
            if int(u.unit_type) != self._marine_type or float(u.health) <= 0:
                continue
            unit = Unit(
                tag=int(u.tag),
                x=float(u.x),
                y=float(u.y),
                health=float(u.health),
                health_max=max(float(getattr(u, "health_max", 45.0)), 45.0),
                radius=float(getattr(u, "radius", 0.375)) if hasattr(u, "radius") else 0.375,
                cooldown=float(getattr(u, "weapon_cooldown", 0.0)),
                shield=0.0,
                shield_max=0.0,
                unit_type=1,
                attack_range=5.0,
                is_medivac=False,
                energy=0.0,
            )
            if int(u.alliance) == self_rel:
                allies[unit.tag] = unit
            elif int(u.alliance) == enemy_rel:
                enemies[unit.tag] = unit
        game_loop = int(ts.observation.game_loop[0])
        return allies, enemies, game_loop

    def _build_step_bundle(self, ts, reward: float, done: bool, info: dict) -> StepBundle:
        allies, enemies, game_loop = self._extract_units(ts)
        obs = Observation(
            friendly=tuple(allies[t] for t in self._slot_ally_tags if t in allies),
            enemy=tuple(enemies[t] for t in self._slot_enemy_tags if t in enemies),
            step=self._ep_steps,
            game_loop=game_loop,
            map_height=int(self.map_size),
            influence_scale=float(self._initial_scale),
        )
        agent_tags = tuple(t if t in allies else None for t in self._slot_ally_tags)

        # Build SMAC-compatible per-agent obs, avail_actions, and global_state
        unit_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        all_coords = [(u.x, u.y) for u in obs.friendly + obs.enemy]
        cx = float(np.mean([p[0] for p in all_coords])) if all_coords else 64.0
        cy = float(np.mean([p[1] for p in all_coords])) if all_coords else 64.0

        for i, tag in enumerate(self._slot_ally_tags):
            u = allies.get(tag)
            if u is None:
                avail[i, 0] = 1.0  # no-op only when dead
                continue
            avail[i, 1] = 1.0  # stop
            can_n = float(u.y + self.move_amount < self.map_size - 1.0)
            can_s = float(u.y - self.move_amount > 1.0)
            can_e = float(u.x + self.move_amount < self.map_size - 1.0)
            can_w = float(u.x - self.move_amount > 1.0)
            avail[i, 2:6] = [can_n, can_s, can_e, can_w]

            feat = [can_n, can_s, can_e, can_w]
            # Enemies (5 dims each: can_attack, dist, dx, dy, hp_ratio)
            for j, etag in enumerate(self._slot_enemy_tags):
                eu = enemies.get(etag)
                if eu is None:
                    feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    dx = eu.x - u.x
                    dy = eu.y - u.y
                    d = float(np.hypot(dx, dy))
                    in_shoot = 1.0 if d <= self.shoot_range else 0.0
                    if in_shoot:
                        avail[i, 6 + j] = 1.0
                    feat.extend([
                        in_shoot,
                        min(d / self.sight_range, 2.0),
                        np.clip(dx / self.sight_range, -2.0, 2.0),
                        np.clip(dy / self.sight_range, -2.0, 2.0),
                        eu.health / max(eu.health_max, 1.0),
                    ])
            # Other allies (5 dims each: visible, dist, dx, dy, hp_ratio)
            for k, atag in enumerate(self._slot_ally_tags):
                if k == i:
                    continue
                au = allies.get(atag)
                if au is None:
                    feat.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    dx = au.x - u.x
                    dy = au.y - u.y
                    d = float(np.hypot(dx, dy))
                    feat.extend([
                        1.0,
                        min(d / self.sight_range, 2.0),
                        np.clip(dx / self.sight_range, -2.0, 2.0),
                        np.clip(dy / self.sight_range, -2.0, 2.0),
                        au.health / max(au.health_max, 1.0),
                    ])
            # Own features (2 dims: hp_ratio, cooldown_norm)
            feat.extend([u.health / max(u.health_max, 1.0), float(np.clip(u.cooldown / 15.0, 0.0, 1.0))])
            unit_obs[i] = np.asarray(feat, dtype=np.float32)

        # Global state
        g = []
        for tag in self._slot_ally_tags:
            u = allies.get(tag)
            if u is None:
                g.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                g.extend([
                    u.health / max(u.health_max, 1.0),
                    float(np.clip(u.cooldown / 15.0, 0.0, 1.0)),
                    (u.x - cx) / 16.0,
                    (u.y - cy) / 16.0,
                    1.0,
                ])
        for tag in self._slot_enemy_tags:
            eu = enemies.get(tag)
            if eu is None:
                g.extend([0.0, 0.0, 0.0, 0.0])
            else:
                g.extend([
                    eu.health / max(eu.health_max, 1.0),
                    (eu.x - cx) / 16.0,
                    (eu.y - cy) / 16.0,
                    1.0,
                ])
        g.append(self._ep_steps / max(float(self.episode_limit), 1.0))
        global_state = np.asarray(g, dtype=np.float32)

        unit_extra = _build_unit_extra(self.n_agents, agent_tags, allies)
        return StepBundle(
            observation=obs,
            unit_obs=unit_obs,
            unit_extra=unit_extra,
            global_state=global_state,
            avail_actions=avail,
            agent_tags=agent_tags,
            reward=float(reward),
            done=bool(done),
            info=dict(info),
        )

    def reset_round(self, layout: Optional[Layout], opponent: str = "bot") -> StepBundle:
        ts = self._env.reset()[0]
        self._last_ts = ts
        self._ep_steps = 0
        allies, enemies, _ = self._extract_units(ts)
        # Sort slots deterministically by spatial position (x, y, tag)
        self._slot_ally_tags = [u.tag for u in sorted(allies.values(), key=lambda u: (u.x, u.y, u.tag))]
        self._slot_enemy_tags = [u.tag for u in sorted(enemies.values(), key=lambda u: (u.x, u.y, u.tag))]
        while len(self._slot_ally_tags) < self.n_agents:
            self._slot_ally_tags.append(-1 - len(self._slot_ally_tags))
        while len(self._slot_enemy_tags) < self.n_enemies:
            self._slot_enemy_tags.append(-100 - len(self._slot_enemy_tags))
        self._prev_ally_hp = {t: allies[t].health for t in self._slot_ally_tags if t in allies}
        self._prev_enemy_hp = {t: enemies[t].health for t in self._slot_enemy_tags if t in enemies}
        return self._build_step_bundle(ts, reward=0.0, done=False, info={})

    def step_unit_actions(self, actions: Sequence[int]) -> StepBundle:
        allies_now, enemies_now, _ = self._extract_units(self._last_ts)
        raw_fn = self._actions.RAW_FUNCTIONS
        cmds = []
        for i, act in enumerate(actions):
            act = int(act)
            tag = self._slot_ally_tags[i]
            u = allies_now.get(tag)
            if u is None or act == 0:
                continue
            if act == 1:
                cmds.append(raw_fn.Stop_quick("now", tag))
            elif act == 2:
                cmds.append(raw_fn.Move_pt("now", tag, (u.x, min(self.map_size - 1.0, u.y + self.move_amount))))
            elif act == 3:
                cmds.append(raw_fn.Move_pt("now", tag, (u.x, max(1.0, u.y - self.move_amount))))
            elif act == 4:
                cmds.append(raw_fn.Move_pt("now", tag, (min(self.map_size - 1.0, u.x + self.move_amount), u.y)))
            elif act == 5:
                cmds.append(raw_fn.Move_pt("now", tag, (max(1.0, u.x - self.move_amount), u.y)))
            else:
                e_idx = act - 6
                if 0 <= e_idx < len(self._slot_enemy_tags):
                    etag = self._slot_enemy_tags[e_idx]
                    if etag in enemies_now:
                        cmds.append(raw_fn.Attack_unit("now", tag, etag))
        if not cmds:
            cmds = [raw_fn.no_op()]

        ts = self._env.step([cmds])[0]
        self._last_ts = ts
        self._ep_steps += 1
        allies_after, enemies_after, _ = self._extract_units(ts)

        # SMAC-aligned battle reward (damage dealt + kills - 0.5*(damage taken + deaths) + win bonus)
        delta_enemy = 0.0
        for t, prev_hp in list(self._prev_enemy_hp.items()):
            cur_hp = enemies_after[t].health if t in enemies_after else 0.0
            delta_enemy += max(0.0, prev_hp - cur_hp)
            if prev_hp > 0 and cur_hp == 0:
                delta_enemy += 10.0
            self._prev_enemy_hp[t] = cur_hp

        delta_ally = 0.0
        for t, prev_hp in list(self._prev_ally_hp.items()):
            cur_hp = allies_after[t].health if t in allies_after else 0.0
            delta_ally += max(0.0, prev_hp - cur_hp)
            if prev_hp > 0 and cur_hp == 0:
                delta_ally += 10.0
            self._prev_ally_hp[t] = cur_hp

        reward = delta_enemy - 0.5 * delta_ally
        red_alive = len(allies_after)
        blue_alive = len(enemies_after)
        done = bool(ts.last() or red_alive == 0 or blue_alive == 0 or self._ep_steps >= self.episode_limit)
        red_won = bool(done and red_alive > 0 and blue_alive == 0)
        blue_won = bool(done and blue_alive > 0 and red_alive == 0)
        if red_won:
            reward += 200.0
        elif blue_won:
            reward -= 50.0
        reward_scaled = reward / (self._max_reward / 20.0)
        info = {
            "red_battle_won": red_won,
            "blue_battle_won": blue_won,
            "dead_red": self.n_agents - red_alive,
            "dead_blue": self.n_enemies - blue_alive,
        }
        return self._build_step_bundle(ts, reward=reward_scaled, done=done, info=info)

    def episode_details(self, first: Observation, final: Observation, info: dict) -> dict:
        outcome = (
            "win" if info.get("red_battle_won")
            else "loss" if info.get("blue_battle_won")
            else "timeout" if final.friendly and final.enemy
            else "draw"
        )
        return {
            "outcome": outcome,
            "steps": int(final.step),
            "step_limit": int(self.episode_limit),
            "final_game_loop": int(final.game_loop),
            "final_hp": [
                float(sum(u.effective_health for u in side))
                for side in (final.friendly, final.enemy)
            ],
            "hp_margin": float(final.margin),
            "engine_restarts": int(self._env.restart_stats.get("ended_recreates", 0)),
        }

    def close(self) -> None:
        self._env.close()


def create_env(cfg: HRLv2Config, *, seed_offset: int = 0) -> Tuple[BaseMicroEnv, dict]:
    """Instantiate either SC2BAUnitEnv or LegacyMMUnitEnv based on cfg.scenario."""
    if cfg.is_legacy_mm():
        engine_seed = int(cfg.seed * 1009 + seed_offset + 17) & 0x7FFFFFFF
        env = LegacyMMUnitEnv(cfg, engine_seed=engine_seed)
        return env, {"backend": "legacy_mm", "plan_sha256": None}

    document, available, _, _ = read_plan(Path(cfg.plan_path))
    if cfg.scenario not in available:
        raise ValueError(f"Unknown scenario '{cfg.scenario}'. SC2BA={list(available)}, Legacy={list(LEGACY_MM_SCENARIOS)}")
    scenario = available[cfg.scenario]
    engine_seed, _, _ = scenario_seeds(cfg.seed + seed_offset, document, scenario.name)
    env = SC2BAUnitEnv(cfg, scenario=scenario, engine_seed=engine_seed)
    return env, {"backend": "sc2ba", "document": document}
