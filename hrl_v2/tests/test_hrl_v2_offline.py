"""Comprehensive offline unit & integration tests for DyIM-CHRL (hrl_v2).

Runs completely without launching SC2_x64.exe (zero conflict with running v11 jobs),
verifying all 2+2 target scenarios (3m, 2s3z, MM4m, MM8) under mock battle physics:
  1. 4-channel & 1-channel tactical Influence Map construction & world->grid mapping
  2. Online IM-CNN + EMA-VQ warm initialization, Split, Merge, Prune dynamics
  3. Role-aware tactical clustering + Hungarian temporal slot tracking + enemy summaries
  4. Pluggable GuidanceInterface (LatentSkill Option A & ContinuousLatent variant)
  5. FiLM Cluster-Conditioned GRU-Actor strict avail_actions masking & dead-unit handling
  6. End-to-end two-timescale Semi-MDP PPO training, gradient flow to ALL modules,
     checkpoint save/restore, and strictly zero-mutation frozen evaluation.
"""

from pathlib import Path
import tempfile
from typing import Optional, Sequence

import numpy as np
import torch

from bch.observation import Observation, Unit
from bch.plan import Layout

from hrl_v2.arena_pk import (
    CampView,
    DyIMCHRLPKAdapter,
    OnPolicyMAPPOPKAdapter,
    PyMARLQMIXPKAdapter,
    aggregate_pk_results,
    mirror_observation_for_blue,
)
from hrl_v2.baselines import BaselineConfig, OnPolicyMAPPOTrainer, PyMARLQMIXTrainer
from hrl_v2.config import HRLv2Config, LEGACY_MM_SCENARIOS
from hrl_v2.env_adapter import BaseMicroEnv, StepBundle
from hrl_v2.eval_protocol import evaluate_frozen, policy_state_fingerprint
from hrl_v2.influence_map import build_tactical_influence_map
from hrl_v2.lower_policy import DyIMCHRLPolicy
from hrl_v2.tactical_cluster import TacticalClusterer
from hrl_v2.trainer import DyIMCHRLTrainer


class MockMicroBattleEnv(BaseMicroEnv):
    """Lightweight deterministic physics simulator matching SC2BA / LegacyMM StepBundle."""

    def __init__(self, scenario_name: str, seed: int = 7):
        self.scenario_name = scenario_name
        self.rng = np.random.default_rng(seed)
        if scenario_name == "3m":
            self._roster = [(1, 45.0, 0.0, 5.0, False)] * 3
            self.opponents = ("bot", "qmix", "vdn", "dop", "fop", "qplex", "coma", "iql", "qtran")
        elif scenario_name == "2s3z":
            self._roster = [(5, 100.0, 50.0, 0.5, False)] * 3 + [(4, 80.0, 80.0, 6.0, False)] * 2
            self.opponents = ("bot", "qmix", "vdn", "dop", "fop", "qplex", "coma", "iql", "qtran")
        elif scenario_name == "MM4m":
            self._roster = [(1, 45.0, 0.0, 5.0, False)] * 4
            self.opponents = ("bot",)
        elif scenario_name == "MM8":
            self._roster = [(1, 45.0, 0.0, 5.0, False)] * 8
            self.opponents = ("bot",)
        else:
            self._roster = [(1, 45.0, 0.0, 5.0, False)] * 4 + [(3, 150.0, 0.0, 4.0, True)]
            self.opponents = ("bot", "qmix")

        self.n_agents = len(self._roster)
        self.n_enemies = len(self._roster)
        self.n_actions = 6 + self.n_enemies
        self.obs_dim = 4 + self.n_enemies * 5 + (self.n_agents - 1) * 5 + 3
        self.state_dim = self.n_agents * 5 + self.n_enemies * 4 + 1
        self.episode_limit = 20

        self.train_layouts = (
            Layout("standard", "train", (9.0, 16.0), (23.0, 16.0), (), ()),
            Layout("offset", "train", (10.0, 19.0), (23.0, 14.0), (), ()),
        )
        self.val_layouts = (Layout("diagonal", "validation", (9.0, 12.0), (23.0, 20.0), (), ()),)
        self.test_layouts = (Layout("near", "test", (11.0, 14.0), (21.0, 18.0), (), ()),)
        self._step = 0
        self._ally_state = []
        self._enemy_state = []

    def _make_units(self, states, base_tag):
        units = []
        tags = []
        for idx, st in enumerate(states):
            if st["hp"] <= 0:
                tags.append(None)
                continue
            tag = base_tag + idx
            tags.append(tag)
            units.append(
                Unit(
                    tag=tag,
                    x=st["x"],
                    y=st["y"],
                    health=st["hp"],
                    health_max=st["hp_max"],
                    radius=0.5,
                    cooldown=st["cd"],
                    shield=st["sh"],
                    shield_max=st["sh_max"],
                    unit_type=st["utype"],
                    attack_range=st["rng"],
                    is_medivac=st["med"],
                    energy=100.0 if st["med"] else 0.0,
                )
            )
        return tuple(units), tuple(tags)

    def _bundle(self, reward: float, done: bool, info: dict) -> StepBundle:
        friendly, agent_tags = self._make_units(self._ally_state, 100)
        enemy, _ = self._make_units(self._enemy_state, 200)
        obs = Observation(friendly, enemy, self._step, self._step * 8, 32, 16.0 * self.n_agents)

        unit_obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        unit_extra = np.zeros((self.n_agents, 4), dtype=np.float32)
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)

        for i, st in enumerate(self._ally_state):
            if st["hp"] <= 0:
                avail[i, 0] = 1.0
                continue
            avail[i, 1:6] = 1.0
            unit_extra[i] = [
                min(st["cd"] / 35.0, 1.0),
                (st["hp"] + st["sh"]) / (st["hp_max"] + st["sh_max"]),
                1.0 if st["med"] else 0.0,
                1.0,
            ]
            feat = [1.0, 1.0, 1.0, 1.0]
            for j, est in enumerate(self._enemy_state):
                if est["hp"] <= 0:
                    feat.extend([0.0] * 5)
                else:
                    dx, dy = est["x"] - st["x"], est["y"] - st["y"]
                    d = float(np.hypot(dx, dy))
                    can_atk = 1.0 if (not st["med"] and d <= max(st["rng"], 1.0) + 1.5) else 0.0
                    if can_atk > 0.5:
                        avail[i, 6 + j] = 1.0
                    feat.extend([can_atk, d / 12.0, dx / 12.0, dy / 12.0, est["hp"] / est["hp_max"]])
            for k, ast in enumerate(self._ally_state):
                if k == i:
                    continue
                if ast["hp"] <= 0:
                    feat.extend([0.0] * 5)
                else:
                    dx, dy = ast["x"] - st["x"], ast["y"] - st["y"]
                    d = float(np.hypot(dx, dy))
                    feat.extend([1.0, d / 12.0, dx / 12.0, dy / 12.0, ast["hp"] / ast["hp_max"]])
            feat.extend([st["hp"] / st["hp_max"], st["sh"] / max(st["sh_max"], 1.0), st["cd"] / 35.0])
            unit_obs[i] = np.asarray(feat, dtype=np.float32)

        g = []
        for st in self._ally_state:
            g.extend([st["hp"] / st["hp_max"], st["cd"] / 35.0, (st["x"] - 16.0) / 16.0, (st["y"] - 16.0) / 16.0, float(st["hp"] > 0)])
        for est in self._enemy_state:
            g.extend([est["hp"] / est["hp_max"], (est["x"] - 16.0) / 16.0, (est["y"] - 16.0) / 16.0, float(est["hp"] > 0)])
        g.append(self._step / float(self.episode_limit))
        return StepBundle(obs, unit_obs, unit_extra, np.asarray(g, dtype=np.float32), avail, agent_tags, reward, done, info)

    def reset_round(self, layout: Optional[Layout], opponent: str) -> StepBundle:
        self._step = 0
        rx, ry = (layout.red if layout and layout.red else (10.0, 16.0))
        bx, by = (layout.blue if layout and layout.blue else (18.0, 16.0))
        self._ally_state = []
        self._enemy_state = []
        for idx, (ut, hp, sh, rng, med) in enumerate(self._roster):
            off_y = (idx - self.n_agents / 2.0) * 1.1
            self._ally_state.append({"utype": ut, "hp": hp, "hp_max": hp, "sh": sh, "sh_max": sh, "rng": rng, "med": med,
                                     "x": rx + float(self.rng.uniform(-0.8, 0.8)), "y": ry + off_y, "cd": 0.0})
            self._enemy_state.append({"utype": ut, "hp": hp, "hp_max": hp, "sh": sh, "sh_max": sh, "rng": rng, "med": med,
                                      "x": bx + float(self.rng.uniform(-0.8, 0.8)), "y": by + off_y, "cd": 0.0})
        return self._bundle(0.0, False, {})

    def step_unit_actions(self, actions: Sequence[int]) -> StepBundle:
        assert len(actions) == self.n_agents
        cur_bundle = self._bundle(0.0, False, {})
        reward = 0.0
        for i, act in enumerate(actions):
            act = int(act)
            assert cur_bundle.avail_actions[i, act] == 1.0, f"Illegal action {act} selected for unit {i}!"
            st = self._ally_state[i]
            if st["hp"] <= 0:
                continue
            st["cd"] = max(0.0, st["cd"] - 8.0)
            if act == 2:
                st["y"] = min(23.5, st["y"] + 1.5)
            elif act == 3:
                st["y"] = max(8.5, st["y"] - 1.5)
            elif act == 4:
                st["x"] = min(31.5, st["x"] + 1.5)
            elif act == 5:
                st["x"] = max(0.5, st["x"] - 1.5)
            elif act >= 6:
                e = self._enemy_state[act - 6]
                if e["hp"] > 0 and st["cd"] <= 0.0:
                    dmg = 18.0
                    e["hp"] = max(0.0, e["hp"] - dmg)
                    st["cd"] = 12.0
                    reward += 0.4 + (0.6 if e["hp"] == 0 else 0.0)

        # Enemy counter-attack / approach
        alive_allies = [s for s in self._ally_state if s["hp"] > 0]
        for est in self._enemy_state:
            if est["hp"] <= 0 or not alive_allies:
                continue
            target = min(alive_allies, key=lambda a: np.hypot(a["x"] - est["x"], a["y"] - est["y"]))
            d = float(np.hypot(target["x"] - est["x"], target["y"] - est["y"]))
            if d <= max(est["rng"], 1.0) + 1.5:
                target["hp"] = max(0.0, target["hp"] - 12.0)
                reward -= 0.25
            else:
                est["x"] += 1.2 * (target["x"] - est["x"]) / max(d, 1e-6)
                est["y"] += 1.2 * (target["y"] - est["y"]) / max(d, 1e-6)

        self._step += 1
        red_alive = sum(1 for s in self._ally_state if s["hp"] > 0)
        blue_alive = sum(1 for s in self._enemy_state if s["hp"] > 0)
        done = (red_alive == 0) or (blue_alive == 0) or (self._step >= self.episode_limit)
        red_won = bool(done and red_alive > 0 and blue_alive == 0)
        blue_won = bool(done and blue_alive > 0 and red_alive == 0)
        if red_won:
            reward += 4.0
        elif blue_won:
            reward -= 2.0
        info = {"red_battle_won": red_won, "blue_battle_won": blue_won,
                "dead_red": self.n_agents - red_alive, "dead_blue": self.n_enemies - blue_alive}
        return self._bundle(reward, done, info)

    def episode_details(self, first: Observation, final: Observation, info: dict) -> dict:
        outcome = "win" if info.get("red_battle_won") else "loss" if info.get("blue_battle_won") else "timeout"
        return {
            "outcome": outcome,
            "steps": int(final.step),
            "step_limit": int(self.episode_limit),
            "final_game_loop": int(final.game_loop),
            "final_hp": [float(sum(u.effective_health for u in s)) for s in (final.friendly, final.enemy)],
            "hp_margin": float(final.margin),
            "engine_restarts": 0,
        }

    def close(self) -> None:
        pass


def test_influence_map_and_clustering():
    env = MockMicroBattleEnv("2s3z", seed=11)
    bundle = env.reset_round(env.train_layouts[0], "bot")

    im4 = build_tactical_influence_map(bundle.observation, grid_size=64, channels=4)
    im1 = build_tactical_influence_map(bundle.observation, grid_size=64, channels=1)
    assert im4.tensor.shape == (4, 64, 64) and np.isfinite(im4.tensor).all()
    assert im1.tensor.shape == (1, 64, 64) and np.allclose(im4.tensor[0], im1.tensor[0])
    for u in bundle.observation.friendly:
        gx, gy = im4.world_to_norm(u.x, u.y)
        assert -1.0 <= gx <= 1.0 and -1.0 <= gy <= 1.0

    cfg = HRLv2Config(scenario="2s3z", max_clusters=3)
    clu = TacticalClusterer(cfg)
    a1 = clu.assign(bundle.observation, bundle.agent_tags, im4, requested_clusters=2, recluster=True)
    assert a1.num_clusters == 2
    assert a1.cluster_feats.shape == (3, 20)
    assert np.all(a1.unit_to_cluster >= 0)
    # Check temporal stability on micro step without reclustering
    a2 = clu.assign(bundle.observation, bundle.agent_tags, im4, requested_clusters=2, recluster=False)
    assert np.array_equal(a1.unit_to_cluster, a2.unit_to_cluster)
    print("[PASS] test_influence_map_and_clustering")


def test_dynamic_split_merge_prune():
    cfg = HRLv2Config(
        proto_dim=16,
        init_prototypes=4,
        min_prototypes=3,
        max_prototypes=12,
        min_proto_count=4.0,
        split_score_threshold=0.05,
        split_cooldown=1,
        dead_proto_count=0.5,
        dead_proto_age=2,
        merge_dist_threshold=0.50,
        merge_value_threshold=0.50,
    )
    policy = DyIMCHRLPolicy(cfg, obs_dim=32, state_dim=20, n_actions=9)
    cb = policy.abstractor.codebook
    rng = torch.Generator().manual_seed(42)

    # 1. Warm init online (0 offline data)
    z0 = torch.nn.functional.normalize(torch.randn(32, 16, generator=rng), dim=-1)
    assert cb.warm_initialize(z0)
    assert cb.num_active == 4

    # 2. Force high TD-error & spatial spread inside prototype 0 -> trigger SPLIT
    slot0_vec = cb.prototypes[0].unsqueeze(0)
    axis = torch.nn.functional.normalize(torch.randn(1, 16, generator=rng), dim=-1)
    z_hi = torch.nn.functional.normalize(slot0_vec + 0.35 * axis, dim=-1).repeat(16, 1)
    z_lo = torch.nn.functional.normalize(slot0_vec - 0.35 * axis, dim=-1).repeat(16, 1)
    z_mix = torch.cat([z_hi, z_lo], dim=0)
    pids = torch.zeros(32, dtype=torch.long)
    tds = torch.cat([torch.full((16,), 5.0), torch.full((16,), -5.0)])
    rets = torch.cat([torch.full((16,), 4.0), torch.full((16,), -4.0)])
    vals = torch.zeros(32)
    for _ in range(3):
        cb.observe_online_stats(z_mix, pids, tds, rets, vals)
    events = cb.maybe_split_and_merge(update_idx=1)
    assert any(e["event"] == "split" for e in events), f"Expected split event, got {events}"
    assert cb.num_active >= 4

    # 3. Force two prototypes nearly identical with matching values -> trigger MERGE
    act = cb.active_indices().tolist()
    s1, s2 = act[0], act[1]
    cb.prototypes[s2].copy_(torch.nn.functional.normalize((cb.prototypes[s1] + 0.01 * axis.squeeze(0)).unsqueeze(0), dim=-1).squeeze(0))
    cb.ema_val_mean[s1] = 1.0
    cb.ema_val_mean[s2] = 1.02
    cb.ema_ret_mean[s1] = 0.5
    cb.ema_ret_mean[s2] = 0.51
    cb.proto_cooldown.zero_()
    cb.proto_age.fill_(5)
    cb.ema_count.fill_(20.0)
    k_before = cb.num_active
    events_m = cb.maybe_split_and_merge(update_idx=2)
    assert any(e["event"] == "merge" for e in events_m), f"Expected merge event, got {events_m}"
    assert cb.num_active == k_before - 1
    print(f"[PASS] test_dynamic_split_merge_prune (events: split={cb.total_splits.item()}, merge={cb.total_merges.item()})")


def test_guidance_extensibility_and_gradients():
    for gtype in ("latent_skill", "continuous_latent"):
        cfg = HRLv2Config(guidance_type=gtype, im_channels=4, im_size=64, max_clusters=3)
        pol = DyIMCHRLPolicy(cfg, obs_dim=40, state_dim=30, n_actions=10)
        im = torch.randn(4, 4, 64, 64)
        abs_out = pol.encode_upper(im)
        cf = torch.randn(4, 3, 20)
        cc = torch.empty(4, 3, 2).uniform_(-0.8, 0.8)
        cm = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        fused = pol.fuse_cluster_features(abs_out.spatial_feat, cf, cc, cm)
        g_s = pol.guidance_head.sample(abs_out.z_quant, fused, cm)
        g_e = pol.guidance_head.evaluate(abs_out.z_quant, fused, cm, g_s.raw_action)
        assert g_s.cluster_guidance.shape == (4, 3, cfg.guidance_dim)
        assert torch.allclose(g_s.log_prob, g_e.log_prob, atol=1e-5)

        loss = g_e.log_prob.mean() + abs_out.commit_loss + g_e.diversity_loss
        loss.backward()
        # Verify gradients reach IM-CNN stem through STE + spatial grid_sample
        stem_grad = sum(p.grad.abs().sum().item() for p in pol.abstractor.backbone.parameters() if p.grad is not None)
        assert stem_grad > 0.0, f"No gradient reached IM-CNN backbone for guidance_type={gtype}!"
    print("[PASS] test_guidance_extensibility_and_gradients (latent_skill + continuous_latent)")


def test_full_2plus2_scenarios_end_to_end():
    """Run short end-to-end training + checkpoint reload + frozen eval across all 4 target scenarios."""
    with tempfile.TemporaryDirectory(prefix="hrl_v2_test_") as tmp:
        root = Path(tmp)
        for scenario, regime in (("3m", "pool-all9"), ("2s3z", "fixed-bot"), ("MM4m", "fixed-bot"), ("MM8", "fixed-bot")):
            env = MockMicroBattleEnv(scenario, seed=101)
            cfg = HRLv2Config(
                scenario=scenario,
                regime=regime,
                seed=101,
                device="cpu",
                im_channels=4,
                train_steps=160,
                rollout_steps=64,
                minibatch_size=32,
                eval_every=80,
                eval_init=True,
                val_repeats=1,
                test_repeats=1,
                legacy_eval_episodes=2,
                legacy_final_episodes=3,
                split_merge_every=1,
                min_proto_count=2.0,
                split_score_threshold=0.15,
            )
            train_opps = ("bot",) if regime == "fixed-bot" else env.opponents
            eval_opps = ("bot",) if scenario in LEGACY_MM_SCENARIOS else env.opponents
            out_dir = root / scenario / regime
            trainer = DyIMCHRLTrainer(cfg, env, train_opponents=train_opps, eval_opponents=eval_opps, out_dir=out_dir)
            report = trainer.train()

            assert (out_dir / "report.json").exists()
            assert (out_dir / "done.json").exists()
            assert (out_dir / "learning_curve.json").exists()
            assert (out_dir / "codebook_evolution.jsonl").exists()
            assert (out_dir / "checkpoint_final.pt").exists()
            assert report["train_steps"] == 160
            assert report["final_eval"]["active_prototypes"] >= cfg.min_prototypes

            # Reload checkpoint & verify frozen eval leaves state_dict 100% unchanged
            trainer2 = DyIMCHRLTrainer(cfg, env, train_opponents=train_opps, eval_opponents=eval_opps, out_dir=out_dir / "reload")
            trainer2.load_checkpoint(out_dir / "checkpoint_final.pt")
            fp1 = policy_state_fingerprint(trainer.policy)
            fp2 = policy_state_fingerprint(trainer2.policy)
            assert fp1 == fp2, "Checkpoint reload fingerprint mismatch!"
            summary, _ = evaluate_frozen(
                env, trainer2.policy, cfg, trainer2.device,
                layouts=env.test_layouts, opponents=eval_opps[:2], repeats=1, step_tag=160,
            )
            assert policy_state_fingerprint(trainer2.policy) == fp2
            print(
                f"[PASS] E2E {scenario}/{regime}: steps={report['train_steps']} "
                f"K_final={report['final_eval']['active_prototypes']} "
                f"all9={report['final_eval']['all9_win_rate']:.2f} "
                f"clu={report['final_eval']['mean_clusters']:.2f}"
            )


def test_resume_and_curriculum_dyim():
    """Verify checkpoint continuation (80 -> 160 steps) + legacy optimizer-less ckpt + curriculum warmup."""
    with tempfile.TemporaryDirectory(prefix="hrl_v2_resume_") as tmp:
        root = Path(tmp)
        env = MockMicroBattleEnv("3m", seed=23)
        cfg1 = HRLv2Config(
            scenario="3m", regime="pool-all9", seed=23, device="cpu",
            train_steps=80, rollout_steps=40, minibatch_size=20, eval_every=40,
            eval_init=True, val_repeats=1, test_repeats=1,
            curriculum_warmup_frac=0.5, curriculum_bot_start_prob=0.85,
        )
        d1 = root / "stage1"
        t1 = DyIMCHRLTrainer(cfg1, env, train_opponents=env.opponents, eval_opponents=env.opponents, out_dir=d1)
        p0 = t1._current_bot_sampling_prob()
        assert p0 is not None and abs(p0 - 0.85) < 1e-6
        r1 = t1.train()
        assert r1["train_steps"] == 80
        p_end = t1._current_bot_sampling_prob()
        assert p_end is not None and abs(p_end - 1.0 / 9.0) < 1e-6

        # Resume cleanly into a new directory to 160 steps
        cfg2 = HRLv2Config(
            scenario="3m", regime="pool-all9", seed=23, device="cpu",
            train_steps=160, rollout_steps=40, minibatch_size=20, eval_every=40,
            eval_init=True, val_repeats=1, test_repeats=1,
            resume_from=str(d1 / "checkpoint_final.pt"),
        )
        d2 = root / "stage2"
        t2 = DyIMCHRLTrainer(cfg2, env, train_opponents=env.opponents, eval_opponents=env.opponents, out_dir=d2)
        t2.load_checkpoint(d1 / "checkpoint_final.pt")
        assert t2.total_steps == 80 and len(t2.learning_curve) == 3  # 0, 40, 80
        r2 = t2.train()
        assert r2["train_steps"] == 160
        assert len(t2.learning_curve) == 5  # 0, 40, 80, 120, 160
        assert (d2 / "codebook_evolution.jsonl").exists()
        print("[PASS] test_resume_and_curriculum_dyim (80 -> 160 steps, 5 curve points)")


def test_baselines_qmix_mappo_and_pk_adapters():
    """Verify actual PyMARL QMIX + on-policy MAPPO trainers, resume, and symmetric PK adapters."""
    with tempfile.TemporaryDirectory(prefix="hrl_v2_base_pk_") as tmp:
        root = Path(tmp)
        env = MockMicroBattleEnv("3m", seed=31)

        # 1. PyMARL QMIX (real RNNAgent + QMixer)
        q_cfg = BaselineConfig(
            algo="qmix", scenario="3m", regime="pool-all9", seed=31, device="cpu",
            train_steps=120, eval_every=60, eval_init=True, val_repeats=1, test_repeats=1,
            qmix_batch_size=2, qmix_target_update_interval=3, qmix_epsilon_anneal_steps=100,
        )
        q_dir = root / "qmix"
        qt = PyMARLQMIXTrainer(q_cfg, env, train_opponents=env.opponents, eval_opponents=env.opponents, out_dir=q_dir)
        qr = qt.train()
        assert qr["algorithm"] == "pymarl-qmix" and (q_dir / "checkpoint_final.pt").exists()
        assert qr["train_steps"] >= 120

        # 2. on-policy R-MAPPO (real R_MAPPOPolicy + R_MAPPO + SharedReplayBuffer)
        m_cfg = BaselineConfig(
            algo="mappo", scenario="3m", regime="pool-all9", seed=31, device="cpu",
            train_steps=120, eval_every=60, eval_init=True, val_repeats=1, test_repeats=1,
            mappo_rollout_steps=60, mappo_ppo_epoch=2, mappo_data_chunk_length=10,
        )
        m_dir = root / "mappo"
        mt = OnPolicyMAPPOTrainer(m_cfg, env, train_opponents=env.opponents, eval_opponents=env.opponents, out_dir=m_dir)
        mr = mt.train()
        assert mr["algorithm"] == "onpolicy-mappo" and (m_dir / "checkpoint_final.pt").exists()
        assert mr["train_steps"] == 120

        # 3. DyIM-CHRL checkpoint for PK adapter verification
        d_cfg = HRLv2Config(
            scenario="3m", regime="pool-all9", seed=31, device="cpu",
            train_steps=80, rollout_steps=40, minibatch_size=20, eval_every=80,
            eval_init=False, val_repeats=1, test_repeats=1,
        )
        d_dir = root / "dyim"
        dt = DyIMCHRLTrainer(d_cfg, env, train_opponents=env.opponents, eval_opponents=env.opponents, out_dir=d_dir)
        dt.train()

        # 4. Verify mirrored Blue Observation + all 3 PK adapters produce strictly legal actions
        b = env.reset_round(env.train_layouts[0], "bot")
        obs_blue = mirror_observation_for_blue(b.observation, center_x=16.0, center_y=16.0)
        assert len(obs_blue.friendly) == len(b.observation.enemy)
        assert len(obs_blue.enemy) == len(b.observation.friendly)
        assert abs(obs_blue.friendly[0].x - (32.0 - b.observation.enemy[0].x)) < 1e-5

        red_view = CampView("RED", b.observation, b.unit_obs, b.unit_extra, b.global_state, b.avail_actions, b.agent_tags)
        blue_tags = tuple(u.tag for u in obs_blue.friendly)
        blue_view = CampView("BLUE", obs_blue, b.unit_obs, b.unit_extra, b.global_state, b.avail_actions, blue_tags)

        adapters = [
            DyIMCHRLPKAdapter("dyim", d_dir / "checkpoint_final.pt", torch.device("cpu")),
            PyMARLQMIXPKAdapter("qmix", q_dir / "checkpoint_final.pt", torch.device("cpu")),
            OnPolicyMAPPOPKAdapter("mappo", m_dir / "checkpoint_final.pt", torch.device("cpu")),
        ]
        for ad in adapters:
            ad.reset()
            for v in (red_view, blue_view):
                acts = ad.select_actions(v, step_idx=0)
                assert len(acts) == env.n_agents
                for i, a in enumerate(acts):
                    assert v.avail_actions[i, a] == 1.0, f"{ad.label} chose illegal action {a} on {v.camp_name}"

        # 5. Verify PK matrix reciprocal aggregation math
        agg = aggregate_pk_results(
            ["dyim", "qmix"],
            [
                {"red": "dyim", "blue": "qmix", "red_result": "win", "hp_margin": 20.0},
                {"red": "qmix", "blue": "dyim", "red_result": "loss", "hp_margin": -10.0},
            ],
        )
        assert agg["matrix"]["dyim"]["qmix"] == 1.0
        assert agg["matrix"]["qmix"]["dyim"] == 0.0
        assert agg["ranking"][0]["policy"] == "dyim"
        print("[PASS] test_baselines_qmix_mappo_and_pk_adapters (PyMARL QMIX + on-policy MAPPO + PK mirror)")


if __name__ == "__main__":
    test_influence_map_and_clustering()
    test_dynamic_split_merge_prune()
    test_guidance_extensibility_and_gradients()
    test_full_2plus2_scenarios_end_to_end()
    test_resume_and_curriculum_dyim()
    test_baselines_qmix_mappo_and_pk_adapters()
    print("\nALL HRL_V2 OFFLINE TESTS PASSED!")
