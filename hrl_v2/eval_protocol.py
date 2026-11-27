"""Frozen evaluation protocol aligned 1:1 with v10/v11 (SC2BA) and v09 (legacy MM)."""

from collections import Counter, defaultdict
from hashlib import sha256
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bch.plan import Layout

from .config import HRLv2Config
from .env_adapter import BaseMicroEnv
from .influence_map import build_tactical_influence_map
from .lower_policy import DyIMCHRLPolicy
from .tactical_cluster import TacticalClusterer

ALGO_OPPONENTS = ("qmix", "vdn", "dop", "fop", "qplex", "coma", "iql", "qtran")


def policy_state_fingerprint(policy: DyIMCHRLPolicy) -> str:
    """Deterministic SHA-256 over all model weights & codebook buffers to verify frozen eval."""
    h = sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def run_frozen_episode(
    env: BaseMicroEnv,
    policy: DyIMCHRLPolicy,
    cfg: HRLv2Config,
    device: torch.device,
    layout: Optional[Layout],
    opponent: str,
) -> dict:
    """Execute one deterministic frozen episode without mutating policy or codebook."""
    clusterer = TacticalClusterer(cfg)
    clusterer.reset()

    bundle = env.reset_round(layout, opponent)
    first_obs = bundle.observation
    rnn_hidden = policy.init_lower_hidden(1, env.n_agents, device)

    visited_protos = set()
    cluster_counts: List[int] = []
    skill_hist = Counter()
    encode_times_ms: List[float] = []

    cached_spatial = None
    cached_guidance = None
    requested_clusters = 1
    step_idx = 0
    total_reward = 0.0

    while not bundle.done:
        is_macro = (step_idx % cfg.macro_interval == 0) or (cached_spatial is None)
        im_grid = build_tactical_influence_map(
            bundle.observation, grid_size=cfg.im_size, channels=cfg.im_channels
        )

        if is_macro:
            t0 = perf_counter()
            im_t = torch.from_numpy(im_grid.tensor).unsqueeze(0).to(device)
            abs_out = policy.encode_upper(im_t)
            alive_count = torch.tensor([len(bundle.observation.friendly)], dtype=torch.long, device=device)
            gran_idx, _, _ = policy.sample_granularity(abs_out.z_quant, alive_count, deterministic=True)
            requested_clusters = int(gran_idx.item()) + 1
            assign = clusterer.assign(
                bundle.observation,
                bundle.agent_tags,
                im_grid,
                requested_clusters=requested_clusters,
                recluster=True,
            )
            c_feats_t = torch.from_numpy(assign.cluster_feats).unsqueeze(0).to(device)
            c_coords_t = torch.from_numpy(assign.cluster_norm_coords).unsqueeze(0).to(device)
            c_mask_t = torch.from_numpy(assign.cluster_mask).unsqueeze(0).to(device)
            fused_cf = policy.fuse_cluster_features(abs_out.spatial_feat, c_feats_t, c_coords_t, c_mask_t)
            g_out = policy.guidance_head.sample(abs_out.z_quant, fused_cf, c_mask_t, deterministic=True)
            encode_times_ms.append((perf_counter() - t0) * 1000.0)

            cached_spatial = abs_out.spatial_feat
            cached_guidance = g_out.cluster_guidance
            visited_protos.add(int(abs_out.proto_id.item()))
            cluster_counts.append(assign.num_clusters)
            if "skill_ids" in g_out.aux_info:
                sids = g_out.aux_info["skill_ids"][0].cpu().numpy()
                for slot in range(cfg.max_clusters):
                    if assign.cluster_mask[slot] > 0.5:
                        skill_hist[int(sids[slot])] += 1
        else:
            assign = clusterer.assign(
                bundle.observation,
                bundle.agent_tags,
                im_grid,
                requested_clusters=requested_clusters,
                recluster=False,
            )
            c_feats_t = torch.from_numpy(assign.cluster_feats).unsqueeze(0).to(device)
            c_coords_t = torch.from_numpy(assign.cluster_norm_coords).unsqueeze(0).to(device)
            c_mask_t = torch.from_numpy(assign.cluster_mask).unsqueeze(0).to(device)
            fused_cf = policy.fuse_cluster_features(cached_spatial, c_feats_t, c_coords_t, c_mask_t)

        unit_obs_t = torch.from_numpy(bundle.unit_obs).unsqueeze(0).to(device)
        unit_extra_t = torch.from_numpy(bundle.unit_extra).unsqueeze(0).to(device)
        unit_rel_t = torch.from_numpy(assign.unit_rel_cluster).unsqueeze(0).to(device)
        u2c_t = torch.from_numpy(assign.unit_to_cluster).unsqueeze(0).to(device)
        avail_t = torch.from_numpy(bundle.avail_actions).unsqueeze(0).to(device)
        gstate_t = torch.from_numpy(bundle.global_state).unsqueeze(0).to(device)

        actions_t, _, _, _, rnn_hidden = policy.act_lower(
            unit_obs_t,
            unit_extra_t,
            unit_rel_t,
            fused_cf,
            cached_guidance,
            u2c_t,
            rnn_hidden,
            avail_t,
            gstate_t,
            deterministic=True,
        )
        bundle = env.step_unit_actions(actions_t[0].cpu().tolist())
        total_reward += bundle.reward
        step_idx += 1

    audit = env.episode_details(first_obs, bundle.observation, bundle.info)
    return {
        **audit,
        "episode_reward": float(total_reward),
        "visited_prototypes": len(visited_protos),
        "mean_clusters": float(np.mean(cluster_counts)) if cluster_counts else 0.0,
        "skill_hist": dict(skill_hist),
        "encode_ms_p50": float(np.median(encode_times_ms)) if encode_times_ms else 0.0,
    }


def evaluate_frozen(
    env: BaseMicroEnv,
    policy: DyIMCHRLPolicy,
    cfg: HRLv2Config,
    device: torch.device,
    *,
    layouts: Sequence[Layout],
    opponents: Sequence[str],
    repeats: int,
    step_tag: int,
) -> Tuple[dict, List[dict]]:
    """Run strictly frozen multi-opponent evaluation with immutability verification."""
    was_training = policy.training
    policy.eval()
    fp_before = policy_state_fingerprint(policy)

    rows: List[dict] = []
    by_opp = defaultdict(Counter)
    by_layout_opp = defaultdict(Counter)
    hp_margins: List[float] = []
    encode_p50s: List[float] = []
    cluster_means: List[float] = []
    skill_totals = Counter()

    try:
        for layout in layouts:
            for opponent in opponents:
                for rep in range(repeats):
                    audit = run_frozen_episode(env, policy, cfg, device, layout, opponent)
                    by_opp[opponent][audit["outcome"]] += 1
                    by_layout_opp[(layout.id, opponent)][audit["outcome"]] += 1
                    hp_margins.append(float(audit.get("hp_margin", 0.0)))
                    encode_p50s.append(audit["encode_ms_p50"])
                    cluster_means.append(audit["mean_clusters"])
                    skill_totals.update(audit["skill_hist"])
                    rows.append({
                        "checkpoint_steps": step_tag,
                        "layout": layout.id,
                        "split": layout.split,
                        "opponent": opponent,
                        "repeat": rep,
                        "outcome": audit["outcome"],
                        "steps": audit["steps"],
                        "final_hp": audit["final_hp"],
                        "hp_margin": audit.get("hp_margin", 0.0),
                        "episode_reward": audit["episode_reward"],
                        "visited_prototypes": audit["visited_prototypes"],
                        "mean_clusters": audit["mean_clusters"],
                        "encode_ms_p50": audit["encode_ms_p50"],
                    })
    finally:
        fp_after = policy_state_fingerprint(policy)
        if was_training:
            policy.train()

    if fp_before != fp_after:
        raise RuntimeError("Frozen evaluation mutated policy weights or dynamic codebook buffers!")

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

    cb_diag = policy.abstractor.codebook.diagnostics()
    summary = {
        "checkpoint_steps": step_tag,
        "episodes": len(rows),
        "all9_win_rate": all9_wr,
        "algos8_win_rate": algos8_wr,
        "worst3_algos_win_rate": worst3_wr,
        "bot_win_rate": bot_wr,
        "mean_hp_margin": float(np.mean(hp_margins)) if hp_margins else 0.0,
        "by_layout_win_rate": by_layout,
        "active_prototypes": cb_diag["active_prototypes"],
        "codebook_diagnostics": cb_diag,
        "mean_clusters": float(np.mean(cluster_means)) if cluster_means else 0.0,
        "skill_usage_counts": {str(k): int(v) for k, v in sorted(skill_totals.items())},
        "encode_ms_p50": float(np.median(encode_p50s)) if encode_p50s else 0.0,
        "per_opponent": per_opponent,
    }
    return summary, rows
