"""Role- and spatial-aware internal tactical clustering with temporal tracking.

Replaces legacy ``bch/lower_action/grouping.py`` (random 1-init KMeans on friendly (x, y)
and enemy-blind (uniformity, crowding) hex keys) with:
  1. Spatial + unit-role (melee / ranged / medivac) + HP-ratio clustering.
  2. Warm-started centroids + Hungarian assignment matching across macro steps so
     cluster identities stay temporally coherent instead of permuting randomly.
  3. Permutation-invariant 20-dim per-cluster summaries INCLUDING local enemy threats,
     plus normalized IM-CNN grid_sample coordinates for spatial feature fusion.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from bch.observation import Observation, Unit

from .config import HRLv2Config
from .influence_map import InfluenceGrid

CLUSTER_STAT_DIM = 20


@dataclass
class ClusterAssignment:
    num_clusters: int
    cluster_feats: np.ndarray        # [M_max, CLUSTER_STAT_DIM] float32
    cluster_norm_coords: np.ndarray  # [M_max, 2] float32 in [-1, 1]^2 for IM-CNN grid_sample
    cluster_mask: np.ndarray         # [M_max] float32 (1=active cluster, 0=pad)
    unit_to_cluster: np.ndarray      # [N_agents] int64 in [0, M_max-1], -1 if dead
    unit_rel_cluster: np.ndarray     # [N_agents, 2] float32 offset (unit - cluster_center) / 8.0
    centroids_xy: np.ndarray         # [M_max, 2] float32 world coordinates (for next-step tracking)
    centroids_valid: np.ndarray      # [M_max] bool


class TacticalClusterer:
    """Temporally coherent internal battlefield clusterer."""

    def __init__(self, cfg: HRLv2Config):
        self.cfg = cfg
        self.max_clusters = cfg.max_clusters
        self._prev_centroids: Optional[np.ndarray] = None
        self._prev_valid: Optional[np.ndarray] = None
        self._cached_unit_cluster: Optional[Dict[int, int]] = None

    def reset(self) -> None:
        self._prev_centroids = None
        self._prev_valid = None
        self._cached_unit_cluster = None

    def _unit_cluster_features(self, units: Sequence[Unit], cx: float, cy: float, span: float) -> np.ndarray:
        feats = np.zeros((len(units), 5), dtype=np.float32)
        inv_span = 1.0 / max(span, 4.0)
        for i, u in enumerate(units):
            melee = 1.0 if (not u.is_medivac and u.attack_range < 2.0) else 0.0
            med = 1.0 if u.is_medivac else 0.0
            hp = u.effective_health / max(u.effective_health_max, 1.0)
            feats[i, 0] = (u.x - cx) * inv_span
            feats[i, 1] = (u.y - cy) * inv_span
            feats[i, 2] = self.cfg.cluster_role_weight * melee
            feats[i, 3] = self.cfg.cluster_role_weight * 1.4 * med
            feats[i, 4] = self.cfg.cluster_hp_weight * hp
        return feats

    def _run_lloyd(self, x: np.ndarray, k: int, init_xy: Optional[np.ndarray], world_xy: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if k <= 1:
            return np.zeros(n, dtype=np.int64)
        if k >= n:
            return np.arange(n, dtype=np.int64)

        # Warm-start from previous world centroids when available, otherwise deterministic farthest-point
        chosen: list = []
        if init_xy is not None and len(init_xy) > 0:
            for pt in init_xy[:k]:
                dists = np.linalg.norm(world_xy - pt[None, :], axis=1)
                for c in chosen:
                    dists[c] = np.inf
                best = int(np.argmin(dists))
                if np.isfinite(dists[best]):
                    chosen.append(best)
        if not chosen:
            # Deterministic start: unit farthest from mean
            d0 = np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=1)
            chosen.append(int(np.argmax(d0)))
        while len(chosen) < k:
            cur = x[chosen]
            dmat = np.linalg.norm(x[:, None, :] - cur[None, :, :], axis=2).min(axis=1)
            for c in chosen:
                dmat[c] = -1.0
            chosen.append(int(np.argmax(dmat)))

        centers = x[chosen].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(self.cfg.cluster_kmeans_iters):
            dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            for c in range(k):
                members = x[labels == c]
                if len(members) > 0:
                    centers[c] = members.mean(axis=0)
        return labels

    def _match_cluster_slots(self, raw_groups: list) -> Tuple[Dict[int, list], np.ndarray, np.ndarray]:
        """Assign each group to a stable slot in [0, M_max-1] via Hungarian centroid matching."""
        m_max = self.max_clusters
        k = len(raw_groups)
        new_xy = np.array([
            [np.mean([u.x for u in grp]), np.mean([u.y for u in grp])] for grp in raw_groups
        ], dtype=np.float32)

        slot_map: Dict[int, list] = {}
        centroids_xy = np.zeros((m_max, 2), dtype=np.float32)
        centroids_valid = np.zeros((m_max,), dtype=bool)

        if self._prev_centroids is not None and self._prev_valid is not None and self._prev_valid.any():
            prev_slots = np.flatnonzero(self._prev_valid)
            prev_xy = self._prev_centroids[prev_slots]
            cost = np.linalg.norm(new_xy[:, None, :] - prev_xy[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_rows = set()
            used_slots = set()
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                slot = int(prev_slots[c])
                slot_map[slot] = raw_groups[r]
                centroids_xy[slot] = new_xy[r]
                centroids_valid[slot] = True
                assigned_rows.add(r)
                used_slots.add(slot)
            free_slots = [s for s in range(m_max) if s not in used_slots]
            for r in range(k):
                if r not in assigned_rows:
                    slot = free_slots.pop(0)
                    slot_map[slot] = raw_groups[r]
                    centroids_xy[slot] = new_xy[r]
                    centroids_valid[slot] = True
        else:
            # Canonical spatial ordering (left-to-right, bottom-to-top) on first step
            order = sorted(range(k), key=lambda r: (float(new_xy[r, 0]), float(new_xy[r, 1])))
            for slot, r in enumerate(order):
                slot_map[slot] = raw_groups[r]
                centroids_xy[slot] = new_xy[r]
                centroids_valid[slot] = True

        return slot_map, centroids_xy, centroids_valid

    def assign(
        self,
        obs: Observation,
        agent_tags: Sequence[Optional[int]],
        im_grid: InfluenceGrid,
        *,
        requested_clusters: int,
        recluster: bool,
    ) -> ClusterAssignment:
        """Compute or refresh internal tactical clusters and build per-cluster summaries."""
        n_agents = len(agent_tags)
        m_max = self.max_clusters
        friendly_by_tag = {u.tag: u for u in obs.friendly}
        alive_units = list(obs.friendly)

        cluster_feats = np.zeros((m_max, CLUSTER_STAT_DIM), dtype=np.float32)
        cluster_norm_coords = np.zeros((m_max, 2), dtype=np.float32)
        cluster_mask = np.zeros((m_max,), dtype=np.float32)
        unit_to_cluster = np.full((n_agents,), -1, dtype=np.int64)
        unit_rel_cluster = np.zeros((n_agents, 2), dtype=np.float32)
        centroids_xy = np.zeros((m_max, 2), dtype=np.float32)
        centroids_valid = np.zeros((m_max,), dtype=bool)

        if not alive_units:
            return ClusterAssignment(
                0, cluster_feats, cluster_norm_coords, cluster_mask,
                unit_to_cluster, unit_rel_cluster, centroids_xy, centroids_valid,
            )

        all_units = obs.friendly + obs.enemy
        battle_cx = float(np.mean([u.x for u in all_units]))
        battle_cy = float(np.mean([u.y for u in all_units]))
        world_xy = np.array([[u.x, u.y] for u in alive_units], dtype=np.float32)
        span = max(
            float(world_xy[:, 0].max() - world_xy[:, 0].min()),
            float(world_xy[:, 1].max() - world_xy[:, 1].min()),
            4.0,
        )

        if recluster or self._cached_unit_cluster is None:
            k = max(1, min(int(requested_clusters), m_max, len(alive_units)))
            feats = self._unit_cluster_features(alive_units, battle_cx, battle_cy, span)
            init_xy = None
            if self._prev_centroids is not None and self._prev_valid is not None and self._prev_valid.any():
                init_xy = self._prev_centroids[self._prev_valid]
            labels = self._run_lloyd(feats, k, init_xy, world_xy)
            raw_groups = [
                [u for u, lbl in zip(alive_units, labels) if int(lbl) == g]
                for g in sorted(set(int(v) for v in labels))
            ]
            slot_map, centroids_xy, centroids_valid = self._match_cluster_slots(raw_groups)
            self._prev_centroids = centroids_xy.copy()
            self._prev_valid = centroids_valid.copy()
            self._cached_unit_cluster = {
                u.tag: slot for slot, grp in slot_map.items() for u in grp
            }
        else:
            # Keep previous macro-step cluster membership for alive units
            slot_map = {}
            for u in alive_units:
                slot = self._cached_unit_cluster.get(u.tag, 0)
                slot = min(max(int(slot), 0), m_max - 1)
                slot_map.setdefault(slot, []).append(u)
            for slot, grp in slot_map.items():
                centroids_xy[slot] = [np.mean([u.x for u in grp]), np.mean([u.y for u in grp])]
                centroids_valid[slot] = True

        # Populate per-cluster features (friendly stats + local enemy threats)
        n_alive_f = max(len(obs.friendly), 1)
        n_alive_e = max(len(obs.enemy), 1)
        enemies = obs.enemy
        for slot, grp in slot_map.items():
            if not grp:
                continue
            cluster_mask[slot] = 1.0
            gx_c = float(centroids_xy[slot, 0])
            gy_c = float(centroids_xy[slot, 1])
            nx, ny = im_grid.world_to_norm(gx_c, gy_c)
            cluster_norm_coords[slot] = [nx, ny]

            xs = np.array([u.x for u in grp], dtype=np.float32)
            ys = np.array([u.y for u in grp], dtype=np.float32)
            hps = np.array([u.effective_health / max(u.effective_health_max, 1.0) for u in grp], dtype=np.float32)
            shs = np.array(
                [u.shield / u.shield_max if u.shield_max > 0 else 0.0 for u in grp], dtype=np.float32
            )
            ready = np.array([1.0 if u.cooldown <= 1.0 else 0.0 for u in grp], dtype=np.float32)
            ranges = np.array([0.0 if u.is_medivac else u.attack_range for u in grp], dtype=np.float32)
            melee_r = float(np.mean([1.0 if (not u.is_medivac and u.attack_range < 2.0) else 0.0 for u in grp]))
            med_r = float(np.mean([1.0 if u.is_medivac else 0.0 for u in grp]))
            ranged_r = max(0.0, 1.0 - melee_r - med_r)

            # Local enemy summary around this cluster
            if enemies:
                ex = np.array([e.x for e in enemies], dtype=np.float32)
                ey = np.array([e.y for e in enemies], dtype=np.float32)
                edists = np.hypot(ex - gx_c, ey - gy_c)
                nearest_idx = int(np.argmin(edists))
                ne_dx = float(np.clip((ex[nearest_idx] - gx_c) / 12.0, -2.0, 2.0))
                ne_dy = float(np.clip((ey[nearest_idx] - gy_c) / 12.0, -2.0, 2.0))
                ne_dist = float(np.clip(edists[nearest_idx] / 12.0, 0.0, 2.0))

                # Count enemies reachable by at least one unit in this cluster & threatening enemies
                in_reach_e = 0
                weakest_hp = 1.0
                for e in enemies:
                    if any(
                        (not u.is_medivac) and np.hypot(u.x - e.x, u.y - e.y) <= max(u.attack_range, 1.0) + 1.0
                        for u in grp
                    ):
                        in_reach_e += 1
                        weakest_hp = min(weakest_hp, e.effective_health / max(e.effective_health_max, 1.0))
                threat_e = sum(
                    1 for e in enemies
                    if (not e.is_medivac)
                    and any(np.hypot(u.x - e.x, u.y - e.y) <= max(e.attack_range, 1.0) + 1.0 for u in grp)
                )
                in_reach_ratio = in_reach_e / n_alive_e
                threat_ratio = threat_e / n_alive_e
            else:
                ne_dx = ne_dy = ne_dist = in_reach_ratio = threat_ratio = 0.0
                weakest_hp = 0.0

            cluster_feats[slot] = np.array([
                float(np.clip((gx_c - battle_cx) / 12.0, -2.0, 2.0)),
                float(np.clip((gy_c - battle_cy) / 12.0, -2.0, 2.0)),
                float(np.clip(xs.std() / 6.0, 0.0, 2.0)),
                float(np.clip(ys.std() / 6.0, 0.0, 2.0)),
                len(grp) / max(float(n_agents), 1.0),
                len(grp) / float(n_alive_f),
                float(hps.mean()),
                float(hps.min()),
                float(shs.mean()),
                float(ready.mean()),
                float(np.clip(ranges.mean() / 6.0, 0.0, 1.5)),
                melee_r,
                ranged_r,
                med_r,
                ne_dx,
                ne_dy,
                ne_dist,
                float(in_reach_ratio),
                float(threat_ratio),
                float(weakest_hp),
            ], dtype=np.float32)

        # Map each fixed agent slot to its assigned cluster and relative cluster offset
        for i, tag in enumerate(agent_tags):
            if tag is None or tag not in friendly_by_tag:
                continue
            slot = int(self._cached_unit_cluster.get(tag, 0))
            if slot not in slot_map:
                slot = next(iter(slot_map))
            unit_to_cluster[i] = slot
            u = friendly_by_tag[tag]
            unit_rel_cluster[i, 0] = float(np.clip((u.x - centroids_xy[slot, 0]) / 8.0, -2.0, 2.0))
            unit_rel_cluster[i, 1] = float(np.clip((u.y - centroids_xy[slot, 1]) / 8.0, -2.0, 2.0))

        return ClusterAssignment(
            num_clusters=int(cluster_mask.sum()),
            cluster_feats=cluster_feats,
            cluster_norm_coords=cluster_norm_coords,
            cluster_mask=cluster_mask,
            unit_to_cluster=unit_to_cluster,
            unit_rel_cluster=unit_rel_cluster,
            centroids_xy=centroids_xy,
            centroids_valid=centroids_valid,
        )
