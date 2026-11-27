"""Online IM-CNN + EMA-VQ Dynamic Prototype Abstraction with Bisimulation Split/Merge.

Replaces offline probe collection + frozen K=32 medoids with an end-to-end online
abstraction tightly coupled to the multi-channel tactical Influence Map CNN:
  1. IM-CNN encodes [C, 64, 64] tactical influence grids into:
       - L2-normalized latent z_t in S^{D-1} (scale-invariant across all scenarios)
       - spatial feature map F_t in R^{32 x 8 x 8} for per-cluster grid_sample fusion
  2. DynamicPrototypeCodebook maintains K_t in [K_min, K_max] active prototypes on S^{D-1}
     updated via EMA (no offline data needed; warm-initialized from first online rollout).
  3. Online losses:
       - VQ commitment loss + Straight-Through Estimator (STE)
       - Online reward & next-value prediction from (z_q, upper_action_summary)
       - Pairwise value-reward bisimulation metric loss aligning ||z_i - z_j||_2
  4. Periodic Split / Merge / Prune driven by per-prototype TD-error variance,
     macro-return variance, latent spread, and bisimulation equivalence.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import HRLv2Config


@dataclass
class AbstractionOutput:
    z_cont: torch.Tensor        # [B, D] continuous unit-norm latent
    z_quant: torch.Tensor       # [B, D] STE quantized prototype embedding
    proto_id: torch.Tensor      # [B] int64 slot index in [0, max_prototypes)
    spatial_feat: torch.Tensor  # [B, 32, 8, 8] intermediate IM-CNN feature map
    commit_loss: torch.Tensor   # scalar tensor
    dist_min: torch.Tensor      # [B] distance to assigned prototype


class TacticalIMBackbone(nn.Module):
    """Lightweight 4-layer 2D CNN tailored for [C, 64, 64] tactical influence maps."""

    SPATIAL_DIM = 32

    def __init__(self, in_channels: int, proto_dim: int, im_size: int = 64):
        super().__init__()
        if im_size % 16 != 0:
            raise ValueError(f"im_size must be divisible by 16, got {im_size}.")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 24, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, self.SPATIAL_DIM, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, self.SPATIAL_DIM),
            nn.SiLU(),
        )
        # 64 -> 32 -> 16 -> 8 (spatial_feat: [B, 32, 8, 8])
        self.down4 = nn.Sequential(
            nn.Conv2d(self.SPATIAL_DIM, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        flat_dim = 32 * (im_size // 16) * (im_size // 16)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, proto_dim),
        )

    def forward(self, im_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial = self.stem(im_grid)
        deep = self.down4(spatial)
        z = F.normalize(self.proj(deep), p=2, dim=-1, eps=1e-6)
        return z, spatial


class DynamicPrototypeCodebook(nn.Module):
    """Online EMA-VQ prototype pool with TD-error / bisimulation Split, Merge & Prune."""

    def __init__(self, cfg: HRLv2Config):
        super().__init__()
        self.cfg = cfg
        m = cfg.max_prototypes
        d = cfg.proto_dim

        # Initialize random unit vectors for the first init_prototypes slots
        init_vecs = F.normalize(torch.randn(m, d), p=2, dim=-1)
        active = torch.zeros(m, dtype=torch.bool)
        active[: cfg.init_prototypes] = True

        self.register_buffer("prototypes", init_vecs)
        self.register_buffer("active_mask", active)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("ema_count", torch.zeros(m, dtype=torch.float32))
        self.register_buffer("ema_sum", init_vecs.clone())
        self.register_buffer("ema_td_mean", torch.zeros(m, dtype=torch.float32))
        self.register_buffer("ema_td_var", torch.zeros(m, dtype=torch.float32))
        self.register_buffer("ema_ret_mean", torch.zeros(m, dtype=torch.float32))
        self.register_buffer("ema_ret_var", torch.zeros(m, dtype=torch.float32))
        self.register_buffer("ema_val_mean", torch.zeros(m, dtype=torch.float32))
        self.register_buffer("ema_spread", torch.zeros(m, dtype=torch.float32))
        self.register_buffer("proto_age", torch.zeros(m, dtype=torch.long))
        self.register_buffer("proto_cooldown", torch.zeros(m, dtype=torch.long))
        self.register_buffer("total_splits", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_merges", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_prunes", torch.tensor(0, dtype=torch.long))

        # Per-prototype recent (z, td_error) ring buffers on CPU for split axis estimation
        self._recent_samples: Dict[int, deque] = {
            k: deque(maxlen=cfg.proto_buffer_size) for k in range(m)
        }

    @property
    def num_active(self) -> int:
        return int(self.active_mask.sum().item())

    def active_indices(self) -> torch.Tensor:
        return torch.nonzero(self.active_mask, as_tuple=False).flatten()

    @torch.no_grad()
    def warm_initialize(self, z_batch: torch.Tensor) -> bool:
        """Online K-Means++ initialization on the first rollout batch (zero offline data)."""
        if bool(self.initialized.item()):
            return False
        z = F.normalize(z_batch.detach(), p=2, dim=-1)
        n = z.shape[0]
        k_init = min(self.cfg.init_prototypes, n)
        if k_init < self.cfg.min_prototypes:
            return False

        # Deterministic-friendly K-Means++ selection on unit hypersphere
        centers = [z[0]]
        for _ in range(1, k_init):
            stacked = torch.stack(centers, dim=0)
            dists = torch.cdist(z, stacked).min(dim=1).values
            probs = (dists * dists).clamp_min(1e-8)
            probs = probs / probs.sum()
            chosen = int(torch.multinomial(probs, 1).item())
            centers.append(z[chosen])
        centers_t = F.normalize(torch.stack(centers, dim=0), p=2, dim=-1)

        # 5 Lloyd steps on hypersphere
        for _ in range(5):
            assign = torch.cdist(z, centers_t).argmin(dim=1)
            for idx in range(k_init):
                members = z[assign == idx]
                if members.shape[0] > 0:
                    centers_t[idx] = F.normalize(members.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)

        self.active_mask.zero_()
        self.active_mask[:k_init] = True
        self.prototypes[:k_init].copy_(centers_t)
        self.ema_sum[:k_init].copy_(centers_t)
        self.ema_count.zero_()
        self.ema_count[:k_init] = 1.0
        self.proto_age.zero_()
        self.proto_cooldown.zero_()
        self.initialized.fill_(True)
        for q in self._recent_samples.values():
            q.clear()
        return True

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign each unit-norm z to nearest active prototype; return (z_q_ste, proto_id, commit_loss, dist_min)."""
        active_idx = self.active_indices()
        active_protos = self.prototypes[active_idx]  # [K_t, D]
        # Squared Euclidean distance on unit hypersphere
        dists = torch.cdist(z, active_protos, p=2.0)  # [B, K_t]
        min_dist, local_idx = dists.min(dim=1)
        proto_id = active_idx[local_idx]
        matched = self.prototypes[proto_id]  # [B, D]
        commit_loss = F.mse_loss(z, matched.detach())
        z_q = z + (matched - z).detach()  # Straight-Through Estimator
        return z_q, proto_id, commit_loss, min_dist

    @torch.no_grad()
    def observe_online_stats(
        self,
        z: torch.Tensor,
        proto_id: torch.Tensor,
        td_errors: torch.Tensor,
        macro_returns: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Update EMA prototypes and per-prototype value/TD/return heterogeneity statistics."""
        z_det = F.normalize(z.detach(), p=2, dim=-1)
        pid = proto_id.detach().long()
        td = td_errors.detach().float()
        ret = macro_returns.detach().float()
        val = values.detach().float()

        decay = self.cfg.ema_decay
        stat_alpha = 0.15  # responsive EMA for variance diagnostics

        # Age & cooldown progression once per PPO rollout update
        active_idx = self.active_indices()
        self.proto_age[active_idx] += 1
        self.proto_cooldown[active_idx] = torch.clamp(self.proto_cooldown[active_idx] - 1, min=0)
        # Global gentle count decay so unused codes become detectable
        self.ema_count[active_idx] *= decay

        z_cpu = z_det.cpu().numpy()
        td_cpu = td.cpu().numpy()
        pid_cpu = pid.cpu().numpy()

        for slot in active_idx.tolist():
            mask = pid == slot
            count = int(mask.sum().item())
            if count == 0:
                continue
            z_m = z_det[mask]
            td_m = td[mask]
            ret_m = ret[mask]
            val_m = val[mask]

            # EMA prototype update on unit hypersphere
            self.ema_count[slot] += float(count) * (1.0 - decay) + float(count) * 0.25
            batch_mean_z = z_m.mean(dim=0)
            self.ema_sum[slot].mul_(decay).add_(batch_mean_z, alpha=1.0 - decay)
            self.prototypes[slot].copy_(F.normalize(self.ema_sum[slot].unsqueeze(0), p=2, dim=-1).squeeze(0))

            # Heterogeneity diagnostics
            td_mean_b = float(td_m.mean().item())
            td_var_b = float(((td_m - self.ema_td_mean[slot]) ** 2).mean().item())
            ret_mean_b = float(ret_m.mean().item())
            ret_var_b = float(((ret_m - self.ema_ret_mean[slot]) ** 2).mean().item())
            val_mean_b = float(val_m.mean().item())
            spread_b = float(((z_m - self.prototypes[slot].unsqueeze(0)) ** 2).sum(dim=-1).mean().item())

            self.ema_td_mean[slot] = (1.0 - stat_alpha) * self.ema_td_mean[slot] + stat_alpha * td_mean_b
            self.ema_td_var[slot] = (1.0 - stat_alpha) * self.ema_td_var[slot] + stat_alpha * td_var_b
            self.ema_ret_mean[slot] = (1.0 - stat_alpha) * self.ema_ret_mean[slot] + stat_alpha * ret_mean_b
            self.ema_ret_var[slot] = (1.0 - stat_alpha) * self.ema_ret_var[slot] + stat_alpha * ret_var_b
            self.ema_val_mean[slot] = (1.0 - stat_alpha) * self.ema_val_mean[slot] + stat_alpha * val_mean_b
            self.ema_spread[slot] = (1.0 - stat_alpha) * self.ema_spread[slot] + stat_alpha * spread_b

            # Push recent samples to CPU buffer for principled TD-axis splitting
            buf = self._recent_samples[slot]
            idxs = np.flatnonzero(pid_cpu == slot)
            for idx in idxs[-self.cfg.proto_buffer_size :]:
                buf.append((z_cpu[idx], float(td_cpu[idx])))

    @torch.no_grad()
    def maybe_split_and_merge(self, update_idx: int) -> List[dict]:
        """Adapt prototype count K_t via TD/return-driven Split, Bisim Merge, and Dead Prune."""
        if not self.cfg.dynamic_codebook or not bool(self.initialized.item()):
            return []
        events: List[dict] = []

        # 1. Prune dead codes (low usage & aged, keeping >= min_prototypes)
        active_idx = self.active_indices().tolist()
        for slot in active_idx:
            if self.num_active <= self.cfg.min_prototypes:
                break
            if (
                int(self.proto_age[slot].item()) >= self.cfg.dead_proto_age
                and int(self.proto_cooldown[slot].item()) == 0
                and float(self.ema_count[slot].item()) < self.cfg.dead_proto_count
            ):
                self.active_mask[slot] = False
                self._recent_samples[slot].clear()
                self.total_prunes += 1
                events.append({
                    "update": update_idx,
                    "event": "prune",
                    "slot": slot,
                    "count": round(float(self.ema_count[slot].item()), 3),
                    "k_after": self.num_active,
                })

        # 2. Split highest value/return heterogeneity prototype (at most 1 per cycle for stability)
        if self.num_active < self.cfg.max_prototypes:
            candidates = []
            active_idx = self.active_indices().tolist()
            td_std_all = torch.sqrt(self.ema_td_var[active_idx].clamp_min(1e-8))
            ret_std_all = torch.sqrt(self.ema_ret_var[active_idx].clamp_min(1e-8))
            td_scale = float(td_std_all.mean().item()) + 1e-4
            ret_scale = float(ret_std_all.mean().item()) + 1e-4

            for slot in active_idx:
                if (
                    float(self.ema_count[slot].item()) < self.cfg.min_proto_count
                    or int(self.proto_cooldown[slot].item()) > 0
                    or len(self._recent_samples[slot]) < 8
                ):
                    continue
                td_norm = float(torch.sqrt(self.ema_td_var[slot].clamp_min(1e-8)).item()) / td_scale
                ret_norm = float(torch.sqrt(self.ema_ret_var[slot].clamp_min(1e-8)).item()) / ret_scale
                spread = float(torch.sqrt(self.ema_spread[slot].clamp_min(1e-8)).item())
                score = 0.5 * (td_norm + ret_norm) * spread
                candidates.append((score, slot, td_norm, ret_norm, spread))

            if candidates:
                candidates.sort(reverse=True)
                best_score, src_slot, td_norm, ret_norm, spread = candidates[0]
                if best_score >= self.cfg.split_score_threshold:
                    free_slots = torch.nonzero(~self.active_mask, as_tuple=False).flatten()
                    if free_slots.numel() > 0:
                        dst_slot = int(free_slots[0].item())
                        child_a, child_b = self._compute_split_children(src_slot)
                        self.prototypes[src_slot].copy_(child_a)
                        self.ema_sum[src_slot].copy_(child_a)
                        self.prototypes[dst_slot].copy_(child_b)
                        self.ema_sum[dst_slot].copy_(child_b)
                        self.active_mask[dst_slot] = True

                        half_count = max(self.cfg.min_proto_count * 0.5, float(self.ema_count[src_slot].item()) * 0.5)
                        for s in (src_slot, dst_slot):
                            self.ema_count[s] = half_count
                            self.ema_td_var[s] *= 0.5
                            self.ema_ret_var[s] *= 0.5
                            self.ema_spread[s] *= 0.5
                            self.proto_age[s] = 0
                            self.proto_cooldown[s] = self.cfg.split_cooldown
                        self.ema_td_mean[dst_slot] = self.ema_td_mean[src_slot]
                        self.ema_ret_mean[dst_slot] = self.ema_ret_mean[src_slot]
                        self.ema_val_mean[dst_slot] = self.ema_val_mean[src_slot]
                        self._recent_samples[src_slot].clear()
                        self._recent_samples[dst_slot].clear()
                        self.total_splits += 1
                        events.append({
                            "update": update_idx,
                            "event": "split",
                            "parent_slot": src_slot,
                            "child_slot": dst_slot,
                            "score": round(best_score, 4),
                            "td_norm": round(td_norm, 3),
                            "ret_norm": round(ret_norm, 3),
                            "spread": round(spread, 4),
                            "k_after": self.num_active,
                        })

        # 3. Bisimulation Merge (closest pair in latent space that is also value/return-equivalent)
        if self.num_active > self.cfg.min_prototypes:
            eligible = [
                slot for slot in self.active_indices().tolist()
                if int(self.proto_cooldown[slot].item()) == 0 and int(self.proto_age[slot].item()) >= 2
            ]
            if len(eligible) >= 2:
                e_t = torch.tensor(eligible, device=self.prototypes.device, dtype=torch.long)
                vecs = self.prototypes[e_t]
                pdist = torch.cdist(vecs, vecs, p=2.0)
                pdist.fill_diagonal_(1e9)
                best_pair = None
                best_dist = self.cfg.merge_dist_threshold
                for i in range(len(eligible)):
                    for j in range(i + 1, len(eligible)):
                        d_ij = float(pdist[i, j].item())
                        if d_ij < best_dist:
                            si, sj = eligible[i], eligible[j]
                            val_gap = abs(float(self.ema_val_mean[si].item() - self.ema_val_mean[sj].item()))
                            ret_gap = abs(float(self.ema_ret_mean[si].item() - self.ema_ret_mean[sj].item()))
                            if (val_gap + ret_gap) <= self.cfg.merge_value_threshold:
                                best_dist = d_ij
                                best_pair = (si, sj, d_ij, val_gap + ret_gap)
                if best_pair is not None:
                    si, sj, d_ij, bisim_gap = best_pair
                    wi = max(float(self.ema_count[si].item()), 1e-3)
                    wj = max(float(self.ema_count[sj].item()), 1e-3)
                    w_tot = wi + wj
                    merged_vec = F.normalize(
                        (wi * self.prototypes[si] + wj * self.prototypes[sj]).unsqueeze(0), p=2, dim=-1
                    ).squeeze(0)
                    self.prototypes[si].copy_(merged_vec)
                    self.ema_sum[si].copy_(merged_vec)
                    self.ema_count[si] = w_tot
                    self.ema_val_mean[si] = (wi * self.ema_val_mean[si] + wj * self.ema_val_mean[sj]) / w_tot
                    self.ema_ret_mean[si] = (wi * self.ema_ret_mean[si] + wj * self.ema_ret_mean[sj]) / w_tot
                    self.proto_cooldown[si] = self.cfg.split_cooldown
                    self.active_mask[sj] = False
                    self._recent_samples[sj].clear()
                    self.total_merges += 1
                    events.append({
                        "update": update_idx,
                        "event": "merge",
                        "keep_slot": si,
                        "drop_slot": sj,
                        "latent_dist": round(d_ij, 4),
                        "bisim_gap": round(bisim_gap, 4),
                        "k_after": self.num_active,
                    })

        return events

    def _compute_split_children(self, slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a prototype along its TD-error separation axis (fallback: principal axis)."""
        device = self.prototypes.device
        parent = self.prototypes[slot]
        samples = list(self._recent_samples[slot])
        zs = np.stack([item[0] for item in samples], axis=0)
        tds = np.asarray([item[1] for item in samples], dtype=np.float32)

        med = float(np.median(tds))
        hi = zs[tds >= med]
        lo = zs[tds < med]
        if len(hi) >= 2 and len(lo) >= 2:
            axis = hi.mean(axis=0) - lo.mean(axis=0)
        else:
            centered = zs - zs.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-5:
            axis = np.random.default_rng(slot + 17).standard_normal(zs.shape[1]).astype(np.float32)
            norm = float(np.linalg.norm(axis))
        direction = torch.from_numpy((axis / max(norm, 1e-6)).astype(np.float32)).to(device)
        step = max(0.08, min(0.25, 0.5 * float(torch.sqrt(self.ema_spread[slot].clamp_min(1e-6)).item())))
        child_a = F.normalize((parent + step * direction).unsqueeze(0), p=2, dim=-1).squeeze(0)
        child_b = F.normalize((parent - step * direction).unsqueeze(0), p=2, dim=-1).squeeze(0)
        return child_a, child_b

    def diagnostics(self) -> dict:
        active_idx = self.active_indices()
        counts = self.ema_count[active_idx].detach().cpu().numpy()
        probs = counts / max(float(counts.sum()), 1e-8)
        usage_entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if len(probs) else 0.0
        return {
            "active_prototypes": int(active_idx.numel()),
            "initialized": bool(self.initialized.item()),
            "usage_entropy": round(usage_entropy, 4),
            "mean_td_std": round(float(torch.sqrt(self.ema_td_var[active_idx].clamp_min(0.0)).mean().item()), 4),
            "mean_ret_std": round(float(torch.sqrt(self.ema_ret_var[active_idx].clamp_min(0.0)).mean().item()), 4),
            "mean_spread": round(float(self.ema_spread[active_idx].mean().item()), 4),
            "total_splits": int(self.total_splits.item()),
            "total_merges": int(self.total_merges.item()),
            "total_prunes": int(self.total_prunes.item()),
        }


class OnlineIMAbstractor(nn.Module):
    """Upper-level online IM-CNN + DynamicPrototypeCodebook + Bisimulation auxiliary heads."""

    def __init__(self, cfg: HRLv2Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = TacticalIMBackbone(cfg.im_channels, cfg.proto_dim, cfg.im_size)
        self.codebook = DynamicPrototypeCodebook(cfg)
        # Predict macro return and next macro value from (z_q, upper_action_summary)
        aux_in = cfg.proto_dim + cfg.max_clusters + cfg.guidance_dim
        self.dynamics_head = nn.Sequential(
            nn.Linear(aux_in, 64),
            nn.SiLU(),
            nn.Linear(64, 2),  # [pred_macro_return, pred_next_value]
        )

    def forward(self, im_grid: torch.Tensor) -> AbstractionOutput:
        z_cont, spatial_feat = self.backbone(im_grid)
        z_quant, proto_id, commit_loss, dist_min = self.codebook.quantize(z_cont)
        return AbstractionOutput(
            z_cont=z_cont,
            z_quant=z_quant,
            proto_id=proto_id,
            spatial_feat=spatial_feat,
            commit_loss=commit_loss,
            dist_min=dist_min,
        )

    def sample_cluster_spatial_feats(
        self,
        spatial_feat: torch.Tensor,
        cluster_norm_coords: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample IM-CNN spatial features at each internal cluster centroid.

        spatial_feat:        [B, 32, H', W']
        cluster_norm_coords: [B, M_max, 2] in [-1, 1]^2
        cluster_mask:        [B, M_max] float/bool
        returns:             [B, M_max, 32]
        """
        grid = cluster_norm_coords.unsqueeze(2)  # [B, M_max, 1, 2]
        sampled = F.grid_sample(
            spatial_feat, grid, mode="bilinear", padding_mode="border", align_corners=False
        )  # [B, 32, M_max, 1]
        out = sampled.squeeze(-1).transpose(1, 2)  # [B, M_max, 32]
        return out * cluster_mask.unsqueeze(-1).float()

    def auxiliary_losses(
        self,
        z_cont: torch.Tensor,
        z_quant: torch.Tensor,
        gran_onehot: torch.Tensor,
        pooled_guidance: torch.Tensor,
        macro_returns: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute online reward/next-value prediction loss and pairwise bisimulation loss."""
        aux_in = torch.cat([z_quant, gran_onehot, pooled_guidance], dim=-1)
        preds = self.dynamics_head(aux_in)
        pred_ret, pred_next_v = preds[:, 0], preds[:, 1]
        reward_pred_loss = (
            F.smooth_l1_loss(pred_ret, macro_returns.detach())
            + F.smooth_l1_loss(pred_next_v, next_values.detach())
        )

        # Pairwise bisimulation metric alignment on unit hypersphere (scale-normalized to [0, 2))
        bsz = z_cont.shape[0]
        if bsz >= 4:
            perm = torch.randperm(bsz, device=z_cont.device)
            z_dist = torch.norm(z_cont - z_cont[perm], p=2, dim=-1)
            r_diff = (macro_returns.detach() - macro_returns.detach()[perm]).abs()
            v_diff = (next_values.detach() - next_values.detach()[perm]).abs()
            gamma_tau = self.cfg.gamma ** self.cfg.macro_interval
            raw_bisim = r_diff + gamma_tau * v_diff
            scale = raw_bisim.std(unbiased=False).clamp_min(0.25)
            target_dist = 1.8 * torch.tanh(raw_bisim / (2.0 * scale))
            bisim_loss = F.mse_loss(z_dist, target_dist)
        else:
            bisim_loss = z_cont.new_zeros(())

        return reward_pred_loss, bisim_loss
