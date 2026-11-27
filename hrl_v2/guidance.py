"""Extensible Upper -> Lower cluster guidance interface.

Primary implementation (Option A): ``LatentSkillGuidanceHead`` — end-to-end emergent
discrete tactical skill codes z_{t,m} in {0..Z-1} assigned per internal cluster
(zero hand-crafted macros). Also ships ``ContinuousLatentGuidanceHead`` and a clean
``BaseGuidanceHead`` contract (including ``intrinsic_reward`` hook) so Option B
(spatial subgoals) or Option C (cluster mixing hypernets) can be plugged in cleanly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from .config import HRLv2Config


@dataclass
class GuidanceSample:
    """Output of upper guidance decision for a single macro step (or batch)."""

    cluster_guidance: torch.Tensor  # [B, M_max, D_g] guidance embedding per cluster
    raw_action: torch.Tensor        # [B, M_max] (int64 for discrete) or [B, M_max, D_g] (continuous)
    log_prob: torch.Tensor          # [B] summed over active clusters
    entropy: torch.Tensor           # [B] mean over active clusters
    diversity_loss: torch.Tensor    # scalar auxiliary encouraging distinct skills across clusters/states
    aux_info: Dict[str, torch.Tensor]


class BaseGuidanceHead(ABC, nn.Module):
    """Abstract contract connecting upper abstract prototypes + internal clusters to lower policy."""

    def __init__(self, proto_dim: int, cluster_feat_dim: int, guidance_dim: int):
        super().__init__()
        self.proto_dim = proto_dim
        self.cluster_feat_dim = cluster_feat_dim
        self.guidance_dim = guidance_dim

    @abstractmethod
    def sample(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> GuidanceSample:
        ...

    @abstractmethod
    def evaluate(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> GuidanceSample:
        ...

    def intrinsic_reward(
        self,
        prev_obs,
        next_obs,
        cluster_assign,
        raw_action: np.ndarray,
    ) -> float:
        """Optional shaping hook for future subgoal-based guidance (Option B); 0.0 for pure Option A."""
        return 0.0


class _ClusterContextEncoder(nn.Module):
    """Light self-attention across internal clusters conditioned on the global abstract prototype."""

    def __init__(self, proto_dim: int, cluster_feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(proto_dim + cluster_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        # proto_emb: [B, D_p], cluster_feats: [B, M, D_c], cluster_mask: [B, M]
        m = cluster_feats.shape[1]
        proto_exp = proto_emb.unsqueeze(1).expand(-1, m, -1)
        h = self.in_proj(torch.cat([proto_exp, cluster_feats], dim=-1))
        key_pad = cluster_mask <= 0.5
        # Guard against all-padded rows (terminal frames)
        all_pad = key_pad.all(dim=1)
        if all_pad.any():
            safe_pad = key_pad.clone()
            safe_pad[all_pad, 0] = False
        else:
            safe_pad = key_pad
        attn_out, _ = self.attn(h, h, h, key_padding_mask=safe_pad, need_weights=False)
        out = self.out_norm(h + attn_out)
        return out * cluster_mask.unsqueeze(-1).float()


class LatentSkillGuidanceHead(BaseGuidanceHead):
    """Option A (default): Emergent discrete latent tactical skill code per internal cluster."""

    def __init__(self, cfg: HRLv2Config, cluster_feat_dim: int):
        super().__init__(cfg.proto_dim, cluster_feat_dim, cfg.guidance_dim)
        self.num_skills = cfg.num_skills
        self.context = _ClusterContextEncoder(cfg.proto_dim, cluster_feat_dim, hidden_dim=64)
        self.logits_head = nn.Linear(64, cfg.num_skills)
        self.skill_embed = nn.Embedding(cfg.num_skills, cfg.guidance_dim)
        nn.init.orthogonal_(self.skill_embed.weight)

    def _forward_logits(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.context(proto_emb, cluster_feats, cluster_mask)
        return self.logits_head(h)  # [B, M, Z]

    def _diversity_loss(self, logits: torch.Tensor, cluster_mask: torch.Tensor) -> torch.Tensor:
        """Encourage balanced marginal skill usage across batches without forcing uniform per-state entropy."""
        mask = cluster_mask.float()
        active_total = mask.sum().clamp_min(1.0)
        probs = F.softmax(logits, dim=-1)  # [B, M, Z]
        marginal = (probs * mask.unsqueeze(-1)).sum(dim=(0, 1)) / active_total  # [Z]
        # Minimize KL(marginal || uniform) = sum p log p + log Z
        marg_entropy = -(marginal * torch.log(marginal.clamp_min(1e-8))).sum()
        target_entropy = float(np.log(self.num_skills))
        return (target_entropy - marg_entropy).clamp_min(0.0)

    def sample(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> GuidanceSample:
        logits = self._forward_logits(proto_emb, cluster_feats, cluster_mask)
        dist = Categorical(logits=logits)
        raw_action = logits.argmax(dim=-1) if deterministic else dist.sample()
        mask = cluster_mask.float()
        lp = (dist.log_prob(raw_action) * mask).sum(dim=-1)
        ent = (dist.entropy() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        emb = self.skill_embed(raw_action) * mask.unsqueeze(-1)
        div_loss = self._diversity_loss(logits, cluster_mask)
        return GuidanceSample(
            cluster_guidance=emb,
            raw_action=raw_action,
            log_prob=lp,
            entropy=ent,
            diversity_loss=div_loss,
            aux_info={"skill_ids": raw_action, "skill_probs": F.softmax(logits, dim=-1)},
        )

    def evaluate(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> GuidanceSample:
        logits = self._forward_logits(proto_emb, cluster_feats, cluster_mask)
        dist = Categorical(logits=logits)
        act = raw_action.long()
        mask = cluster_mask.float()
        lp = (dist.log_prob(act) * mask).sum(dim=-1)
        ent = (dist.entropy() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        emb = self.skill_embed(act) * mask.unsqueeze(-1)
        div_loss = self._diversity_loss(logits, cluster_mask)
        return GuidanceSample(
            cluster_guidance=emb,
            raw_action=act,
            log_prob=lp,
            entropy=ent,
            diversity_loss=div_loss,
            aux_info={"skill_ids": act, "skill_probs": F.softmax(logits, dim=-1)},
        )


class ContinuousLatentGuidanceHead(BaseGuidanceHead):
    """Extensible Option A (continuous variant): diagonal Gaussian guidance vector per cluster."""

    def __init__(self, cfg: HRLv2Config, cluster_feat_dim: int):
        super().__init__(cfg.proto_dim, cluster_feat_dim, cfg.guidance_dim)
        self.context = _ClusterContextEncoder(cfg.proto_dim, cluster_feat_dim, hidden_dim=64)
        self.mean_head = nn.Linear(64, cfg.guidance_dim)
        self.log_std = nn.Parameter(torch.full((cfg.guidance_dim,), -0.5))

    def _dist(self, proto_emb, cluster_feats, cluster_mask) -> Normal:
        h = self.context(proto_emb, cluster_feats, cluster_mask)
        mean = torch.tanh(self.mean_head(h))
        std = torch.exp(self.log_std.clamp(-2.5, 0.5)).view(1, 1, -1).expand_as(mean)
        return Normal(mean, std)

    def sample(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> GuidanceSample:
        dist = self._dist(proto_emb, cluster_feats, cluster_mask)
        raw_action = dist.mean if deterministic else dist.rsample()
        mask = cluster_mask.float()
        lp = (dist.log_prob(raw_action).sum(dim=-1) * mask).sum(dim=-1)
        ent = (dist.entropy().sum(dim=-1) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        emb = torch.tanh(raw_action) * mask.unsqueeze(-1)
        return GuidanceSample(
            cluster_guidance=emb,
            raw_action=raw_action,
            log_prob=lp,
            entropy=ent,
            diversity_loss=proto_emb.new_zeros(()),
            aux_info={},
        )

    def evaluate(
        self,
        proto_emb: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_mask: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> GuidanceSample:
        dist = self._dist(proto_emb, cluster_feats, cluster_mask)
        mask = cluster_mask.float()
        lp = (dist.log_prob(raw_action).sum(dim=-1) * mask).sum(dim=-1)
        ent = (dist.entropy().sum(dim=-1) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        emb = torch.tanh(raw_action) * mask.unsqueeze(-1)
        return GuidanceSample(
            cluster_guidance=emb,
            raw_action=raw_action,
            log_prob=lp,
            entropy=ent,
            diversity_loss=proto_emb.new_zeros(()),
            aux_info={},
        )


def build_guidance_head(cfg: HRLv2Config, cluster_feat_dim: int) -> BaseGuidanceHead:
    """Factory for pluggable upper->lower guidance interfaces."""
    if cfg.guidance_type == "latent_skill":
        return LatentSkillGuidanceHead(cfg, cluster_feat_dim)
    if cfg.guidance_type == "continuous_latent":
        return ContinuousLatentGuidanceHead(cfg, cluster_feat_dim)
    raise ValueError(f"Unsupported guidance_type '{cfg.guidance_type}'.")
