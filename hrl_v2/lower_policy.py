"""Unified DyIM-CHRL neural policy: Upper IM-VQ + Guidance + FiLM Cluster-Conditioned MAPPO.

Zero manual macros: the lower shared-parameter GRU-Actor outputs native SC2BA/SMAC
atomic unit actions masked by legal avail_actions, modulated via bounded FiLM from the
upper cluster guidance embedding w_{t, G(i)}.
"""

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from .config import HRLv2Config
from .guidance import BaseGuidanceHead, GuidanceSample, build_guidance_head
from .tactical_cluster import CLUSTER_STAT_DIM
from .upper_abstraction import AbstractionOutput, OnlineIMAbstractor, TacticalIMBackbone


class FiLMClusterActor(nn.Module):
    """Parameter-shared unit-level GRU actor modulated by its cluster's upper guidance."""

    def __init__(
        self,
        obs_dim: int,
        cluster_feat_dim: int,
        guidance_dim: int,
        hidden_dim: int,
        n_actions: int,
        use_guidance: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_guidance = use_guidance
        # obs_i + 4 unit_extra + 2 unit_rel_cluster + cluster_feat
        in_dim = obs_dim + 4 + 2 + cluster_feat_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        if use_guidance:
            self.film = nn.Linear(guidance_dim, 2 * hidden_dim)
            # Initialize FiLM near identity so early micro learning stays stable
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.action_head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        unit_obs: torch.Tensor,         # [B, N, obs_dim]
        unit_extra: torch.Tensor,       # [B, N, 4]
        unit_rel_cluster: torch.Tensor, # [B, N, 2]
        unit_cluster_feat: torch.Tensor,# [B, N, D_c]
        unit_guidance: torch.Tensor,    # [B, N, D_g]
        rnn_hidden: torch.Tensor,       # [B, N, H]
        avail_actions: torch.Tensor,    # [B, N, A]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_agents, _ = unit_obs.shape
        x = torch.cat([unit_obs, unit_extra, unit_rel_cluster, unit_cluster_feat], dim=-1)
        x_flat = self.encoder(x.reshape(bsz * n_agents, -1))
        h_next = self.gru(x_flat, rnn_hidden.reshape(bsz * n_agents, self.hidden_dim))
        if self.use_guidance:
            g_flat = unit_guidance.reshape(bsz * n_agents, -1)
            gamma, beta = self.film(g_flat).chunk(2, dim=-1)
            h_mod = (1.0 + 0.5 * torch.tanh(gamma)) * h_next + 0.5 * beta
        else:
            h_mod = h_next
        logits = self.action_head(h_mod).reshape(bsz, n_agents, -1)
        masked_logits = logits.masked_fill(avail_actions <= 0.5, -1e9)
        return masked_logits, h_next.reshape(bsz, n_agents, self.hidden_dim)


class DyIMCHRLPolicy(nn.Module):
    """Complete two-timescale DyIM-CHRL policy and centralized dual critics."""

    def __init__(self, cfg: HRLv2Config, obs_dim: int, state_dim: int, n_actions: int):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions

        # 1. Upper online IM-CNN + Dynamic VQ Prototype Abstraction
        self.abstractor = OnlineIMAbstractor(cfg)
        self.fused_cluster_dim = CLUSTER_STAT_DIM + TacticalIMBackbone.SPATIAL_DIM

        # 2. Upper granularity selector (chooses M_t in {1..max_clusters} from abstract prototype)
        self.granularity_head = nn.Sequential(
            nn.Linear(cfg.proto_dim, 48),
            nn.SiLU(),
            nn.Linear(48, cfg.max_clusters),
        )

        # 3. Extensible Upper -> Lower cluster guidance module (Option A default)
        self.guidance_head: BaseGuidanceHead = build_guidance_head(cfg, self.fused_cluster_dim)

        # 4. Upper centralized macro-critic V_high(z_q, global_state)
        self.upper_critic = nn.Sequential(
            nn.Linear(cfg.proto_dim + state_dim, cfg.critic_hidden_dim),
            nn.LayerNorm(cfg.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )

        # 5. Lower cluster-conditioned GRU actor (zero manual macros)
        self.lower_actor = FiLMClusterActor(
            obs_dim=obs_dim,
            cluster_feat_dim=self.fused_cluster_dim,
            guidance_dim=cfg.guidance_dim,
            hidden_dim=cfg.lower_hidden_dim,
            n_actions=n_actions,
            use_guidance=cfg.use_guidance,
        )

        # 6. Lower centralized CTDE critic V_low(global_state, obs_i, cluster_feat, guidance)
        low_critic_in = state_dim + obs_dim + self.fused_cluster_dim + cfg.guidance_dim
        self.lower_critic = nn.Sequential(
            nn.Linear(low_critic_in, cfg.critic_hidden_dim),
            nn.LayerNorm(cfg.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )

    def encode_upper(self, im_grid: torch.Tensor) -> AbstractionOutput:
        return self.abstractor(im_grid)

    def fuse_cluster_features(
        self,
        spatial_feat: torch.Tensor,
        cluster_feats: torch.Tensor,
        cluster_norm_coords: torch.Tensor,
        cluster_mask: torch.Tensor,
    ) -> torch.Tensor:
        local_im = self.abstractor.sample_cluster_spatial_feats(
            spatial_feat, cluster_norm_coords, cluster_mask
        )
        fused = torch.cat([cluster_feats, local_im], dim=-1)
        return fused * cluster_mask.unsqueeze(-1).float()

    def sample_granularity(
        self,
        proto_emb: torch.Tensor,
        alive_count: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select cluster count index in {0..M_max-1} (actual clusters = idx + 1), masked by alive_count."""
        logits = self.granularity_head(proto_emb)  # [B, M_max]
        bsz, m_max = logits.shape
        choices = torch.arange(1, m_max + 1, device=logits.device).unsqueeze(0).expand(bsz, -1)
        valid = choices <= alive_count.unsqueeze(1).clamp_min(1)
        masked = logits.masked_fill(~valid, -1e9)
        dist = Categorical(logits=masked)
        idx = masked.argmax(dim=-1) if deterministic else dist.sample()
        return idx, dist.log_prob(idx), dist.entropy()

    def evaluate_granularity(
        self,
        proto_emb: torch.Tensor,
        alive_count: torch.Tensor,
        gran_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.granularity_head(proto_emb)
        bsz, m_max = logits.shape
        choices = torch.arange(1, m_max + 1, device=logits.device).unsqueeze(0).expand(bsz, -1)
        valid = choices <= alive_count.unsqueeze(1).clamp_min(1)
        masked = logits.masked_fill(~valid, -1e9)
        dist = Categorical(logits=masked)
        return dist.log_prob(gran_idx.long()), dist.entropy()

    def value_upper(self, proto_emb: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        return self.upper_critic(torch.cat([proto_emb, global_state], dim=-1)).squeeze(-1)

    @staticmethod
    def gather_unit_cluster_tensors(
        fused_cluster_feats: torch.Tensor,  # [B, M_max, D_fc]
        cluster_guidance: torch.Tensor,     # [B, M_max, D_g]
        unit_to_cluster: torch.Tensor,      # [B, N] int64 (-1 if dead)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather each unit's assigned cluster summary and upper guidance embedding."""
        alive = (unit_to_cluster >= 0).float().unsqueeze(-1)  # [B, N, 1]
        safe_idx = unit_to_cluster.clamp(min=0)               # [B, N]
        d_fc = fused_cluster_feats.shape[-1]
        d_g = cluster_guidance.shape[-1]
        u_cf = torch.gather(fused_cluster_feats, 1, safe_idx.unsqueeze(-1).expand(-1, -1, d_fc)) * alive
        u_gd = torch.gather(cluster_guidance, 1, safe_idx.unsqueeze(-1).expand(-1, -1, d_g)) * alive
        return u_cf, u_gd

    def act_lower(
        self,
        unit_obs: torch.Tensor,
        unit_extra: torch.Tensor,
        unit_rel_cluster: torch.Tensor,
        fused_cluster_feats: torch.Tensor,
        cluster_guidance: torch.Tensor,
        unit_to_cluster: torch.Tensor,
        rnn_hidden: torch.Tensor,
        avail_actions: torch.Tensor,
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u_cf, u_gd = self.gather_unit_cluster_tensors(fused_cluster_feats, cluster_guidance, unit_to_cluster)
        logits, h_next = self.lower_actor(
            unit_obs, unit_extra, unit_rel_cluster, u_cf, u_gd, rnn_hidden, avail_actions
        )
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        bsz, n_agents, _ = unit_obs.shape
        g_exp = global_state.unsqueeze(1).expand(-1, n_agents, -1)
        v_in = torch.cat([g_exp, unit_obs, u_cf, u_gd], dim=-1)
        values = self.lower_critic(v_in).squeeze(-1)  # [B, N]
        return actions, log_probs, entropy, values, h_next

    def evaluate_lower(
        self,
        unit_obs: torch.Tensor,
        unit_extra: torch.Tensor,
        unit_rel_cluster: torch.Tensor,
        unit_cluster_feat: torch.Tensor,
        unit_guidance: torch.Tensor,
        rnn_hidden: torch.Tensor,
        avail_actions: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _ = self.lower_actor(
            unit_obs, unit_extra, unit_rel_cluster, unit_cluster_feat, unit_guidance, rnn_hidden, avail_actions
        )
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())
        entropy = dist.entropy()
        bsz, n_agents, _ = unit_obs.shape
        g_exp = global_state.unsqueeze(1).expand(-1, n_agents, -1)
        v_in = torch.cat([g_exp, unit_obs, unit_cluster_feat, unit_guidance], dim=-1)
        values = self.lower_critic(v_in).squeeze(-1)
        return log_probs, entropy, values

    def init_lower_hidden(self, batch_size: int, n_agents: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, n_agents, self.cfg.lower_hidden_dim), dtype=torch.float32, device=device)

    def upper_parameters(self):
        return (
            list(self.abstractor.parameters())
            + list(self.granularity_head.parameters())
            + list(self.guidance_head.parameters())
        )

    def actor_parameters(self):
        return list(self.lower_actor.parameters())

    def critic_parameters(self):
        return list(self.upper_critic.parameters()) + list(self.lower_critic.parameters())
