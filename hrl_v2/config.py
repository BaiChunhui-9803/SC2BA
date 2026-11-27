"""Configuration dataclasses for DyIM-CHRL (hrl_v2)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


DEFAULT_SC2BA_PLAN = Path(__file__).resolve().parent.parent / "bch" / "configs" / "v10_all13.json"
DEFAULT_LEGACY_MAP_ROOT = Path(r"D:\pycharm\PycharmProjects\HRL-IMCBS-test\experiment_map")

LEGACY_MM_SCENARIOS = {
    "MM4": {"map_name": "local_enemy_test_1", "filename": "MM4.SC2Map", "n_units": 4, "limit": 312},
    "MM4m": {"map_name": "local_enemy_test_1_mirror", "filename": "MM4m.SC2Map", "n_units": 4, "limit": 312},
    "MM4dist": {"map_name": "MarineMicro_MvsM_4_dist", "filename": "MM4dist.SC2Map", "n_units": 4, "limit": 312},
    "MM4distm": {"map_name": "MarineMicro_MvsM_4_dist_mirror", "filename": "MM4distm.SC2Map", "n_units": 4, "limit": 312},
    "MM8": {"map_name": "MarineMicro_MvsM_8_far", "filename": "MM8.SC2Map", "n_units": 8, "limit": 312},
    "MM8m": {"map_name": "MarineMicro_MvsM_8_far_mirror", "filename": "MM8m.SC2Map", "n_units": 8, "limit": 312},
}


@dataclass
class HRLv2Config:
    # Scenario & protocol
    scenario: str = "3m"
    regime: str = "pool-all9"  # pool-all9 | fixed-bot | pool-algos8 | fixed-<opp>
    seed: int = 101
    step_mul: int = 8
    device: str = "auto"

    # Multi-channel tactical Influence Map
    im_size: int = 64
    im_channels: int = 4  # 4 = full tactical IM; 1 = signed position-only ablation

    # Upper online adaptive state abstraction (IM-CNN + EMA-VQ + Bisim Split/Merge)
    proto_dim: int = 32
    init_prototypes: int = 8
    min_prototypes: int = 4
    max_prototypes: int = 48
    ema_decay: float = 0.98
    commitment_beta: float = 0.25
    bisim_coef: float = 0.20
    reward_pred_coef: float = 0.15
    dynamic_codebook: bool = True
    split_merge_every: int = 4
    split_cooldown: int = 2
    min_proto_count: float = 16.0
    dead_proto_count: float = 1.5
    dead_proto_age: int = 6
    split_score_threshold: float = 0.35
    merge_dist_threshold: float = 0.22
    merge_value_threshold: float = 0.35
    proto_buffer_size: int = 64

    # Mid-level tactical spatial-role clustering
    macro_interval: int = 4  # tau: upper decision horizon in env steps
    max_clusters: int = 3    # granularity choices M_t in {1..max_clusters}
    cluster_role_weight: float = 0.45
    cluster_hp_weight: float = 0.20
    cluster_kmeans_iters: int = 8

    # Extensible Upper -> Lower guidance interface
    guidance_type: str = "latent_skill"  # latent_skill | continuous_latent
    num_skills: int = 6
    guidance_dim: int = 16
    use_guidance: bool = True  # False = flat MAPPO ablation
    skill_entropy_coef: float = 0.02
    skill_diversity_coef: float = 0.05

    # Lower cluster-conditioned MAPPO
    lower_hidden_dim: int = 64
    critic_hidden_dim: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    ppo_epochs: int = 4
    rollout_steps: int = 1024
    minibatch_size: int = 256
    lr_actor: float = 5e-4
    lr_critic: float = 1e-3
    lr_upper: float = 3e-4
    entropy_coef: float = 0.015
    value_coef: float = 0.50
    max_grad_norm: float = 1.0

    # Training & evaluation budgets (aligned with v11 by default)
    train_steps: int = 40000
    eval_every: int = 8000
    eval_init: bool = True
    val_repeats: int = 2
    test_repeats: int = 5
    legacy_eval_episodes: int = 30
    legacy_final_episodes: int = 120
    legacy_episode_limit: Optional[int] = None

    # Optional curriculum warm-start & checkpoint continuation
    curriculum_warmup_frac: float = 0.0
    curriculum_bot_start_prob: float = 0.85
    resume_from: Optional[str] = None

    # Paths (read-only references to existing configs/maps; outputs always isolated)
    plan_path: str = str(DEFAULT_SC2BA_PLAN)
    legacy_map_root: str = str(DEFAULT_LEGACY_MAP_ROOT)
    legacy_fast_reset: bool = True

    def is_legacy_mm(self) -> bool:
        return self.scenario in LEGACY_MM_SCENARIOS

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "HRLv2Config":
        valid = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
