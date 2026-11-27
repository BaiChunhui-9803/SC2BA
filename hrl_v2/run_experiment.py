"""CLI entrypoint for isolated DyIM-CHRL (hrl_v2) runs & the 2+2 MVP comparison suite.

Strictly isolated from ``bch/`` and ``bch/runs/``:
  - Uses dedicated TEMP/TMP subdirectories (``sc2ba_hrl_v2_*``)
  - Writes outputs solely under ``D:\\Binich\\project\\SC2BA-Lab\\hrl_v2\\runs\\`` (or --out-dir)
  - Supports SC2BA multi-opponent regimes (pool-all9 vs fixed-bot) on 3m, 2s3z, etc.
  - Supports legacy HRL-IMCBS maps (MM4m, MM8) under the identical atomic-action protocol.
"""

import argparse
import os
from pathlib import Path
import tempfile

from bch.v10_suite import ALL_OPPONENTS, resolve_regime_opponents

from .config import HRLv2Config, LEGACY_MM_SCENARIOS
from .env_adapter import create_env
from .trainer import DyIMCHRLTrainer

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent / "runs"

# Default 2+2 MVP comparison matrix requested by user:
#   SC2BA (pool-all9 vs fixed-bot): 3m, 2s3z
#   Legacy HRL-IMCBS (built-in):    MM4m, MM8
MVP_2PLUS2_JOBS = (
    ("3m", "pool-all9"),
    ("3m", "fixed-bot"),
    ("2s3z", "pool-all9"),
    ("2s3z", "fixed-bot"),
    ("MM4m", "fixed-bot"),
    ("MM8", "fixed-bot"),
)


def _isolate_temp_dir(tag: str) -> None:
    tmp_dir = Path(tempfile.gettempdir()) / f"sc2ba_hrl_v2_{tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    # gettempdir() above caches the original directory; SC2 uses mkdtemp().
    tempfile.tempdir = str(tmp_dir)


def run_single(cfg: HRLv2Config, out_dir: Path) -> dict:
    tag = f"{cfg.scenario}_{cfg.regime}_s{cfg.seed}".replace(":", "_")
    _isolate_temp_dir(tag)

    env, _meta = create_env(cfg)
    try:
        if cfg.is_legacy_mm():
            train_opps = ("bot",)
            eval_opps = ("bot",)
        else:
            train_opps = resolve_regime_opponents(cfg.regime)
            eval_opps = ALL_OPPONENTS

        trainer = DyIMCHRLTrainer(
            cfg,
            env,
            train_opponents=train_opps,
            eval_opponents=eval_opps,
            out_dir=out_dir,
        )
        if cfg.resume_from:
            trainer.load_checkpoint(Path(cfg.resume_from), load_optimizers=True, load_history=True)
            print(
                f"[RESUME hrl_v2] {cfg.scenario}/{cfg.regime} loaded {cfg.resume_from} "
                f"at step={trainer.total_steps} -> target={cfg.train_steps}",
                flush=True,
            )
        report = trainer.train()
    finally:
        env.close()

    fe = report["final_eval"]
    print(
        f"[DONE hrl_v2] {cfg.scenario} | {cfg.regime} | seed={cfg.seed} | "
        f"all9={fe['all9_win_rate']:.3f} | 8algo={fe['algos8_win_rate']:.3f} | "
        f"worst3={fe['worst3_algos_win_rate']:.3f} | bot={fe['bot_win_rate']} | "
        f"K_final={fe['active_prototypes']} | {report['wall_seconds']}s -> {out_dir}",
        flush=True,
    )
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Run isolated DyIM-CHRL (hrl_v2) experiments.")
    p.add_argument("--scenario", default="3m", help="SC2BA scenario (3m, 2s3z, ...) or Legacy map (MM4m, MM8)")
    p.add_argument("--regime", default="pool-all9", help="pool-all9 | fixed-bot | pool-algos8 | fixed-<opp>")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--eval-every", type=int, default=8000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--im-channels", type=int, choices=[1, 4], default=4)
    p.add_argument("--macro-interval", type=int, default=4)
    p.add_argument("--num-skills", type=int, default=6)
    p.add_argument("--guidance-type", choices=["latent_skill", "continuous_latent"], default="latent_skill")
    p.add_argument("--no-dynamic-codebook", action="store_true", help="Ablation: disable Split/Merge")
    p.add_argument("--no-guidance", action="store_true", help="Ablation: flat MAPPO without upper guidance")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--mvp-2plus2",
        action="store_true",
        help="Run the full 2+2 minimal verification suite sequentially (single-worker, zero conflict with v11).",
    )
    p.add_argument("--val-repeats", type=int, default=2)
    p.add_argument("--test-repeats", type=int, default=5)
    p.add_argument("--legacy-eval-episodes", type=int, default=30)
    p.add_argument("--legacy-final-episodes", type=int, default=120)
    p.add_argument("--legacy-episode-limit", type=int, default=None, help="Override legacy MM max step limit (default 312 = 2496 loops)")
    p.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint_final.pt/checkpoint_latest.pt to resume training")
    p.add_argument("--curriculum-warmup-frac", type=float, default=0.0, help="Bot-anchored smooth curriculum warmup fraction for pool-all9 (e.g. 0.25)")
    p.add_argument("--curriculum-bot-start-prob", type=float, default=0.85, help="Initial bot sampling probability during curriculum warmup")
    p.add_argument("--merge-dist-threshold", type=float, default=0.22, help="Dynamic codebook cosine-distance threshold for prototype merging")
    return p.parse_args()


def main():
    args = parse_args()
    base_kwargs = dict(
        seed=args.seed,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        rollout_steps=args.rollout_steps,
        im_channels=args.im_channels,
        macro_interval=args.macro_interval,
        num_skills=args.num_skills,
        guidance_type=args.guidance_type,
        dynamic_codebook=not args.no_dynamic_codebook,
        use_guidance=not args.no_guidance,
        device=args.device,
        val_repeats=args.val_repeats,
        test_repeats=args.test_repeats,
        legacy_eval_episodes=args.legacy_eval_episodes,
        legacy_final_episodes=args.legacy_final_episodes,
        legacy_episode_limit=args.legacy_episode_limit,
        resume_from=args.resume_from,
        curriculum_warmup_frac=args.curriculum_warmup_frac,
        curriculum_bot_start_prob=args.curriculum_bot_start_prob,
        merge_dist_threshold=args.merge_dist_threshold,
    )

    if args.mvp_2plus2:
        root = args.out_dir or (DEFAULT_RUNS_ROOT / f"mvp_2plus2_seed{args.seed}")
        for scenario, regime in MVP_2PLUS2_JOBS:
            job_dir = root / scenario / regime / f"seed_{args.seed}"
            if (job_dir / "done.json").exists():
                print(f"[SKIP completed] {job_dir}", flush=True)
                continue
            cfg = HRLv2Config(scenario=scenario, regime=regime, **base_kwargs)
            run_single(cfg, job_dir)
        return

    regime = "fixed-bot" if args.scenario in LEGACY_MM_SCENARIOS else args.regime
    cfg = HRLv2Config(scenario=args.scenario, regime=regime, **base_kwargs)
    out_dir = args.out_dir or (DEFAULT_RUNS_ROOT / cfg.scenario / cfg.regime / f"seed_{cfg.seed}")
    run_single(cfg, out_dir)


if __name__ == "__main__":
    main()
