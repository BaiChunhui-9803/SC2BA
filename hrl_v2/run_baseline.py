"""CLI entrypoint for matched-protocol PyMARL (QMIX) and on-policy (MAPPO) baselines.

Isolated 100% from ``bch/`` and running v11 jobs:
  - Uses dedicated TEMP/TMP subdirectories (``sc2ba_hrl_v2_base_*``)
  - Writes outputs under ``D:\\Binich\\project\\SC2BA-Lab\\hrl_v2\\runs\\baselines\\`` (or --out-dir)
"""

import argparse
import os
from pathlib import Path
import tempfile

from bch.v10_suite import ALL_OPPONENTS, resolve_regime_opponents

from .baselines import BaselineConfig, OnPolicyMAPPOTrainer, PyMARLQMIXTrainer
from .config import LEGACY_MM_SCENARIOS
from .env_adapter import create_env

DEFAULT_BASELINES_ROOT = Path(__file__).resolve().parent / "runs" / "baselines"


def _isolate_temp_dir(tag: str) -> None:
    tmp_dir = Path(tempfile.gettempdir()) / f"sc2ba_hrl_v2_base_{tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)


def run_baseline(cfg: BaselineConfig, out_dir: Path) -> dict:
    tag = f"{cfg.algo}_{cfg.scenario}_{cfg.regime}_s{cfg.seed}".replace(":", "_")
    _isolate_temp_dir(tag)

    env, _meta = create_env(cfg.to_env_cfg())
    try:
        if cfg.is_legacy_mm():
            train_opps = ("bot",)
            eval_opps = ("bot",)
        else:
            train_opps = resolve_regime_opponents(cfg.regime)
            eval_opps = ALL_OPPONENTS

        if cfg.algo == "qmix":
            trainer = PyMARLQMIXTrainer(
                cfg, env, train_opponents=train_opps, eval_opponents=eval_opps, out_dir=out_dir
            )
        elif cfg.algo == "mappo":
            trainer = OnPolicyMAPPOTrainer(
                cfg, env, train_opponents=train_opps, eval_opponents=eval_opps, out_dir=out_dir
            )
        else:
            raise ValueError(f"Unsupported baseline algo '{cfg.algo}' (choose qmix or mappo).")

        if cfg.resume_from:
            trainer.load_checkpoint(Path(cfg.resume_from), load_optimizer=True, load_history=True)
            print(
                f"[RESUME {cfg.algo}] {cfg.scenario}/{cfg.regime} loaded {cfg.resume_from} "
                f"at step={trainer.total_steps} -> target={cfg.train_steps}",
                flush=True,
            )

        report = trainer.train()
    finally:
        env.close()

    fe = report["final_eval"]
    print(
        f"[DONE {report['algorithm']}] {cfg.scenario} | {cfg.regime} | seed={cfg.seed} | "
        f"steps={report['train_steps']} | all9={fe['all9_win_rate']:.3f} | "
        f"8algo={fe['algos8_win_rate']:.3f} | worst3={fe['worst3_algos_win_rate']:.3f} | "
        f"bot={fe['bot_win_rate']} | hp={fe['mean_hp_margin']:+.1f} | {report['wall_seconds']}s -> {out_dir}",
        flush=True,
    )
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Run matched-protocol PyMARL QMIX / on-policy MAPPO baselines.")
    p.add_argument("--algo", choices=["qmix", "mappo"], required=True)
    p.add_argument("--scenario", default="3m", help="SC2BA scenario (3m, 2s3z, ...) or Legacy map (MM4m, MM8)")
    p.add_argument("--regime", default="pool-all9", help="pool-all9 | fixed-bot | pool-algos8 | fixed-<opp>")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--eval-every", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--curriculum-warmup-frac", type=float, default=0.0)
    p.add_argument("--curriculum-bot-start-prob", type=float, default=0.85)
    p.add_argument("--qmix-epsilon-anneal-steps", type=int, default=50000)
    p.add_argument("--qmix-target-update-interval", type=int, default=200)
    p.add_argument("--mappo-rollout-steps", type=int, default=1020)
    p.add_argument("--val-repeats", type=int, default=2)
    p.add_argument("--test-repeats", type=int, default=5)
    p.add_argument("--legacy-eval-episodes", type=int, default=30)
    p.add_argument("--legacy-final-episodes", type=int, default=120)
    p.add_argument("--legacy-episode-limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    regime = "fixed-bot" if args.scenario in LEGACY_MM_SCENARIOS else args.regime
    cfg = BaselineConfig(
        algo=args.algo,
        scenario=args.scenario,
        regime=regime,
        seed=args.seed,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        device=args.device,
        resume_from=args.resume_from,
        curriculum_warmup_frac=args.curriculum_warmup_frac,
        curriculum_bot_start_prob=args.curriculum_bot_start_prob,
        qmix_epsilon_anneal_steps=args.qmix_epsilon_anneal_steps,
        qmix_target_update_interval=args.qmix_target_update_interval,
        mappo_rollout_steps=args.mappo_rollout_steps,
        val_repeats=args.val_repeats,
        test_repeats=args.test_repeats,
        legacy_eval_episodes=args.legacy_eval_episodes,
        legacy_final_episodes=args.legacy_final_episodes,
        legacy_episode_limit=args.legacy_episode_limit,
    )
    out_dir = args.out_dir or (
        DEFAULT_BASELINES_ROOT / f"{cfg.algo}_steps{cfg.train_steps}" / cfg.scenario / cfg.regime / f"seed_{cfg.seed}"
    )
    run_baseline(cfg, out_dir)


if __name__ == "__main__":
    main()
