"""v10 SC2BA multi-scenario, 4-architecture & fixed-vs-pool opponent experiment suite.

Subcommands:
    smoke      Verify all 13 scenarios x all 9 opponents x all 11 macros x 4 encoders, and measure throughput.
    collect    Collect probes across selected scenarios in parallel (resumable).
    fit        Fit frozen state encoders (im-medoid, unit-set, im-grid) per scenario (resumable).
    run        Run matched-step-budget Q training + periodic 9-opponent validation curves + final frozen eval.
    analyze    Aggregate all completed arms under --out-root into CSV/JSON summary tables.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .plan import Scenario, plan_digest, read_plan, scenario_seeds
from .records import json_line, write_json

DEFAULT_PLAN = Path(__file__).parent / "configs" / "v10_all13.json"
ALL_OPPONENTS = ("bot", "qmix", "vdn", "dop", "fop", "qplex", "coma", "iql", "qtran")
ALGO_OPPONENTS = ("qmix", "vdn", "dop", "fop", "qplex", "coma", "iql", "qtran")
CORE_METHODS = ("im-hash", "im-medoid", "unit-set", "im-grid")

SCENARIO_GROUPS = {
    "all13": (
        "3m", "8m", "25m", "2s3z", "3s5z", "1c3s5z", "MMM",
        "5m_vs_6m", "10m_vs_11m", "MMM2",
        "3s5z_vs_3s6z", "3z_vs_3s", "5z_vs_3s",
    ),
    "symmetric7": ("3m", "8m", "25m", "2s3z", "3s5z", "1c3s5z", "MMM"),
    "asymmetric3": ("5m_vs_6m", "10m_vs_11m", "MMM2"),
    "missing3": ("3s5z_vs_3s6z", "3z_vs_3s", "5z_vs_3s"),
    "core4": ("3m", "8m", "2s3z", "MMM"),
    "tier1": ("3m", "8m", "5m_vs_6m", "2s3z", "MMM", "3z_vs_3s"),
}


def resolve_scenarios(spec, available):
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token in SCENARIO_GROUPS:
            for item in SCENARIO_GROUPS[token]:
                if item not in out:
                    out.append(item)
        else:
            if token not in available:
                raise ValueError(f"Unknown scenario '{token}'. Available: {', '.join(available)}")
            if token not in out:
                out.append(token)
    return out


def resolve_regime_opponents(regime):
    if regime == "pool-all9":
        return ALL_OPPONENTS
    if regime == "pool-algos8":
        return ALGO_OPPONENTS
    if regime == "pool-top4":
        return ("qmix", "dop", "iql", "fop")
    if regime.startswith("fixed-"):
        opp = regime[len("fixed-"):]
        if opp not in ALL_OPPONENTS:
            raise ValueError(f"Unknown fixed opponent '{opp}' in regime '{regime}'.")
        return (opp,)
    if regime.startswith("pool:"):
        opps = tuple(x.strip() for x in regime[len("pool:"):].split("+") if x.strip())
        if not opps or not set(opps).issubset(ALL_OPPONENTS):
            raise ValueError(f"Invalid custom pool regime '{regime}'.")
        return opps
    raise ValueError(f"Unknown regime '{regime}'.")


def _worker_env(worker_slot, rollout_device="cpu"):
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parent.parent)
    existing_pp = env.get("PYTHONPATH", "")
    if repo_root not in existing_pp.split(os.pathsep):
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
    if rollout_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    # Isolate temp dir per worker to avoid Windows SC2 temp race conditions.
    tmp_dir = Path(tempfile.gettempdir()) / f"sc2ba_v10_w{worker_slot}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    return env


def _run_episode_timed(arena, policy, layout, opponent, episode_seed, train, *, step_budget=None, checkpoint=None):
    rng = np.random.default_rng(episode_seed)
    first = observation = arena.reset_round(layout, opponent)
    policy.begin_episode(train)
    visited = set()
    upper_unseen = 0
    lower_unseen = 0
    encode_times_ms = []
    orig_key = policy.encoder.key

    def timed_key(obs):
        t0 = perf_counter()
        result = orig_key(obs)
        encode_times_ms.append((perf_counter() - t0) * 1000.0)
        return result

    policy.encoder.key = timed_key
    steps, done = 0, False
    try:
        while not done and (step_budget is None or steps < step_budget):
            commands, (key, upper, lower) = policy.decide(observation, rng)
            if checkpoint is not None and steps:
                checkpoint(steps)
            visited.add(key)
            upper_unseen += int(upper)
            lower_unseen += int(lower)
            observation, done, info = arena.step_macros(commands)
            steps += 1
        if train:
            # A budget cutoff bootstraps from the last observation.
            policy.decide(observation, rng, terminal=done)
            if checkpoint is not None:
                checkpoint(steps)
    finally:
        policy.encoder.key = orig_key

    audit = arena.episode_details(first, observation, info)
    if not done:
        audit["outcome"] = "budget_cutoff"
    return {
        **audit,
        "visited_upper_keys": len(visited),
        "upper_unseen_steps": upper_unseen,
        "lower_unseen_steps": lower_unseen,
        "encode_ms_mean": float(np.mean(encode_times_ms)) if encode_times_ms else 0.0,
        "encode_ms_p50": float(np.median(encode_times_ms)) if encode_times_ms else 0.0,
        "encode_ms_p95": float(np.percentile(encode_times_ms, 95)) if encode_times_ms else 0.0,
    }


def _evaluate_frozen(arena, policy, layouts, opponents, repeats, eval_seed, step_tag):
    """Run frozen evaluation without mutating Q-tables; return summary + per-episode rows."""
    q_before = policy.snapshot()
    rng = np.random.default_rng(eval_seed)
    rows = []
    by_opp = defaultdict(Counter)
    by_layout_opp = defaultdict(Counter)
    unseen_upper = 0
    unseen_lower = 0
    total_steps = 0
    encode_p50s = []

    for layout in layouts:
        for opponent in opponents:
            for rep in range(repeats):
                ep_seed = int(rng.integers(2**32))
                audit = _run_episode_timed(arena, policy, layout, opponent, ep_seed, train=False)
                by_opp[opponent][audit["outcome"]] += 1
                by_layout_opp[(layout.id, opponent)][audit["outcome"]] += 1
                unseen_upper += audit["upper_unseen_steps"]
                unseen_lower += audit["lower_unseen_steps"]
                total_steps += max(1, audit["steps"])
                encode_p50s.append(audit["encode_ms_p50"])
                rows.append({
                    "checkpoint_steps": step_tag,
                    "layout": layout.id,
                    "split": layout.split,
                    "opponent": opponent,
                    "repeat": rep,
                    "episode_seed": ep_seed,
                    "outcome": audit["outcome"],
                    "steps": audit["steps"],
                    "final_hp": audit["final_hp"],
                    "visited_upper_keys": audit["visited_upper_keys"],
                    "upper_unseen_steps": audit["upper_unseen_steps"],
                    "lower_unseen_steps": audit["lower_unseen_steps"],
                    "encode_ms_p50": audit["encode_ms_p50"],
                })

    if q_before != policy.snapshot():
        raise RuntimeError("Frozen evaluation mutated Q-table!")

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
    algos8_wr = float(np.mean([per_opponent[o]["win_rate"] for o in algos_present])) if algos_present else 0.0
    worst3_wr = float(np.mean(sorted([per_opponent[o]["win_rate"] for o in algos_present])[:3])) if len(algos_present) >= 3 else algos8_wr
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

    summary = {
        "checkpoint_steps": step_tag,
        "episodes": len(rows),
        "all9_win_rate": all9_wr,
        "algos8_win_rate": algos8_wr,
        "worst3_algos_win_rate": worst3_wr,
        "bot_win_rate": bot_wr,
        "by_layout_win_rate": by_layout,
        "upper_q_rows": len(policy.upper.rows),
        "lower_q_rows": sum(len(t.rows) for t in policy.lower.values()),
        "upper_unseen_rate": unseen_upper / max(1, total_steps),
        "lower_unseen_rate": unseen_lower / max(1, total_steps),
        "encode_ms_p50": float(np.median(encode_p50s)) if encode_p50s else 0.0,
        "per_opponent": per_opponent,
    }
    return summary, rows


# ---------------------------------------------------------------------------
# Subcommand 1: SMOKE TEST across all 13 scenarios x 9 opponents x 11 macros
# ---------------------------------------------------------------------------

def _smoke_one_scenario(plan_path, scenario_name, seed, step_mul, out_dir):
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    from .arena import Arena
    from .lower_action.grouping import GROUPINGS, group_units
    from .lower_action.macros import MACROS, macro_commands
    from .upper_state.influence import HistogramHash, grid_input, influence_window
    from .upper_state.models import make_encoder, model_inputs

    t_start = perf_counter()
    with threadpool_limits(limits=1):
        document, available, _, _ = read_plan(Path(plan_path))
        scenario = available[scenario_name]
        engine_seed, _, rng = scenario_seeds(seed, document, scenario_name)
        std_layout = next(l for l in scenario.layouts if l.id == "standard")

        opp_results = {}
        sample_obs = []
        total_steps = 0
        macro_idx = 0
        group_idx = 0

        with Arena(scenario, engine_seed, step_mul=step_mul) as arena:
            for opponent in scenario.opponents:
                t_ep = perf_counter()
                first = obs = arena.reset_round(std_layout, opponent)
                sample_obs.append(obs)
                done = False
                steps = 0
                while not done:
                    grouping = GROUPINGS[group_idx % len(GROUPINGS)]
                    macro = MACROS[macro_idx % len(MACROS)]
                    group_idx += 1
                    macro_idx += 1
                    formation = group_units(obs.friendly, grouping, rng)
                    commands = macro_commands(obs, formation, macro)
                    obs, done, info = arena.step_macros(commands)
                    steps += 1
                    if len(sample_obs) < 32 and steps % 3 == 0:
                        sample_obs.append(obs)
                audit = arena.episode_details(first, obs, info)
                dt_ep = perf_counter() - t_ep
                total_steps += audit["steps"]
                opp_results[opponent] = {
                    "outcome": audit["outcome"],
                    "steps": audit["steps"],
                    "seconds": round(dt_ep, 3),
                    "final_hp": [round(x, 1) for x in audit["final_hp"]],
                    "restarts": audit["engine_restarts"],
                }

        # Verify all 4 encoder forward passes + measure latency on collected observations
        enc_bench = {}
        hh = HistogramHash()
        t_h = []
        for o in sample_obs[:16]:
            t0 = perf_counter()
            _ = hh(influence_window(o), o.influence_scale)
            t_h.append((perf_counter() - t0) * 1000.0)
        enc_bench["im-hash_ms_p50"] = round(float(np.median(t_h)), 3)

        for kind in ("unit-set", "im-grid"):
            model = make_encoder(kind, 32).eval()
            t_n = []
            with torch.inference_mode():
                for o in sample_obs[:16]:
                    t0 = perf_counter()
                    _ = model(*model_inputs(kind, [o]))
                    t_n.append((perf_counter() - t0) * 1000.0)
            enc_bench[f"{kind}_ms_p50"] = round(float(np.median(t_n)), 3)

        total_sec = perf_counter() - t_start
        result = {
            "scenario": scenario_name,
            "status": "PASS",
            "n_opponents_verified": len(opp_results),
            "total_episodes": len(opp_results),
            "total_steps": total_steps,
            "mean_steps_per_episode": round(total_steps / max(1, len(opp_results)), 2),
            "wall_seconds": round(total_sec, 2),
            "steps_per_second": round(total_steps / max(1e-6, total_sec), 2),
            "episodes_per_minute": round(len(opp_results) / max(1e-6, total_sec) * 60.0, 2),
            "encoder_latency": enc_bench,
            "opponents": opp_results,
        }
        out_file = Path(out_dir) / f"smoke_{scenario_name}.json"
        write_json(out_file, result)
        return result


def cmd_smoke(args):
    document, available, _, _ = read_plan(args.plan)
    scenarios = resolve_scenarios(args.scenarios, available)
    args.out_root.mkdir(parents=True, exist_ok=True)
    smoke_dir = args.out_root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v10 smoke] Running {len(scenarios)} scenarios across {args.workers} workers...", flush=True)
    t0 = perf_counter()
    results = {}
    failures = {}

    if args.workers == 1:
        for sc in scenarios:
            try:
                res = _smoke_one_scenario(str(args.plan), sc, args.seed, args.step_mul, str(smoke_dir))
                results[sc] = res
                print(
                    f"  PASS {sc:14s}: 9/9 opps, {res['total_steps']:4d} steps, "
                    f"{res['wall_seconds']:5.1f}s ({res['steps_per_second']:5.1f} st/s), "
                    f"hash={res['encoder_latency']['im-hash_ms_p50']:.2f}ms, "
                    f"grid={res['encoder_latency']['im-grid_ms_p50']:.2f}ms, "
                    f"uset={res['encoder_latency']['unit-set_ms_p50']:.2f}ms",
                    flush=True,
                )
            except Exception as exc:
                failures[sc] = repr(exc)
                print(f"  FAIL {sc:14s}: {exc!r}", flush=True)
    else:
        # Spawn isolated subprocesses so each SC2 worker gets its own TEMP dir and CPU-only env
        def run_sub(idx_sc):
            idx, sc = idx_sc
            env = _worker_env(idx % args.workers, rollout_device="cpu")
            cmd = [
                sys.executable, "-m", "bch.v10_suite", "smoke-worker",
                "--plan", str(args.plan),
                "--scenario", sc,
                "--seed", str(args.seed),
                "--step-mul", str(args.step_mul),
                "--out-dir", str(smoke_dir),
            ]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"rc={proc.returncode}\nSTDOUT:\n{proc.stdout[-800:]}\nSTDERR:\n{proc.stderr[-1200:]}")
            return json.loads((smoke_dir / f"smoke_{sc}.json").read_text(encoding="utf-8"))

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fut_map = {pool.submit(run_sub, (i, sc)): sc for i, sc in enumerate(scenarios)}
            for fut in as_completed(fut_map):
                sc = fut_map[fut]
                try:
                    res = fut.result()
                    results[sc] = res
                    print(
                        f"  PASS {sc:14s}: 9/9 opps, {res['total_steps']:4d} steps, "
                        f"{res['wall_seconds']:5.1f}s ({res['steps_per_second']:5.1f} st/s), "
                        f"hash={res['encoder_latency']['im-hash_ms_p50']:.2f}ms, "
                        f"grid={res['encoder_latency']['im-grid_ms_p50']:.2f}ms, "
                        f"uset={res['encoder_latency']['unit-set_ms_p50']:.2f}ms",
                        flush=True,
                    )
                except Exception as exc:
                    failures[sc] = str(exc)
                    print(f"  FAIL {sc:14s}: {exc}", flush=True)

    summary = {
        "protocol": "v10-smoke",
        "plan": str(args.plan),
        "passed": len(results),
        "failed": len(failures),
        "total_wall_seconds": round(perf_counter() - t0, 2),
        "results": [results[sc] for sc in scenarios if sc in results],
        "failures": failures,
    }
    write_json(smoke_dir / "smoke_summary.json", summary)
    print(f"[v10 smoke] Done: {len(results)}/{len(scenarios)} PASS in {summary['total_wall_seconds']}s.", flush=True)
    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand 2: COLLECT PROBES across scenarios in parallel
# ---------------------------------------------------------------------------

def cmd_collect(args):
    _, available, _, _ = read_plan(args.plan)
    scenarios = resolve_scenarios(args.scenarios, available)
    probes_root = args.out_root / "probes"
    probes_root.mkdir(parents=True, exist_ok=True)

    todo = []
    for sc in scenarios:
        sc_dir = probes_root / sc
        if (sc_dir / "manifest.json").exists() and not args.force:
            print(f"[collect] SKIP {sc} (already exists)", flush=True)
            continue
        if sc_dir.exists() and args.force:
            shutil.rmtree(sc_dir)
        elif sc_dir.exists():
            shutil.rmtree(sc_dir)
        todo.append(sc)

    if not todo:
        print("[collect] All requested probe archives already complete.", flush=True)
        return

    print(f"[collect] Collecting probes for {len(todo)} scenarios with {args.workers} workers...", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def run_one(idx_sc):
        idx, sc = idx_sc
        sc_dir = probes_root / sc
        env = _worker_env(idx % args.workers, rollout_device="cpu")
        cmd = [
            sys.executable, "-m", "bch", "collect",
            "--plan", str(args.plan),
            "--scenario", sc,
            "--seed", str(args.seed),
            "--repeats", str(args.repeats),
            "--horizon", str(args.horizon),
            "--step-mul", str(args.step_mul),
            "--threads", "1",
            "--output", str(sc_dir),
        ]
        for b in (args.behavior or ()):
            cmd.extend(["--behavior", b])
        t0 = perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"collect {sc} failed (rc={proc.returncode}): {proc.stderr[-1000:]}")
        manifest = json.loads((sc_dir / "manifest.json").read_text(encoding="utf-8"))
        return sc, manifest["samples"], dt

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, (i, sc)) for i, sc in enumerate(todo)]
        for fut in as_completed(futs):
            try:
                sc, n_samples, dt = fut.result()
                print(f"  [collect] DONE {sc:14s}: {n_samples:6d} probes in {dt:6.1f}s", flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(f"  [collect] FAIL: {exc}", flush=True)
    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand 3: FIT FROZEN ENCODERS (im-medoid, unit-set, im-grid) per scenario
# ---------------------------------------------------------------------------

def cmd_fit(args):
    _, available, _, _ = read_plan(args.plan)
    scenarios = resolve_scenarios(args.scenarios, available)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() and m.strip() != "im-hash"]
    probes_root = args.out_root / "probes"
    enc_root = args.out_root / "encoders"
    enc_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for sc in scenarios:
        probe_dir = probes_root / sc
        if not (probe_dir / "manifest.json").exists():
            raise FileNotFoundError(f"Missing probes for {sc}: {probe_dir}")
        for method in methods:
            out_dir = enc_root / sc / method
            if (out_dir / "encoder.pt").exists() and not args.force:
                print(f"[fit] SKIP {sc}/{method} (already exists)", flush=True)
                continue
            if out_dir.exists():
                shutil.rmtree(out_dir)
            tasks.append((sc, method, probe_dir, out_dir))

    if not tasks:
        print("[fit] All requested encoders already fitted.", flush=True)
        _write_fit_summary(enc_root, scenarios, methods)
        return

    # Run neural fits sequentially on GPU (or parallel on CPU), im-medoid in parallel on CPU
    neural_tasks = [t for t in tasks if t[1] != "im-medoid"]
    medoid_tasks = [t for t in tasks if t[1] == "im-medoid"]

    def run_fit_cmd(sc, method, probe_dir, out_dir, device, threads, slot):
        env = _worker_env(slot, rollout_device=device)
        cmd = [
            sys.executable, "-m", "bch", "fit",
            "--probes", str(probe_dir),
            "--scenario", sc,
            "--method", method,
            "--epochs", str(args.epochs),
            "--codes", str(args.codes),
            "--latent-dim", str(args.latent_dim),
            "--batch-size", str(args.batch_size),
            "--codebook-samples", str(args.codebook_samples),
            "--device", device,
            "--threads", str(threads),
            "--seed", str(args.seed),
            "--output", str(out_dir),
        ]
        t0 = perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"fit {sc}/{method} failed: {proc.stderr[-1200:]}")
        m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        return sc, method, dt, m

    failures = []
    if medoid_tasks:
        print(f"[fit] Fitting {len(medoid_tasks)} im-medoid encoders with {args.workers} CPU workers...", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [
                pool.submit(run_fit_cmd, sc, method, pdir, odir, "cpu", 1, i % args.workers)
                for i, (sc, method, pdir, odir) in enumerate(medoid_tasks)
            ]
            for fut in as_completed(futs):
                try:
                    sc, method, dt, m = fut.result()
                    print(f"  [fit] DONE {sc:14s}/{method:10s}: cost={m['medoid_cost']:.4f} ({dt:.1f}s)", flush=True)
                except Exception as exc:
                    failures.append(str(exc))
                    print(f"  [fit] FAIL: {exc}", flush=True)

    if neural_tasks:
        n_workers = 1 if args.device.startswith("cuda") else args.workers
        print(f"[fit] Fitting {len(neural_tasks)} neural encoders on {args.device} (workers={n_workers})...", flush=True)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [
                pool.submit(run_fit_cmd, sc, method, pdir, odir, args.device, args.threads, i % max(1, n_workers))
                for i, (sc, method, pdir, odir) in enumerate(neural_tasks)
            ]
            for fut in as_completed(futs):
                try:
                    sc, method, dt, m = fut.result()
                    print(
                        f"  [fit] DONE {sc:14s}/{method:10s}: best_ep={m['best_epoch']:3d}, "
                        f"val_loss={m['validation_loss']:.4f}, geo_mae={m['test_geometry_mae']:.4f}, "
                        f"hp_mae={m['test_hp_mae']:.2f} ({dt:.1f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(str(exc))
                    print(f"  [fit] FAIL: {exc}", flush=True)

    _write_fit_summary(enc_root, scenarios, methods)
    if failures:
        sys.exit(1)


def _write_fit_summary(enc_root, scenarios, methods):
    rows = []
    for sc in scenarios:
        for method in methods:
            mfile = enc_root / sc / method / "metrics.json"
            if not mfile.exists():
                continue
            m = json.loads(mfile.read_text(encoding="utf-8"))
            rows.append({
                "scenario": sc,
                "method": method,
                "codes": m.get("codes"),
                "train_samples": m.get("sample_counts", {}).get("train"),
                "val_samples": m.get("sample_counts", {}).get("validation"),
                "test_samples": m.get("sample_counts", {}).get("test"),
                "codebook_samples": m.get("codebook_samples"),
                "medoid_cost": round(m.get("medoid_cost", 0.0), 5),
                "best_epoch": m.get("best_epoch", ""),
                "validation_loss": round(m["validation_loss"], 5) if "validation_loss" in m else "",
                "test_geometry_mae": round(m["test_geometry_mae"], 5) if "test_geometry_mae" in m else "",
                "test_hp_mae": round(m["test_hp_mae"], 3) if "test_hp_mae" in m else "",
                "test_mae_readout": m.get("test_mae_readout", ""),
                "unique_codebook_inputs": m.get("unique_codebook_inputs", ""),
                "fit_seconds": round(m.get("seconds", 0.0), 2),
            })
    if rows:
        csv_path = enc_root / "fit_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Subcommand 4: RUN ARM WORKER (matched step-budget training + curves + eval)
# ---------------------------------------------------------------------------

def _run_arm_worker(args):
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    from .arena import Arena
    from .learner import HierarchicalPolicy
    from .upper_state.encoder import UpperEncoder

    t_start = perf_counter()
    with threadpool_limits(limits=1):
        document, available, _, _ = read_plan(args.plan)
        scenario = available[args.scenario]
        train_opponents = resolve_regime_opponents(args.regime)
        eval_opponents = scenario.opponents

        train_layouts = tuple(l for l in scenario.layouts if l.split == "train")
        val_layouts = tuple(l for l in scenario.layouts if l.split == "validation")
        test_layout_names = [x.strip() for x in args.test_layouts.split(",") if x.strip()]
        test_layouts = tuple(l for l in scenario.layouts if l.id in test_layout_names)
        if not test_layouts:
            test_layouts = tuple(l for l in scenario.layouts if l.split == "test")

        encoder_path = None if args.method == "im-hash" else args.encoder
        encoder = UpperEncoder(args.method, encoder_path)
        policy = HierarchicalPolicy(encoder)
        if encoder_path is not None:
            for code in range(len(encoder.medoids)):
                policy.upper.row(str(code))

        plan_sha = plan_digest(document)
        if encoder.fingerprint and (encoder.fit_manifest["plan_sha256"], encoder.fit_manifest["step_mul"]) != (
            plan_sha, args.step_mul
        ):
            raise ValueError("Encoder checkpoint does not match experiment plan or step_mul.")

        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)

        engine_seed, order_rng, play_rng = scenario_seeds(args.seed, document, scenario.name)
        val_seed_base = int(np.random.SeedSequence([args.seed, 7777]).generate_state(1)[0])
        test_seed = int(np.random.SeedSequence([args.seed, 9999]).generate_state(1)[0])

        # Build checkpoint schedule (always include 0 if eval_init, intermediate multiples of eval_every, and final train_steps)
        checkpoints = []
        if args.eval_init:
            checkpoints.append(0)
        if args.eval_every > 0:
            s = args.eval_every
            while s < args.train_steps:
                checkpoints.append(s)
                s += args.eval_every
        checkpoints.append(args.train_steps)
        checkpoints = sorted(set(checkpoints))
        next_cp_idx = 0

        train_pool = [(layout, opp) for layout in train_layouts for opp in train_opponents]
        pool_order = []
        train_steps = 0
        train_episodes = 0
        train_counts = defaultdict(Counter)
        train_encode_p50s = []
        curve_summaries = []

        snapshots = []
        if checkpoints[0] == 0:
            snapshots.append((0, policy.snapshot(), 0, 0.0))
            next_cp_idx += 1

        def freeze_checkpoint(episode_steps):
            nonlocal next_cp_idx
            step = train_steps + episode_steps
            if next_cp_idx < len(checkpoints) and step == checkpoints[next_cp_idx]:
                snapshots.append((step, policy.snapshot(), policy.episodes, perf_counter() - t_start))
                next_cp_idx += 1

        with Arena(scenario, engine_seed, step_mul=args.step_mul) as arena:
            with (out_dir / "train_episodes.jsonl").open("w", encoding="utf-8") as train_log:
                while train_steps < args.train_steps:
                    if not pool_order:
                        pool_order = list(order_rng.permutation(len(train_pool)))
                    layout, opponent = train_pool[pool_order.pop()]
                    ep_seed = int(play_rng.integers(2**32))
                    audit = _run_episode_timed(
                        arena, policy, layout, opponent, ep_seed, train=True,
                        step_budget=args.train_steps - train_steps, checkpoint=freeze_checkpoint,
                    )
                    train_steps += audit["steps"]
                    train_episodes += 1
                    train_counts[opponent][audit["outcome"]] += 1
                    train_encode_p50s.append(audit["encode_ms_p50"])
                    train_log.write(json_line({
                        "episode": train_episodes,
                        "cum_steps": train_steps,
                        "layout": layout.id,
                        "opponent": opponent,
                        "episode_seed": ep_seed,
                        "outcome": audit["outcome"],
                        "steps": audit["steps"],
                        "final_hp": audit["final_hp"],
                        "visited_upper_keys": audit["visited_upper_keys"],
                        "upper_q_rows": len(policy.upper.rows),
                        "encode_ms_p50": audit["encode_ms_p50"],
                    }))

        if train_steps != args.train_steps or [entry[0] for entry in snapshots] != checkpoints:
            raise RuntimeError("Training did not reach the exact decision budget and checkpoint schedule")

        # Evaluation uses the frozen snapshots and a separate game session.
        frozen = HierarchicalPolicy(encoder)
        with Arena(scenario, engine_seed, step_mul=args.step_mul) as arena:
            with (out_dir / "val_episodes.jsonl").open("w", encoding="utf-8") as val_log:
                for index, (step, snapshot, episodes, elapsed) in enumerate(snapshots):
                    frozen.restore(snapshot)
                    val_sum, val_rows = _evaluate_frozen(
                        arena, frozen, val_layouts, eval_opponents, args.val_repeats, val_seed_base + index, step
                    )
                    val_sum["actual_train_steps"] = step
                    val_sum["train_episodes"] = episodes
                    val_sum["elapsed_seconds"] = round(elapsed, 2)
                    curve_summaries.append(val_sum)
                    for row in val_rows:
                        val_log.write(json_line(row))

            test_sum, test_rows = _evaluate_frozen(
                arena, frozen, test_layouts, eval_opponents, args.test_repeats, test_seed, train_steps
            )
            with (out_dir / "test_episodes.jsonl").open("w", encoding="utf-8") as test_log:
                for r in test_rows:
                    test_log.write(json_line(r))

        wall_seconds = perf_counter() - t_start
        # Compute in-distribution vs out-of-distribution win rates on test
        in_opps = [o for o in train_opponents if o in test_sum["per_opponent"]]
        out_opps = [o for o in eval_opponents if o not in train_opponents and o in test_sum["per_opponent"]]
        in_dist_wr = float(np.mean([test_sum["per_opponent"][o]["win_rate"] for o in in_opps])) if in_opps else None
        out_dist_wr = float(np.mean([test_sum["per_opponent"][o]["win_rate"] for o in out_opps])) if out_opps else None

        identity = {
            "format": "bch-sc2ba-v10-arm-v1",
            "scenario": args.scenario,
            "method": args.method,
            "regime": args.regime,
            "train_opponents": list(train_opponents),
            "eval_opponents": list(eval_opponents),
            "seed": args.seed,
            "step_mul": args.step_mul,
            "plan_sha256": plan_sha,
            "encoder_sha256": encoder.fingerprint,
        }
        write_json(out_dir / "policy.json", {**identity, "q": policy.snapshot()})
        write_json(out_dir / "learning_curve.json", {**identity, "checkpoints": curve_summaries})
        report = {
            **identity,
            "train_step_budget": args.train_steps,
            "actual_train_steps": train_steps,
            "checkpoint_protocol": "exact_budget_frozen_replay_v1",
            "train_episodes": train_episodes,
            "upper_q_rows": len(policy.upper.rows),
            "lower_q_rows": sum(len(t.rows) for t in policy.lower.values()),
            "train_encode_ms_p50": float(np.median(train_encode_p50s)) if train_encode_p50s else 0.0,
            "wall_seconds": round(wall_seconds, 2),
            "test_in_dist_win_rate": in_dist_wr,
            "test_out_dist_win_rate": out_dist_wr,
            "test": test_sum,
            "val_final": curve_summaries[-1] if curve_summaries else None,
        }
        write_json(out_dir / "report.json", report)
        write_json(out_dir / "done.json", {"status": "done", "wall_seconds": round(wall_seconds, 2)})


def cmd_run(args):
    _, available, _, _ = read_plan(args.plan)
    scenarios = resolve_scenarios(args.scenarios, available)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    enc_root = args.out_root / "encoders"
    runs_root = args.out_root / "arms"
    runs_root.mkdir(parents=True, exist_ok=True)

    arms = []
    for sc in scenarios:
        for regime in regimes:
            _ = resolve_regime_opponents(regime)  # validate early
            for method in methods:
                enc_path = None
                if method != "im-hash":
                    enc_path = enc_root / sc / method / "encoder.pt"
                    if not enc_path.exists():
                        raise FileNotFoundError(
                            f"Missing encoder for {sc}/{method}: {enc_path}. Run collect + fit first."
                        )
                for seed in seeds:
                    arm_dir = runs_root / sc / regime / method / f"seed_{seed}"
                    if (arm_dir / "done.json").exists() and not args.force:
                        continue
                    arms.append((sc, regime, method, seed, enc_path, arm_dir))

    total_planned = len(scenarios) * len(regimes) * len(methods) * len(seeds)
    print(
        f"[v10 run] {len(arms)} pending arms (of {total_planned} total) across {args.workers} workers | "
        f"train_steps={args.train_steps}, eval_every={args.eval_every}, "
        f"val_rep={args.val_repeats}, test_rep={args.test_repeats}",
        flush=True,
    )

    if not arms:
        aggregate_results(args.out_root)
        return

    from concurrent.futures import ThreadPoolExecutor
    from threading import get_ident

    def launch_arm(idx_arm):
        idx, (sc, regime, method, seed, enc_path, arm_dir) = idx_arm
        if arm_dir.exists():
            shutil.rmtree(arm_dir)
        env = _worker_env(get_ident(), rollout_device="cpu")
        cmd = [
            sys.executable, "-m", "bch.v10_suite", "arm-worker",
            "--plan", str(args.plan),
            "--scenario", sc,
            "--regime", regime,
            "--method", method,
            "--seed", str(seed),
            "--train-steps", str(args.train_steps),
            "--eval-every", str(args.eval_every),
            "--val-repeats", str(args.val_repeats),
            "--test-repeats", str(args.test_repeats),
            "--test-layouts", args.test_layouts,
            "--step-mul", str(args.step_mul),
            "--output", str(arm_dir),
        ]
        if args.eval_init:
            cmd.append("--eval-init")
        if enc_path is not None:
            cmd.extend(["--encoder", str(enc_path)])
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logs = args.out_root / "driver_logs"
            logs.mkdir(exist_ok=True)
            (logs / f"failed-{sc}-{regime}-{method}-{seed}.log").write_text(
                proc.stdout + "\n" + proc.stderr, encoding="utf-8"
            )
            raise RuntimeError(
                f"{sc}/{regime}/{method}/seed_{seed} failed (rc={proc.returncode}): {proc.stderr[-1200:]}"
            )
        return json.loads((arm_dir / "report.json").read_text(encoding="utf-8"))

    t0 = perf_counter()
    done_count = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(launch_arm, (i, arm)): arm for i, arm in enumerate(arms)}
        for fut in as_completed(futs):
            arm = futs[fut]
            try:
                rep = fut.result()
                done_count += 1
                t_test = rep["test"]
                print(
                    f"  [{done_count:3d}/{len(arms)}] {rep['scenario']:13s} | {rep['regime']:11s} | "
                    f"{rep['method']:9s} | s={rep['seed']} | "
                    f"all9={t_test['all9_win_rate']*100:5.1f}% | algos8={t_test['algos8_win_rate']*100:5.1f}% | "
                    f"bot={(t_test['bot_win_rate'] or 0)*100:5.1f}% | Q={rep['upper_q_rows']:4d} | "
                    f"unseen={t_test['upper_unseen_rate']*100:4.1f}% | {rep['wall_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append(str(exc))
                print(f"  FAIL {arm[0]}/{arm[1]}/{arm[2]}/seed_{arm[3]}: {exc}", flush=True)
                for pending in futs:
                    pending.cancel()
                break
            if done_count % 24 == 0:
                aggregate_results(args.out_root)

    aggregate_results(args.out_root)
    print(f"[v10 run] Finished {done_count}/{len(arms)} arms in {perf_counter() - t0:.1f}s.", flush=True)
    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand 5: ANALYZE / AGGREGATE
# ---------------------------------------------------------------------------

def aggregate_results(out_root):
    arms_root = Path(out_root) / "arms"
    if not arms_root.exists():
        print(f"[analyze] No arms directory at {arms_root}", flush=True)
        return
    analysis_dir = Path(out_root) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    arm_rows = []
    curve_rows = []
    opp_rows = []
    grouped = defaultdict(list)

    for report_file in sorted(arms_root.rglob("report.json")):
        if not (report_file.parent / "done.json").exists():
            continue
        rep = json.loads(report_file.read_text(encoding="utf-8"))
        t = rep["test"]
        row = {
            "scenario": rep["scenario"],
            "regime": rep["regime"],
            "method": rep["method"],
            "seed": rep["seed"],
            "train_steps": rep["actual_train_steps"],
            "train_episodes": rep["train_episodes"],
            "upper_q_rows": rep["upper_q_rows"],
            "lower_q_rows": rep["lower_q_rows"],
            "all9_win_rate": round(t["all9_win_rate"], 4),
            "algos8_win_rate": round(t["algos8_win_rate"], 4),
            "worst3_algos_win_rate": round(t["worst3_algos_win_rate"], 4),
            "bot_win_rate": round(t["bot_win_rate"], 4) if t["bot_win_rate"] is not None else "",
            "in_dist_win_rate": round(rep["test_in_dist_win_rate"], 4) if rep["test_in_dist_win_rate"] is not None else "",
            "out_dist_win_rate": round(rep["test_out_dist_win_rate"], 4) if rep["test_out_dist_win_rate"] is not None else "",
            "standard_win_rate": round(t["by_layout_win_rate"].get("standard", float("nan")), 4) if "standard" in t["by_layout_win_rate"] else "",
            "near_win_rate": round(t["by_layout_win_rate"].get("near", float("nan")), 4) if "near" in t["by_layout_win_rate"] else "",
            "upper_unseen_rate": round(t["upper_unseen_rate"], 4),
            "lower_unseen_rate": round(t["lower_unseen_rate"], 4),
            "encode_ms_p50": round(t["encode_ms_p50"], 3),
            "wall_seconds": rep["wall_seconds"],
        }
        arm_rows.append(row)
        grouped[(rep["scenario"], rep["regime"], rep["method"])].append(rep)

        for opp, stats in sorted(t["per_opponent"].items()):
            opp_rows.append({
                "scenario": rep["scenario"],
                "regime": rep["regime"],
                "method": rep["method"],
                "seed": rep["seed"],
                "opponent": opp,
                "in_train_pool": int(opp in rep["train_opponents"]),
                "episodes": stats["episodes"],
                "win": stats["win"],
                "loss": stats["loss"],
                "draw": stats["draw"],
                "timeout": stats["timeout"],
                "win_rate": round(stats["win_rate"], 4),
            })

        lc_file = report_file.parent / "learning_curve.json"
        if lc_file.exists():
            lc = json.loads(lc_file.read_text(encoding="utf-8"))
            for cp in lc["checkpoints"]:
                curve_rows.append({
                    "scenario": rep["scenario"],
                    "regime": rep["regime"],
                    "method": rep["method"],
                    "seed": rep["seed"],
                    "checkpoint_steps": cp["checkpoint_steps"],
                    "train_episodes": cp.get("train_episodes", 0),
                    "all9_win_rate": round(cp["all9_win_rate"], 4),
                    "algos8_win_rate": round(cp["algos8_win_rate"], 4),
                    "worst3_algos_win_rate": round(cp["worst3_algos_win_rate"], 4),
                    "bot_win_rate": round(cp["bot_win_rate"], 4) if cp["bot_win_rate"] is not None else "",
                    "upper_q_rows": cp["upper_q_rows"],
                    "upper_unseen_rate": round(cp["upper_unseen_rate"], 4),
                })

    def dump_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    dump_csv(analysis_dir / "arms_per_seed.csv", arm_rows)
    dump_csv(analysis_dir / "learning_curves.csv", curve_rows)
    dump_csv(analysis_dir / "opponent_matrix.csv", opp_rows)

    # Aggregated mean ± std across seeds per (scenario, regime, method)
    agg_rows = []
    for (sc, regime, method), reps in sorted(grouped.items()):
        def mstd(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return "", ""
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return round(mean, 4), round(std, 4)

        a9_m, a9_s = mstd([r["test"]["all9_win_rate"] for r in reps])
        a8_m, a8_s = mstd([r["test"]["algos8_win_rate"] for r in reps])
        w3_m, w3_s = mstd([r["test"]["worst3_algos_win_rate"] for r in reps])
        bt_m, bt_s = mstd([r["test"]["bot_win_rate"] for r in reps])
        in_m, in_s = mstd([r["test_in_dist_win_rate"] for r in reps])
        oo_m, oo_s = mstd([r["test_out_dist_win_rate"] for r in reps])
        q_m, q_s = mstd([r["upper_q_rows"] for r in reps])
        un_m, _ = mstd([r["test"]["upper_unseen_rate"] for r in reps])
        ms_m, _ = mstd([r["test"]["encode_ms_p50"] for r in reps])
        agg_rows.append({
            "scenario": sc,
            "regime": regime,
            "method": method,
            "n_seeds": len(reps),
            "all9_wr_mean": a9_m,
            "all9_wr_std": a9_s,
            "algos8_wr_mean": a8_m,
            "algos8_wr_std": a8_s,
            "worst3_wr_mean": w3_m,
            "worst3_wr_std": w3_s,
            "bot_wr_mean": bt_m,
            "bot_wr_std": bt_s,
            "in_dist_wr_mean": in_m,
            "out_dist_wr_mean": oo_m,
            "upper_q_mean": q_m,
            "upper_unseen_mean": un_m,
            "encode_ms_p50": ms_m,
        })
    dump_csv(analysis_dir / "summary_by_condition.csv", agg_rows)
    write_json(analysis_dir / "summary.json", {
        "n_completed_arms": len(arm_rows),
        "n_conditions": len(agg_rows),
        "conditions": agg_rows,
    })
    print(f"[analyze] Wrote {len(arm_rows)} arms / {len(agg_rows)} conditions to {analysis_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # smoke
    p_smoke = sub.add_parser("smoke", help="Smoke-verify all scenarios x 9 opponents x 4 encoders.")
    p_smoke.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p_smoke.add_argument("--scenarios", default="all13")
    p_smoke.add_argument("--seed", type=int, default=1001)
    p_smoke.add_argument("--step-mul", type=int, default=8)
    p_smoke.add_argument("--workers", type=int, default=4)
    p_smoke.add_argument("--out-root", type=Path, required=True)

    # internal smoke worker
    p_sw = sub.add_parser("smoke-worker", help=argparse.SUPPRESS)
    p_sw.add_argument("--plan", type=Path, required=True)
    p_sw.add_argument("--scenario", required=True)
    p_sw.add_argument("--seed", type=int, required=True)
    p_sw.add_argument("--step-mul", type=int, required=True)
    p_sw.add_argument("--out-dir", type=Path, required=True)

    # collect
    p_col = sub.add_parser("collect", help="Collect probe datasets across scenarios in parallel.")
    p_col.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p_col.add_argument("--scenarios", default="all13")
    p_col.add_argument("--seed", type=int, default=20261126)
    p_col.add_argument("--repeats", type=int, default=2)
    p_col.add_argument("--horizon", type=int, default=4)
    p_col.add_argument("--step-mul", type=int, default=8)
    p_col.add_argument("--behavior", action="append")
    p_col.add_argument("--workers", type=int, default=4)
    p_col.add_argument("--force", action="store_true")
    p_col.add_argument("--out-root", type=Path, required=True)

    # fit
    p_fit = sub.add_parser("fit", help="Fit frozen encoders per scenario.")
    p_fit.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p_fit.add_argument("--scenarios", default="all13")
    p_fit.add_argument("--methods", default="im-medoid,unit-set,im-grid")
    p_fit.add_argument("--epochs", type=int, default=160)
    p_fit.add_argument("--codes", type=int, default=32)
    p_fit.add_argument("--latent-dim", type=int, default=32)
    p_fit.add_argument("--batch-size", type=int, default=64)
    p_fit.add_argument("--codebook-samples", type=int, default=4096)
    p_fit.add_argument("--device", default="cuda")
    p_fit.add_argument("--threads", type=int, default=4)
    p_fit.add_argument("--workers", type=int, default=4)
    p_fit.add_argument("--seed", type=int, default=20261126)
    p_fit.add_argument("--force", action="store_true")
    p_fit.add_argument("--out-root", type=Path, required=True)

    # run
    p_run = sub.add_parser("run", help="Run matched-step training + curves + 9-opponent frozen evaluation.")
    p_run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p_run.add_argument("--scenarios", default="all13")
    p_run.add_argument("--methods", default="im-hash,im-medoid,unit-set,im-grid")
    p_run.add_argument("--regimes", default="pool-all9")
    p_run.add_argument("--seeds", default="101,202,303")
    p_run.add_argument("--train-steps", type=int, default=8000)
    p_run.add_argument("--eval-every", type=int, default=2000)
    p_run.add_argument("--eval-init", action="store_true")
    p_run.add_argument("--val-repeats", type=int, default=2)
    p_run.add_argument("--test-repeats", type=int, default=5)
    p_run.add_argument("--test-layouts", default="standard,near")
    p_run.add_argument("--step-mul", type=int, default=8)
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--out-root", type=Path, required=True)

    # internal arm worker
    p_aw = sub.add_parser("arm-worker", help=argparse.SUPPRESS)
    p_aw.add_argument("--plan", type=Path, required=True)
    p_aw.add_argument("--scenario", required=True)
    p_aw.add_argument("--regime", required=True)
    p_aw.add_argument("--method", required=True)
    p_aw.add_argument("--encoder", type=Path)
    p_aw.add_argument("--seed", type=int, required=True)
    p_aw.add_argument("--train-steps", type=int, required=True)
    p_aw.add_argument("--eval-every", type=int, required=True)
    p_aw.add_argument("--eval-init", action="store_true")
    p_aw.add_argument("--val-repeats", type=int, required=True)
    p_aw.add_argument("--test-repeats", type=int, required=True)
    p_aw.add_argument("--test-layouts", required=True)
    p_aw.add_argument("--step-mul", type=int, required=True)
    p_aw.add_argument("--output", type=Path, required=True)

    # analyze
    p_an = sub.add_parser("analyze", help="Aggregate completed arms into summary CSV/JSON.")
    p_an.add_argument("--out-root", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "smoke-worker":
        _smoke_one_scenario(str(args.plan), args.scenario, args.seed, args.step_mul, str(args.out_dir))
    elif args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "arm-worker":
        _run_arm_worker(args)
    elif args.cmd == "analyze":
        aggregate_results(args.out_root)


if __name__ == "__main__":
    main()
