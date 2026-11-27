"""Validate the SC2BA v10 handoff gates from actual run artifacts."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from .records import digest_file, write_json
from .v10_suite import ALL_OPPONENTS, CORE_METHODS, SCENARIO_GROUPS, resolve_regime_opponents


SCENARIOS = SCENARIO_GROUPS["all13"]
CORE = SCENARIO_GROUPS["core4"]
REGIMES = ("fixed-bot", "fixed-qmix", "fixed-dop", "fixed-vdn", "fixed-coma", "pool-algos8", "pool-all9")
SEEDS = (101, 202, 303)
STEPS = (0, 2000, 4000, 6000, 8000)
FOLDS = ("train", "validation", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def json_lines(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def smoke(root):
    summary = read_json(root / "smoke" / "smoke_summary.json")
    require(summary["passed"] == 13 and summary["failed"] == 0 and not summary["failures"],
            "Stage 0 must pass all 13 scenarios")
    rows = summary["results"]
    require(len(rows) == 13 and {row["scenario"] for row in rows} == set(SCENARIOS), "Missing smoke scenario")
    for row in rows:
        require(row["status"] == "PASS" and row["total_episodes"] == row["n_opponents_verified"] == 9,
                f"Incomplete smoke for {row['scenario']}")
        require(set(row["opponents"]) == set(ALL_OPPONENTS)
                and all(item["restarts"] == 0 for item in row["opponents"].values()),
                f"Opponent coverage or engine restarts in {row['scenario']}")
    steps = sum(row["total_steps"] for row in rows)
    return {"scenarios": 13, "episodes": 117, "steps": steps,
            "aggregate_steps_per_second": steps / summary["total_wall_seconds"], "engine_restarts": 0}


def fitted(root):
    import torch

    counts = {}
    manifests = {}
    for scenario in SCENARIOS:
        folder = root / "probes" / scenario
        manifest = manifests[scenario] = read_json(folder / "manifest.json")
        require(set(manifest["behaviors"]) == {"mixed", "legacy-nearest"}, f"Missing probe behavior in {scenario}")
        require(all(manifest["counts"]["split"].get(fold, 0) > 0 for fold in FOLDS), f"Empty probe fold for {scenario}")
        require(digest_file(folder / "probes.jsonl.gz") == manifest["probes_sha256"]
                and digest_file(folder / "episodes.jsonl") == manifest["episodes_sha256"],
                f"Probe fingerprint mismatch in {scenario}")
        episodes = json_lines(folder / "episodes.jsonl")
        expected = 4 * 9 * 2 * manifest["repeats"]
        require(manifest["repeats"] in (2, 4) and len(episodes) == expected, f"Wrong probe episode count in {scenario}")
        require(all(row["engine_restarts"] == 0 for row in episodes), f"Probe game restarted in {scenario}")
        counts[scenario] = manifest["counts"]["split"]

    encoders = root / "encoders"
    rows = csv_rows(encoders / "fit_summary.csv")
    expected = set(itertools.product(SCENARIOS, CORE_METHODS[1:]))
    require(len(rows) == 39 and {(row["scenario"], row["method"]) for row in rows} == expected,
            "Stage 1 requires 39 distinct fitted encoders")
    for row in rows:
        scenario, method = row["scenario"], row["method"]
        label = f"{scenario}/{method}"
        bundle = encoders / scenario / method
        metrics = read_json(bundle / "metrics.json")
        weight = torch.load(bundle / "encoder.pt", weights_only=True, map_location="cpu")
        require(weight["manifest"] == metrics and weight["kind"] == method and len(weight["medoids"]) == 32,
                f"Checkpoint or codebook mismatch for {label}")
        require(metrics["probes_sha256"] == manifests[scenario]["probes_sha256"]
                and metrics["plan_sha256"] == manifests[scenario]["plan_sha256"], f"Probe link mismatch for {label}")
        require(metrics["unique_codebook_inputs"] >= 32 and metrics["codes"] == 32 and metrics["latent_dim"] == 32,
                f"Insufficient training states for {label}")
        require(metrics["epochs"] == 160 and metrics["fit_device"] == ("cpu" if method == "im-medoid" else "cuda"),
                f"Wrong fit device or epoch budget for {label}")
        for metric in ("medoid_cost", "test_geometry_mae", "test_hp_mae"):
            value = float(row[metric])
            require(math.isfinite(value) and value > 0, f"Invalid {metric} for {label}: {value}")
    return {"scenarios": 13, "encoders": 39, "cuda_encoders": 26, "probe_sample_folds": counts}


def arm(root, scenario, regime, method, seed):
    directory = root / "arms" / scenario / regime / method / f"seed_{seed}"
    done = read_json(directory / "done.json")
    rep = read_json(directory / "report.json")
    require(done["status"] == "done", f"Unfinished arm {directory}")
    require((rep["scenario"], rep["regime"], rep["method"], rep["seed"]) == (scenario, regime, method, seed),
            f"Identity mismatch for {directory}")
    require(rep["actual_train_steps"] == rep["train_step_budget"] == 8000
            and rep.get("checkpoint_protocol") == "exact_budget_frozen_replay_v1", f"Wrong decision count in {directory}")
    require(set(rep["train_opponents"]) == set(resolve_regime_opponents(regime))
            and set(rep["eval_opponents"]) == set(ALL_OPPONENTS), f"Wrong opponents in {directory}")
    if method != "im-hash":
        require(rep["encoder_sha256"] == digest_file(root / "encoders" / scenario / method / "encoder.pt"),
                f"Frozen encoder mismatch in {directory}")
        require(rep["upper_q_rows"] <= 32 and rep["test"]["upper_unseen_rate"] == 0.0,
                f"Undefined code or excessive Q rows in {directory}")

    curve = read_json(directory / "learning_curve.json")["checkpoints"]
    require(tuple(check["checkpoint_steps"] for check in curve) == STEPS
            and all(check["episodes"] == 18 and check["actual_train_steps"] == step
                    for check, step in zip(curve, STEPS)), f"Incomplete validation schedule in {directory}")
    val = json_lines(directory / "val_episodes.jsonl")
    val_keys = {(row["checkpoint_steps"], row["layout"], row["opponent"], row["repeat"]) for row in val}
    require(len(val) == 90 and val_keys == set(itertools.product(STEPS, ("diagonal",), ALL_OPPONENTS, range(2))),
            f"Incomplete validation matchups in {directory}")
    test = json_lines(directory / "test_episodes.jsonl")
    test_keys = {(row["layout"], row["opponent"], row["repeat"]) for row in test}
    require(len(test) == 90 and test_keys == set(itertools.product(("standard", "near"), ALL_OPPONENTS, range(5))),
            f"Incomplete test matchups in {directory}")
    require(rep["test"]["episodes"] == 90 and all(row["checkpoint_steps"] == 8000 for row in test),
            f"Final test was not frozen at 8000 in {directory}")
    require(all(math.isfinite(rep["test"][key]) for key in ("all9_win_rate", "algos8_win_rate", "encode_ms_p50")),
            f"Nonfinite score in {directory}")
    return directory


def analyze_rows(root, selected_conditions):
    rows = csv_rows(root / "analysis" / "summary_by_condition.csv")
    by_condition = {(row["scenario"], row["regime"], row["method"]): row for row in rows}
    require(len(by_condition) == len(rows), "Duplicate analyze conditions")
    require(selected_conditions <= by_condition.keys() and
            all(int(by_condition[key]["n_seeds"]) == 3 for key in selected_conditions),
            "Incomplete three-seed analyze conditions")


def a1(root):
    keys = set(itertools.product(SCENARIOS, ("pool-all9",), CORE_METHODS))
    completed = {key: arm(root, *key) for key in itertools.product(SCENARIOS, ("pool-all9",), CORE_METHODS, SEEDS)}
    analyze_rows(root, keys)
    reused = {
        file.relative_to(root).as_posix(): digest_file(file)
        for key, directory in completed.items() if key[0] in CORE
        for file in (directory / "done.json", directory / "report.json", directory / "policy.json")
    }
    return {"arms": len(completed), "conditions": len(keys), "core4_pool_all9_hashes": reused}


def b1(root):
    before = read_json(root / "gates" / "stage2.json")["core4_pool_all9_hashes"]
    require(len(before) == 144, "Gate 2 did not fingerprint all 48 shared arms")
    for rel, digest in before.items():
        require(digest_file(root / rel) == digest, f"Stage 2 shared arm changed: {rel}")
    keys = set(itertools.product(CORE, REGIMES, CORE_METHODS))
    for key in itertools.product(CORE, REGIMES, CORE_METHODS, SEEDS):
        arm(root, *key)
    analyze_rows(root, keys)
    analysis = root / "analysis"

    def core_rows(filename):
        return [row for row in csv_rows(analysis / filename) if row["scenario"] in CORE]

    curves = core_rows("learning_curves.csv")
    curve_keys = {(row["scenario"], row["regime"], row["method"], int(row["seed"]), int(row["checkpoint_steps"]))
                  for row in curves}
    require(len(curves) == 1680 and curve_keys == set(itertools.product(CORE, REGIMES, CORE_METHODS, SEEDS, STEPS)),
            "Missing core4 learning curve rows")
    opponents = core_rows("opponent_matrix.csv")
    opponent_keys = {(row["scenario"], row["regime"], row["method"], int(row["seed"]), row["opponent"]) for row in opponents}
    require(len(opponents) == 3024 and opponent_keys == set(itertools.product(CORE, REGIMES, CORE_METHODS, SEEDS, ALL_OPPONENTS)),
            "Missing core4 opponent matrix rows")
    require(all(int(row["episodes"]) == 10 and int(row["in_train_pool"]) ==
                (row["opponent"] in resolve_regime_opponents(row["regime"])) for row in opponents),
            "Invalid opponent membership or evaluation counts")
    return {"arms": 336, "new_arms": 288, "reused_arms": 48, "curve_rows": 1680, "opponent_rows": 3024}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    result = (smoke, fitted, a1, b1)[args.stage](args.out_root)
    record = {"stage": args.stage, "status": "PASS", **result}
    path = args.out_root / "gates" / f"stage{args.stage}.json"
    path.parent.mkdir(exist_ok=True)
    write_json(path, record)
    print(json.dumps({key: value for key, value in record.items() if key != "core4_pool_all9_hashes"}, ensure_ascii=False))

