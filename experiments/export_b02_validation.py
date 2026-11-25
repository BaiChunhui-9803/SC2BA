"""Export B02 paired validation curves and same-budget comparisons."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


ARM_NAMES = ("qmix", "mixed")
DISPLAY = {"qmix": "QMIX only", "mixed": "Mixed opponents"}
BOT = "computer ai"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pair(root):
    configs = {arm: read_json(root / arm / "config.json") for arm in ARM_NAMES}
    same = ("seed", "steps_per_update", "warmup_steps", "validation_layouts", "test_layouts", "opponents",
            "batch_size", "replay_size", "target_update", "initial_lr", "final_lr", "lr_horizon")
    if any(configs["qmix"][key] != configs["mixed"][key] for key in same):
        raise ValueError("B02 arms did not share their evaluation and update protocol")
    if ([name for name, _ in configs["qmix"]["training_opponents"]] != ["qmix"]
            or configs["mixed"]["training_opponents"] != configs["mixed"]["opponents"]):
        raise ValueError("Invalid B02 training opponent sets")
    return configs["qmix"]


def read_validation(root, config):
    expected = {(name, tuple(spawn)) for name, _ in config["opponents"] for spawn in config["validation_layouts"]}
    lines = defaultdict(dict)
    with (root / "validation_episodes.jsonl").open(encoding="utf-8") as file:
        for text in file:
            row = json.loads(text)
            key = (row["opponent"], tuple(row["red_spawn"]))
            series = lines[row["milestone"]]
            if key not in expected or key in series:
                raise ValueError(f"Invalid validation episode: {row}")
            series[key] = row
    if any(set(block) != expected for block in lines.values()):
        raise ValueError(f"Missing validation decisions for {root}")
    return lines


def win(rows):
    return sum(row["outcome"] == "win" for row in rows) / len(rows)


def curve(blocks):
    return [
        {"milestone": milestone, "training_steps": next(iter(rows.values()))["training_steps"],
         "all_win_rate": win(list(rows.values())),
         "learned_win_rate": win([value for (name, _), value in rows.items() if name != BOT])}
        for milestone, rows in sorted(blocks.items())
    ]


def compare(at, config):
    rows = []
    for policy, _ in config["opponents"]:
        comparisons = {arm: [entry for (name, _), entry in at[arm].items() if name == policy]
                       for arm in ARM_NAMES}
        row = {"opponent": policy, "games_per_arm": len(comparisons["mixed"])}
        for arm, sample in comparisons.items():
            outcome = Counter(ep["outcome"] for ep in sample)
            row.update({f"{arm}_{key}": outcome[key] for key in ("win", "loss", "draw")})
            row[f"{arm}_win_rate"] = win(sample)
        row["mixed_minus_qmix_pp"] = 100 * (row["mixed_win_rate"] - row["qmix_win_rate"])
        rows.append(row)
    learned = [name for name, _ in config["opponents"] if name != BOT]
    points = [tuple(spawn) for spawn in config["validation_layouts"]]
    differences = np.array([
        np.mean([float(at["mixed"][name, point]["outcome"] == "win")
                 - float(at["qmix"][name, point]["outcome"] == "win") for name in learned])
        for point in points
    ])
    rng = np.random.default_rng(26112504)
    boot = differences[rng.integers(0, len(points), size=(10_000, len(points)))].mean(axis=1)
    return rows, {
        "learned_win_rate_difference_pp": float(100 * differences.mean()),
        "paired_layout_interval95_pp": (100 * np.quantile(boot, (.025, .975))).tolist(),
        "weakest_three_mean": {
            arm: float(np.mean(sorted(row[f"{arm}_win_rate"] for row in rows if row["opponent"] != BOT)[:3]))
            for arm in ARM_NAMES
        },
    }


def plot(curves, output):
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42,
                                "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), sharex=True, sharey=True, layout="constrained")
    for axis, metric, label in zip(axes, ("learned_win_rate", "all_win_rate"), ("8 learned opponents", "8 opponents + native Bot")):
        for arm, color in zip(ARM_NAMES, ("#bb5c25", "#17796a")):
            points = curves[arm]
            axis.plot([row["training_steps"] / 1e6 for row in points], [row[metric] * 100 for row in points],
                      label=DISPLAY[arm], color=color, linewidth=1.7, marker="o", markersize=2.7)
        axis.set_ylim(0, 102)
        axis.set_xlabel("Training steps (millions)")
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", weight="semibold")
        axis.grid(axis="y", color="#d8e3ee", lw=.7)
    axes[0].set_ylabel("Frozen validation win rate (%)")
    axes[1].legend(loc="lower right", frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"B02_validation_curves.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_directory", type=Path)
    args = parser.parse_args()
    root = args.pair_directory
    config = validate_pair(root)
    blocks = {arm: read_validation(root / arm, config) for arm in ARM_NAMES}
    common = sorted(set(blocks["qmix"]) & set(blocks["mixed"]))
    if not common:
        raise ValueError("No shared validation checkpoint")
    milestone = common[-1]
    curves = {arm: curve(blocks[arm]) for arm in ARM_NAMES}
    rows, comparison = compare({arm: blocks[arm][milestone] for arm in ARM_NAMES}, config)
    status = read_json(root / "pair-status.json")
    output = root / "comparison"
    output.mkdir(exist_ok=True)
    write_csv(output / "validation-curves.csv", [{"arm": arm, **row} for arm in ARM_NAMES for row in curves[arm]])
    write_csv(output / "validation-matchups.csv", rows)
    (output / "validation-comparison.json").write_text(json.dumps({
        "training_seeds": [config["seed"]], "milestone": milestone,
        "actual_steps": {arm: next(iter(blocks[arm][milestone].values()))["training_steps"] for arm in ARM_NAMES},
        "status": status, **comparison,
    }, indent=2) + "\n", encoding="utf-8")
    plot(curves, output)
    print(json.dumps({"milestone": milestone, **comparison}, ensure_ascii=False))
