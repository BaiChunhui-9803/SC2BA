"""Validate and package B01 metrics and figures for delivery."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_bot(source, main, bot):
    names = main["policies"]
    layouts = {int(i): tuple(point) for i, point in read_json(source / "protocol.json")["layouts"]}
    with (source / "bot-matches.csv").open(encoding="utf-8", newline="") as file:
        matches = list(csv.DictReader(file))
    keys = {(m["policy"], int(m["layout_id"])) for m in matches}
    if keys != {(policy, layout) for policy in names for layout in layouts} or len(matches) != len(keys):
        raise ValueError("The Bot addition does not cover each frozen layout and policy once")
    if any(m["agent_camp"] != "red" or (float(m["x"]), float(m["y"])) != layouts[int(m["layout_id"])]
           for m in matches):
        raise ValueError("Unexpected camp assignment or layout")
    counts = Counter((m["policy"], m["agent_result"]) for m in matches)
    for row in bot["rows"]:
        for outcome, field in (("win", "wins"), ("loss", "losses"), ("draw", "draws")):
            if counts[row["policy"], outcome] != row[field]:
                raise ValueError(f"Bot summary disagrees with raw outcomes: {row}")
    if len(matches) != bot["games"] or {r["policy"] for r in bot["rows"]} != set(names):
        raise ValueError("Bot summary has the wrong coverage")


def plot(main, bot, output):
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "pdf.fonttype": 42, "axes.spines.top": False,
                                "axes.spines.right": False})
    names = main["policies"]
    cells = {(r["policy"], r["opponent"]): r["win_rate"] * 100 for r in main["cells"]}
    array = np.array([[cells.get((a, b), np.nan) for b in names] for a in names])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 6.1),
                                     gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    heat = left.imshow(array, cmap="YlGnBu", vmin=0, vmax=100)
    for i, row in enumerate(array):
        for j, value in enumerate(row):
            if np.isnan(value):
                left.text(j, i, "—", ha="center", va="center", color="#718093")
                continue
            rgb = heat.cmap(heat.norm(value))[:3]
            ink = "#fff" if sum(value * weight for value, weight in zip(rgb, (0.21, 0.72, 0.07))) < .48 else "#172438"
            left.text(j, i, f"{value:.1f}", ha="center", va="center", color=ink, fontsize=9)
    ticks = [name.upper() for name in names]
    left.set_xticks(range(len(names)), ticks)
    left.set_yticks(range(len(names)), ticks)
    left.set_xlabel("Column policy")
    left.set_ylabel("Row policy")
    left.text(0, 1.06, "(a) Frozen policies, alternating camps", transform=left.transAxes, weight="semibold")
    fig.colorbar(heat, ax=left, shrink=.8, label="Row win rate (%)")

    rows = bot["rows"]
    values = np.array([100 * row["win_rate"] for row in rows])
    labels = [row["policy"].upper() for row in rows]
    bars = right.barh(labels, values, color="#19796e", height=.58)
    right.invert_yaxis()
    right.set_xlim(0, 110)
    right.set_xticks(range(0, 101, 20))
    right.set_xlabel("Frozen policy win rate (%)")
    right.set_axisbelow(True)
    right.grid(axis="x", color="#dbe4eb", lw=.7)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    right.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=5, color="#172438")
    right.text(0, 1.03, "(b) Frozen policy (red) vs native Bot (blue)", transform=right.transAxes, weight="semibold")
    fig.savefig(output / "B01_figures.pdf", bbox_inches="tight")
    fig.savefig(output / "B01_figures.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def summary(main, bot, output):
    top = main["ranking"][0]
    strongest_bot = bot["rows"][0]
    notes = [
        "# SC2BA B01 结果",
        "",
        "游戏版本为 " + main["game_version"] + "。场景为 3m，所有对手权重冻结，平局计入胜率分母。",
        "",
        f"- 8×8 主矩阵含 {main['pairwise_games']:,} 局两两对抗。每对在同一组 144 个出生点互换红蓝，共 288 局。",
        f"- 另有 {main['games'] - main['pairwise_games']:,} 局同策略自对抗，用于检查阵营偏差。",
        f"- 原生 Bot 补测共 {bot['games']:,} 局。策略固定红方，Bot 固定蓝方、难度 {bot['bot_difficulty']}，每个策略 144 局。",
        "",
        "## 结果",
        "",
        f"{top['policy'].upper()} 的两两对抗等权平均胜率最高，为 {top['mean_win_rate']:.2%}。",
        f"面对难度 7 内置 Bot，{strongest_bot['policy'].upper()} 的胜率最高，为 {strongest_bot['win_rate']:.2%}。",
        "Bot 补测的阵营协议与主矩阵不同。图表分别展示双 Agent 对抗和固定阵营的 Bot 参考结果。",
        "",
        "| 冻结策略 | 双 Agent 等权平均胜率 | 对内置 Bot 胜率 | 对 Bot 胜／负／平 |",
        "|---|---:|---:|---:|",
    ]
    by_policy = {row["policy"]: row for row in bot["rows"]}
    for entry in main["ranking"]:
        row = by_policy[entry["policy"]]
        notes.append(f"| {entry['policy'].upper()} | {entry['mean_win_rate']:.2%} | {row['win_rate']:.2%} | "
                     f"{row['wins']}／{row['losses']}／{row['draws']} |")
    (output / "B01_conclusions.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main = read_json(args.source / "report.json")
    bot = read_json(args.source / "bot-report.json")
    verify_bot(args.source, main, bot)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(main, bot, args.output)
    summary(main, bot, args.output)
    files = ("report.json", "protocol.json", "matches.csv", "pairwise.csv", "self-play.csv",
             "win-rate-matrix.csv", "bot-report.json", "bot-matches.csv", "bot-reference.csv")
    destination = args.output / "B01-SC2BA-results-20261125.zip"
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for filename in files:
            archive.write(args.source / filename, "data/" + filename)
        for filename in ("B01_figures.pdf", "B01_figures.png", "B01_conclusions.md"):
            archive.write(args.output / filename, filename)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Invalid ZIP archive")
    print(destination.resolve())
