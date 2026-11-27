# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) Annotated matrices -> heatmap/plot_comparison.py -> param inherit
# (b) Checkpoint trends -> LineTrend/plot_posttraining.py -> param inherit
# (c) Opponent matrix and paired summary -> heatmap/plot_comparison.py -> param inherit
# RULE: native run panels use the source export; these layouts require v10-specific axes.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from .verify_v10_gates import CORE, REGIMES, SCENARIOS, SEEDS, STEPS, read_json
from .v10_suite import ALL_OPPONENTS, CORE_METHODS, resolve_regime_opponents


REPO = Path(__file__).resolve().parents[1]
METHOD_LABELS = ("IM-hash", "IM-medoid", "Unit-set", "IM-grid")
REGIME_LABELS = ("Bot only", "QMIX only", "DOP only", "VDN only", "COMA only", "8-policy pool", "9-opponent pool")
CMAP = LinearSegmentedColormap.from_list("sc2ba-sequential", SEQUENTIAL)
HRL = Path(r"D:\pycharm\PycharmProjects\HRL-IMCBS-test\state_runs\all-mm-encoders-20261125"
           r"\reports\mm-encoders-summary\metrics.csv")
METRICS = ("all9_win_rate", "algos8_win_rate", "worst3_algos_win_rate")
PAIR_KEYS = ["scenario", "method", "seed"]
MATCH_KEYS = [*PAIR_KEYS, "opponent"]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def panel(fig, ax, values, row_labels, column_labels, norm, title, bar_label, fmt, *, outline=None):
    check(np.isfinite(values).all(), f"Nonfinite cells in {title}")
    image = ax.imshow(values, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    for (i, j), value in np.ndenumerate(values):
        rgb = CMAP(norm(value))[:3]
        ink = "white" if np.dot(rgb, (0.2126, 0.7152, 0.0722)) < 0.47 else BLACK
        ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=6, color=ink)
        if outline is not None and outline[i, j]:
            ax.add_patch(Rectangle((j - .47, i - .45), .94, .90, fill=False,
                                   edgecolor=ink, linewidth=.65))
    bar = fig.colorbar(image, ax=ax, shrink=.78, pad=.025)
    bar.set_label(bar_label, fontsize=7)
    bar.ax.tick_params(labelsize=6, length=2)
    return image


def grid(frame, column):
    table = frame.pivot(index="scenario", columns="method", values=column)
    return table.loc[list(SCENARIOS), list(CORE_METHODS)].to_numpy(dtype=float)


def figure_a(summary, out):
    base = summary[summary["regime"] == "pool-all9"]
    check(len(base) == 52 and base["n_seeds"].eq(3).all(), "Incomplete A1 data")
    win_all = grid(base, "all9_wr_mean") * 100
    win_algos = grid(base, "algos8_wr_mean") * 100
    q = grid(base, "upper_q_mean")
    latency = grid(base, "encode_ms_p50")
    check((q > 0).all() and (latency > 0).all(), "Log-scaled metrics need positive values")
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 177/25.4), layout="constrained")
    for ax, values, norm, title, label, fmt in zip(
            axes.flat, (win_all, win_algos, q, latency),
            (Normalize(0, 100), Normalize(0, 100), LogNorm(q.min(), q.max()), LogNorm(latency.min(), latency.max())),
            ("a  All nine opponents", "b  Eight learned opponents", "c  Upper Q-table size", "d  Encoding latency"),
            ("Win rate (%)", "Win rate (%)", "Rows (log scale)", "p50 (ms, log scale)"),
            (".1f", ".1f", ".0f", ".2g")):
        panel(fig, ax, values, SCENARIOS, METHOD_LABELS, norm, title, label, fmt)
    save_cns_figure(fig, out / "v10a_all13_four_encoders")
    plt.close(fig)


def figure_b(curves, out):
    subset = curves[curves.scenario.isin(CORE)]
    check(len(subset) == 1680, "Incomplete core4 learning curves")
    points = subset.groupby(["scenario", "method", "regime", "checkpoint_steps"]).all9_win_rate.agg(["mean", "count"])
    check(points["count"].eq(3).all() and len(points) == 560, "Missing training seed in a curve")
    fig, axes = plt.subplots(4, 4, figsize=(183/25.4, 156/25.4), sharex=True, sharey=True)
    palette = [GREY, CATEGORICAL[3], CATEGORICAL[2], "#75507b", CATEGORICAL[5], CATEGORICAL[0], ACCENT_RED]
    markers = ("o", "^", "s", "v", "d", "P", "X")
    handles = []
    for i, scenario in enumerate(CORE):
        for j, method in enumerate(CORE_METHODS):
            ax = axes[i, j]
            ax.set_xlim(-.25, 8.25)
            ax.set_ylim(-2, 102)
            ax.set_xticks([0, 2, 4, 6, 8])
            ax.set_yticks([0, 50, 100])
            ax.tick_params(labelsize=6, length=2, pad=2)
            if i == 0:
                ax.set_title(METHOD_LABELS[j], fontweight="bold")
            if j == 0:
                ax.set_ylabel(scenario, fontweight="bold")
            for k, regime in enumerate(REGIMES):
                rows = points.loc[(scenario, method, regime)].loc[list(STEPS), "mean"].to_numpy() * 100
                check(np.isfinite(rows).all(), f"Invalid curve for {scenario}/{method}/{regime}")
                (line,) = ax.plot(np.asarray(STEPS) / 1000, rows, marker=markers[k], markersize=2.0,
                                  lw=1.1 if k >= 5 else .68, ls="-" if k >= 5 else 
                                  (":", "--", "-.", "--", ":")[k],
                                  color=palette[k], label=REGIME_LABELS[k])
                if i == 0 and j == 0:
                    handles.append(line)
    fig.subplots_adjust(left=.12, right=.98, bottom=.17, top=.94, wspace=.20, hspace=.25)
    fig.supxlabel("Training decisions (thousands)", y=.095)
    fig.supylabel("All-9 validation win rate (%)", x=.01, fontsize=7)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .008), ncols=4, fontsize=7)
    save_cns_figure(fig, out / "v10b_fixed_vs_pool_curves")
    plt.close(fig)


def figure_c(opponents, arms, out):
    values = opponents.groupby(["regime", "opponent"]).win_rate.agg(["mean", "count"])
    check(len(values) == 63 and values["count"].eq(48).all(), "Incomplete opponent heatmap")
    heat = values["mean"].unstack().loc[list(REGIMES), list(ALL_OPPONENTS)].to_numpy() * 100
    in_pool = np.array([[opponent in resolve_regime_opponents(regime) for opponent in ALL_OPPONENTS] for regime in REGIMES])
    fig, (left, right) = plt.subplots(1, 2, figsize=(183/25.4, 86/25.4),
                                     gridspec_kw={"width_ratios": [2.7, 1.35]}, layout="constrained")
    panel(fig, left, heat, REGIME_LABELS, tuple(o.upper() for o in ALL_OPPONENTS), Normalize(0, 100),
          "a  Training regime × test opponent", "Win rate (%)", ".1f", outline=in_pool)
    left.tick_params(axis="x", labelrotation=40)
    left.set_xlabel("Outlined cells were in the training pool", fontsize=7)

    y = np.arange(len(REGIMES))
    rates = arms.groupby("regime")[["in_dist_win_rate", "out_dist_win_rate"]].mean().loc[list(REGIMES)] * 100
    i, o = rates.in_dist_win_rate.to_numpy(), rates.out_dist_win_rate.to_numpy()
    for row, first, second in zip(y, i, o):
        if np.isfinite(second):
            right.plot([first, second], [row, row], color=GREY, linewidth=.75)
        else:
            right.text(99, row, "N/A", ha="right", va="center", fontsize=6, color=GREY)
    right.scatter(i, y, s=19, color=CATEGORICAL[0], marker="o", zorder=3, label="In-pool")
    right.scatter(o, y, s=16, facecolors="white", edgecolors=ACCENT_RED, marker="s", zorder=4, label="Out-pool")
    right.set_xlim(-1, 104)
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_ylim(len(y) - .5, -.5)
    right.set_yticks([])
    right.spines["left"].set_visible(False)
    right.set_xlabel("Win rate (%)")
    right.set_title("b  In- and out-of-pool means", loc="left", fontweight="bold", pad=8)
    right.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncols=2, fontsize=6)
    save_cns_figure(fig, out / "v10b_in_vs_out_dist_heatmap")
    plt.close(fig)


def rates(value):
    return f"{value*100:.2f}%" if np.isfinite(value) else "—"


def seed_mean_sd(values, *, signed=False):
    check(set(values.index) == set(SEEDS), "A summary requires all three paired seeds")
    if values.isna().all():
        return "—"
    check(np.isfinite(values).all(), "Nonfinite value in a three-seed summary")
    fmt = "+.2f" if signed else ".2f"
    return f"{format(values.mean() * 100, fmt)} ± {values.std(ddof=1) * 100:.2f}"


def section_table(lines, header, rows):
    lines += ["", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    lines.append("")


def group_best(a1, scenarios):
    scores = a1[a1.scenario.isin(scenarios)].groupby("method").all9_win_rate.mean().loc[list(CORE_METHODS)]
    highest = scores.max()
    return "、".join(name for name, value in scores.items() if np.isclose(value, highest)), highest


def write_report(arms, curves, opponent_rows, path, root):
    a1 = arms[arms.regime.eq("pool-all9")]
    core = arms[arms.scenario.isin(CORE)]
    bscore = core.groupby("regime")[list(METRICS) + ["in_dist_win_rate", "out_dist_win_rate"]].mean().loc[list(REGIMES)]
    best_a, best_rate = group_best(a1, SCENARIOS)
    best_regime = bscore["algos8_win_rate"].idxmax()
    gate0 = read_json(root / "gates" / "stage0.json")
    gate1 = read_json(root / "gates" / "stage1.json")
    protoss_scenarios = ("2s3z", "3s5z", "1c3s5z", "3s5z_vs_3s6z", "3z_vs_3s", "5z_vs_3s")
    lines = [
        "# SC2BA v10 实机实验结果", "",
        "批次 v10-all13-20261126，StarCraft II 5.0.16.97425。",
        "Stage 0–3 均已通过验收。A1 共 156 个训练项，B1 共 336 个，其中 48 个复用 A1 的产物并通过指纹一致性校验。去重后合计 444 个训练项、148 组条件、3,552,000 个训练决策步。", "",
        f"Stage 0 的 117 局共执行 {gate0['steps']:,} 步，13/13 场景通过且引擎零重启。医疗兵保留同目标上的在途治疗，并在修复后重验全量场景。",
        f"Stage 1 的 {gate1['scenarios']} 个场景共采集 1,872 局探针，训练、验证和测试折均齐全。"
        f"最终冻结 {gate1['encoders']} 个编码器，其中 {gate1['cuda_encoders']} 个神经编码器由 RTX 4070 拟合。"
        "IM-medoid 的测试误差由训练折的码字中位数读出估计，神经编码器的测试误差来自冻结监督头。",
        "A1 的 6 worker 运行在一次 SC2 websocket 启动失败后保留已完成结果，以 4 worker 续跑。B1 使用 4 worker。", "",
        "红方训练分层 Q 策略。九对手库由难度 7 的内置 Bot 和八个场景专用冻结策略组成，分别是 QMIX、VDN、DOP、FOP、QPLEX、COMA、IQL 与 QTRAN。"
        "每轮训练打乱并遍历允许的布局与对手组合，所有条件按相同决策步预算截断。",
        "",
        "## A1 四种状态编码",
        "",
        f"在 13 个场景上等权汇总，All-9 最高为 {best_a}（{rates(best_rate)}）。胜率表中的误差为先对场景取平均、再对三个种子计算的样本标准差。"
    ]
    a1_seed = a1.groupby(["method", "seed"]).mean(numeric_only=True)
    section_table(lines, ("编码", "All-9（%）", "8-Algo（%）", "Worst-3（%）", "Bot（%）",
                          "平均上层 Q 行数", "平均未见键率", "平均编码 p50（ms）"), (
        (method, *(seed_mean_sd(a1_seed.loc[method, metric]) for metric in (*METRICS, "bot_win_rate")),
         f"{a1_seed.loc[method, 'upper_q_rows'].mean():.1f}",
         rates(a1_seed.loc[method, "upper_unseen_rate"].mean()),
         f"{a1_seed.loc[method, 'encode_ms_p50'].mean():.3f}")
        for method in CORE_METHODS
    ))
    section_table(lines, ("场景组", "场景数", *METHOD_LABELS), (
        (name, len(group),
         *(rates(a1.loc[a1.scenario.isin(group) & a1.method.eq(method), "all9_win_rate"].mean())
           for method in CORE_METHODS))
        for name, group in (
            ("七个对称场景", SCENARIOS[:7]),
            ("三个人族非对称场景", ("5m_vs_6m", "10m_vs_11m", "MMM2")),
            ("三个新注册场景", ("3s5z_vs_3s6z", "3z_vs_3s", "5z_vs_3s")),
            ("神族场景", protoss_scenarios),
            ("人族异构含医疗兵", ("MMM", "MMM2")),
            ("25v25", ("25m",)),
        )
    ))
    section_table(lines, ("场景", "IM-hash", "IM-medoid", "Unit-set", "IM-grid", "最高 near 表现"), (
        (scenario, *(rates(a1[(a1.scenario == scenario) & (a1.method == m)].all9_win_rate.mean()) for m in CORE_METHODS),
         rates(a1[a1.scenario == scenario].groupby("method").near_win_rate.mean().max()))
        for scenario in SCENARIOS
    ))
    section_table(lines, ("编码", "standard All-9（%）", "near All-9（%）", "near − standard（百分点）"), (
        (method, seed_mean_sd(a1_seed.loc[method, "standard_win_rate"]),
         seed_mean_sd(a1_seed.loc[method, "near_win_rate"]),
         seed_mean_sd(a1_seed.loc[method, "near_win_rate"] - a1_seed.loc[method, "standard_win_rate"], signed=True))
        for method in CORE_METHODS
    ))
    hash_rows = a1[a1.method.eq("im-hash")].upper_q_rows.mean()
    hash_ms = a1[a1.method.eq("im-hash")].encode_ms_p50.mean()
    unit_ms = a1[a1.method.eq("unit-set")].encode_ms_p50.mean()
    protoss = a1[a1.scenario.isin(protoss_scenarios)]
    protoss_rate = protoss.groupby("method").all9_win_rate.mean()
    large_map_peak = a1[a1.scenario.eq("25m")].groupby("method").all9_win_rate.mean().max()
    lines += [
        f"原始哈希平均有 {hash_rows:.1f} 个上层 Q 行，约为固定码本 32 行的 {hash_rows / 32:.1f} 倍。固定码本的未见键率为零，表示查询总能落入预定义码字。",
        "IM-hash 与 IM-medoid 共用势力图及 24 位哈希前端。IM-medoid 将不同哈希映射到 32 个代表码字，使这些输入共享 Q 行，也会合并部分战术状态。"
        "Unit-set 从单位属性和相对位置提取集合特征，保留兵种与个体信息。IM-grid 在有符号网格上保留空间结构后再量化。"
        "两种神经编码的前端、嵌入和距离同时变化，因此胜率衡量整条编码路径的效果。",
        f"IM-hash 和 Unit-set 的逐 arm 编码 p50 均值分别为 {hash_ms:.3f} 与 {unit_ms:.3f} ms。延迟包含各自的实际预处理与编码路径。",
        f"在六个神族场景上，Unit-set 较 IM-hash 的 All-9 平均差为 "
        f"{100 * (protoss_rate['unit-set'] - protoss_rate['im-hash']):+.2f} 个百分点。"
        f"25v25 上三种固定码本都将 Q 表压到 32 行，四种编码的 All-9 均值最高为 {rates(large_map_peak)}。",
        "场景类型有交叠。25v25 的单位密度、神族场景的护盾以及 MMM/MMM2 的治疗动作都可能改变编码可辨性与策略收益。表中分组差异来自地图、编队和对手共同作用。",
        "",
        "## B1 固定单对手与训练池",
        "",
        f"固定对手与池训练在相同编码器、种子和 8,000 决策步下比较。8-Algo 均值最高的是 {best_regime}（{rates(bscore.loc[best_regime, 'algos8_win_rate'])}）。"
        "下表先在同一种子内对四个场景、四种编码等权，再汇总三个种子的均值与样本标准差。",
    ]
    b1_seed = core.groupby(["regime", "seed"]).mean(numeric_only=True)
    section_table(lines, ("制度", "All-9（%）", "8-Algo（%）", "Worst-3（%）", "池内（%）", "池外（%）"), (
        (regime, *(seed_mean_sd(b1_seed.loc[regime, metric]) for metric in bscore.columns))
        for regime in REGIMES
    ))
    lines += [
        "池外范围对固定组是其余八类对手，对 pool-algos8 仅是原生 Bot。pool-all9 覆盖全部九类，池外结果记缺项。"
        "各制度的池内、池外均值受所含对手的难度影响。训练池成员资格只针对 Q 表训练，冻结编码器的探针训练折覆盖九个对手。"
    ]

    contrasts = []
    specialization = []
    fixed_arms = core[core.regime.isin(REGIMES[:5])]
    fixed_mean = fixed_arms.groupby(PAIR_KEYS)[list(METRICS)].mean()
    method_contrasts = []
    best_fixed = bscore.loc[list(REGIMES[:5]), "algos8_win_rate"].idxmax()
    for pool in REGIMES[-2:]:
        pooled = core[core.regime.eq(pool)].set_index(PAIR_KEYS)
        pool_matchups = opponent_rows.loc[opponent_rows.regime.eq(pool), MATCH_KEYS + ["win_rate"]]
        for fixed in REGIMES[:5]:
            baseline = fixed_arms[fixed_arms.regime.eq(fixed)].set_index(PAIR_KEYS)
            delta = pooled.loc[baseline.index, list(METRICS)] - baseline[list(METRICS)]
            fixed_matchups = opponent_rows.loc[opponent_rows.regime.eq(fixed), MATCH_KEYS + ["win_rate", "in_train_pool"]]
            common = fixed_matchups.merge(pool_matchups, on=MATCH_KEYS, suffixes=("_fixed", "_pool"),
                                          validate="one_to_one")
            common["delta"] = common.win_rate_pool - common.win_rate_fixed
            seen = common[common.in_train_pool.eq(1)]
            unseen = common[common.in_train_pool.eq(0)]
            check(len(seen) == 48 and len(unseen) == 384, f"Wrong common test targets for {fixed}/{pool}")
            if seen.delta.mean() < 0 < unseen.delta.mean():
                specialization.append(f"{pool} 对 {fixed}")
            seed_delta = delta.groupby("seed").mean()
            contrasts.append((
                pool, fixed, *(seed_mean_sd(seed_delta[metric], signed=True) for metric in METRICS),
                seed_mean_sd(seen.groupby("seed").delta.mean(), signed=True),
                seed_mean_sd(unseen.groupby("seed").delta.mean(), signed=True),
                f"{int(delta.algos8_win_rate.gt(0).sum())}/48",
            ))
        by_method = (pooled.loc[fixed_mean.index, list(METRICS)] - fixed_mean).groupby(["method", "seed"]).mean()
        method_contrasts.extend((pool, method, *(seed_mean_sd(by_method.loc[method, m], signed=True) for m in METRICS))
                                for method in CORE_METHODS)

    lines += ["", "下表为同场景、同编码、同种子的混池减单训差值，单位为百分点。"]
    section_table(lines, ("混池", "单训基准", "Δ All-9", "Δ 8-Algo", "Δ Worst-3",
                          "单训目标 Δ", "单训池外八对手 Δ", "8-Algo 正差配对数"), contrasts)
    fixed_scores = bscore.loc[list(REGIMES[:5]), "algos8_win_rate"]
    num_fixed_beaten = int(fixed_scores.lt(bscore.loc["pool-all9", "algos8_win_rate"]).sum())
    delta_best = bscore.loc["pool-all9"] - bscore.loc[best_fixed]
    lines.append(f"全混池在 8-Algo 上的均值超过五种单训中的 {num_fixed_beaten} 种。相对均值最高的单训 {best_fixed}，"
                 f"全混池的 8-Algo 差为 {delta_best['algos8_win_rate'] * 100:+.2f} 个百分点，"
                 f"Worst-3 差为 {delta_best['worst3_algos_win_rate'] * 100:+.2f} 个百分点。")
    lines.append(f"10 个制度对照中，{len(specialization)} 个呈现单训目标上单训占优、其余八个对手上混池占优。")
    if specialization:
        lines.append("出现这种专化模式的对照为 " + "、".join(specialization) + "。")
    lines += [
        "同一测试对手的配对控制了基准难度。单训池外的对手可能已进入混合组训练池，表中的差值衡量扩大训练池后的跨对手收益。",
        "",
        "按编码进一步比较混池与五种单训的等权均值，差值单位为百分点。",
    ]
    section_table(lines, ("混池", "编码", "Δ All-9", "Δ 8-Algo", "Δ Worst-3"), method_contrasts)

    b1_curves = curves[curves.scenario.isin(CORE)].groupby(["regime", "checkpoint_steps", "seed"]).all9_win_rate.mean()
    section_table(lines, ("训练制度", "0 步 All-9（%）", "8k 步 All-9（%）", "6k→8k（百分点）"), (
        (regime, seed_mean_sd(b1_curves.loc[regime, 0]), seed_mean_sd(b1_curves.loc[regime, 8000]),
         seed_mean_sd(b1_curves.loc[regime, 8000] - b1_curves.loc[regime, 6000], signed=True))
        for regime in REGIMES
    ))
    lines += [
        "训练曲线使用保存于各决策步的 Q 表，在独立 SC2 会话的 diagonal 布局重放评估。每个检查点对九个对手各评 2 局，共 18 局。"
        "终局分别在 standard 和 near 开局对每个对手评 5 局，每个训练项合计 90 局。standard 是训练也使用的规范布局，near 是留出的布局。",
        "每个检查点对各制度使用配对评估种子，评估种子随检查点变化。本实验采用固定步数对照，6k 到 8k 的差值由两个验证点估计。"
        "结论对应所用决策预算、固定对手权重和指定布局。",
        "",
        "## 与前序实验的对照",
        "",
    ]
    if HRL.is_file():
        h = pd.read_csv(HRL)
        final_h = h[h.steps.eq(40000)]
        hstats = final_h.groupby("method").agg({"win_pp": "mean", "upper_rows": "mean", "unseen_pp": "mean"})
        labels = {"im": "IM-hash", "im-medoid-32": "IM-medoid", "independent-raw-medoid-32": "Unit-set",
                  "grid-zero-medoid-32": "IM-CNN"}
        lines.append("HRL-IMCBS v09 六个 MM 场景在 40,000 决策步的描述性结果如下。每种编码合计 60 个场景与种子配对项。")
        section_table(lines, ("v09 架构", "平均胜率", "平均 Q 行数", "平均未见键率"), (
            (name, f"{hstats.loc[key,'win_pp']:.2f}%", f"{hstats.loc[key,'upper_rows']:.1f}",
             f"{hstats.loc[key,'unseen_pp']:.2f}%") for key, name in labels.items()
        ))
        lines.append("v09 与 v10 的地图、对手集合、预算和评估开局不同。Q 行数和未见键率的结构性效应可以对读，胜率保留各自协议。"
                     " [v09 逐种子结果](D:/pycharm/PycharmProjects/HRL-IMCBS-test/state_runs/all-mm-encoders-20261125/reports/mm-encoders-summary/metrics.csv)")
    b01 = read_json(REPO / "docs" / "b01" / "data" / "report.json")
    top = b01["ranking"][0]
    lines += [
        f"B01 的双 Agent 冻结 3m 矩阵中，{top['policy'].upper()} 等权平均胜率为 {rates(top['mean_win_rate'])} 居表首。B1 的固定 QMIX 是训练对手，评价的是新的上层 Q 策略在完整对手库上的表现。",
        "B02a 在单个配对种子、约 200 万步、VDN 端到端学习的验证集中，混合训练较 QMIX 单训的八算法平均差为 +0.52 个百分点，出生点配对 95% 区间为 [−2.60, +3.47]。"
        f"v10 的同方向对比为 {(bscore.loc['pool-all9', 'algos8_win_rate'] - bscore.loc['fixed-qmix', 'algos8_win_rate']) * 100:+.2f} 个百分点。"
        "训练算法、预算、场景与评估布局有差异，两个百分点差值保留各自协议。 [B01 和 B02a 记录](EXPERIMENT_RESULTS.md)",
        "",
        "## 图表口径与复现",
        "",
        "- 图 1 展示全部 52 个 A1 条件的种子均值。Q 表行数和编码 p50 使用对数色标，格内数值使用原始单位。"
        " [图 1](../bch/runs/v10_all13/analysis/v10a_all13_four_encoders.pdf) · [逐条件结果](../bch/runs/v10_all13/analysis/summary_by_condition.csv)",
        "- 图 2 展示 4×4 面板内全部七个制度在 0/2k/4k/6k/8k 的三种子均值。"
        " [图 2](../bch/runs/v10_all13/analysis/v10b_fixed_vs_pool_curves.pdf) · [逐种子曲线](../bch/runs/v10_all13/analysis/learning_curves.csv)",
        "- 图 3 每格使用 4 场景 × 4 编码 × 3 种子，合计 48 个训练项，每项在该对手上评 10 局。描框表示训练池内。"
        " [图 3](../bch/runs/v10_all13/analysis/v10b_in_vs_out_dist_heatmap.pdf) · [逐对手结果](../bch/runs/v10_all13/analysis/opponent_matrix.csv)",
        "- 胜率以胜局除以全部评估局，平局和超时都计入分母。Worst-3 先在每个训练项内取胜率最低的三个冻结算法对手，再对训练项求平均。"
        " 表中的标准差是三种子描述统计。 [逐种子终局结果](../bch/runs/v10_all13/analysis/arms_per_seed.csv)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_figures(out, names):
    for name in names:
        with Image.open(out / f"{name}.png") as image:
            dpi = image.info.get("dpi", (0, 0))
            check(min(dpi) >= 299 and image.width > 1500 and image.height > 650, f"Wrong PNG export for {name}")
            image.verify()
        check((out / f"{name}.pdf").read_bytes().startswith(b"%PDF-"), f"Missing PDF for {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render paper figures and analyze the verified v10 arms.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for stage in range(4):
        check(read_json(args.out_root / "gates" / f"stage{stage}.json")["status"] == "PASS",
              f"Gate {stage} has not passed")
    out = args.out_root / "analysis"
    arms = pd.read_csv(out / "arms_per_seed.csv")
    summary = pd.read_csv(out / "summary_by_condition.csv")
    curves = pd.read_csv(out / "learning_curves.csv")
    opponents = pd.read_csv(out / "opponent_matrix.csv")
    check((len(arms), len(summary), len(curves), len(opponents)) == (444, 148, 2220, 3996),
          "Stage 4 requires all A1 and B1 artifacts")
    figure_a(summary, out)
    figure_b(curves, out)
    figure_c(opponents[opponents.scenario.isin(CORE)], arms[arms.scenario.isin(CORE)], out)
    write_report(arms, curves, opponents[opponents.scenario.isin(CORE)], args.report, args.out_root)
    verify_figures(out, ("v10a_all13_four_encoders", "v10b_fixed_vs_pool_curves",
                         "v10b_in_vs_out_dist_heatmap"))
    print(f"Verified 444 arms and exported three PDF/PNG pairs and {args.report}.", flush=True)
