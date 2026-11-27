"""Validate completed artifacts and record the requested compact result table."""
import json
import math
import re
from datetime import datetime
from pathlib import Path

runs = Path(__file__).resolve().parent
root = runs / "mvp_2plus2_steps40000_seed101"
jobs = [("3m", "pool-all9"), ("3m", "fixed-bot"), ("2s3z", "pool-all9"),
        ("2s3z", "fixed-bot"), ("MM4m", "fixed-bot"), ("MM8", "fixed-bot")]
required = ["done.json", "report.json", "learning_curve.json", "codebook_evolution.jsonl"]
summary = {"verified_at": datetime.now().astimezone().isoformat(), "gate1": json.loads((runs / "smoke_real" / "gate1_summary.json").read_text()), "gate2": []}
for sc, rg in jobs:
    d = root / sc / rg / "seed_101"
    assert all((d / name).is_file() for name in required), f"Missing required artifacts: {d}"
    report = json.loads((d / "report.json").read_text())
    cfg = report["config"]
    assert report["train_steps"] == cfg["train_steps"] == 40000
    assert cfg["seed"] == 101 and cfg["eval_every"] == 8000 and cfg["rollout_steps"] == 1024 and cfg["device"] == "cuda"
    curve = json.loads((d / "learning_curve.json").read_text())
    evo = [json.loads(s) for s in (d / "codebook_evolution.jsonl").read_text().splitlines() if s.strip()]
    assert len(evo) == 40 and evo[-1]["steps"] == 40000 and all(row["initialized"] for row in evo)
    loss_values = [float(value) for row in evo for key, value in row.items() if key.endswith("_loss")]
    assert all(math.isfinite(v) for v in loss_values), f"Nonfinite loss: {d}"
    assert len(curve) == 6 and curve[0]["checkpoint_steps"] == 0 and curve[-1]["checkpoint_steps"] == 40000
    final_rows = [json.loads(s) for s in (d / "final_eval_episodes.jsonl").read_text().splitlines() if s.strip()]
    legacy = report["backend"] == "legacy_mm"
    assert len(final_rows) == (120 if legacy else 90)
    fe = report["final_eval"]
    cb = fe["codebook_diagnostics"]
    summary["gate2"].append({
        "scenario": sc, "regime": rg, "backend": report["backend"], "out_dir": str(d),
        "train_steps": report["train_steps"], "final_eval_episodes": len(final_rows),
        "all9": None if legacy else fe["all9_win_rate"], "algos8": None if legacy else fe["algos8_win_rate"],
        "worst3": None if legacy else fe["worst3_algos_win_rate"], "bot": fe["bot_win_rate"],
        "hp_margin": fe["mean_hp_margin"], "K_final": fe["active_prototypes"],
        "splits": cb["total_splits"], "merges": cb["total_merges"], "prunes": cb["total_prunes"],
        "mean_clusters": fe["mean_clusters"], "wall_seconds": report["wall_seconds"],
        "finite_loss_values_checked": len(loss_values), "online_initialized": True,
    })
log_text = "\n".join((root / name).read_text(encoding="utf-8-sig", errors="replace") for name in ["runner.stdout.log", "runner.stderr.log"])
errors = re.findall(r"Traceback|AssertionError|ConnectionError|ConnectionReset|ProtocolError|SpawnError|D3DERR|illegal action|invalid action|Retrying the SC2 scene reset", log_text, re.I)
assert not errors, errors
assert "All selected DyIM-CHRL MVP jobs completed cleanly" in log_text
summary["observed_runtime_error_matches"] = errors
summary["runtime_error_check_scope"] = "Successful run console logs; no separate illegal-action/disconnection counters are emitted by the existing trainer."
summary["first_arm_guard"] = json.loads((root / "first_arm_guard.json").read_text())
assert summary["first_arm_guard"]["status"] == "passed"
summary["source_fixes"] = {
    "hrl_v2/run_experiment.py": "Synchronize tempfile.tempdir after setting isolated TEMP/TMP; SC2's mkdtemp otherwise reused the cached parent.",
    "hrl_v2/env_adapter.py": "Legacy: add the required very_hard Terran Bot player and disable_fog=True to match the existing Legacy environment and make fast-reset spawn observations complete.",
}
summary["legacy_metric_note"] = "Legacy evaluates only bot. Its raw report aliases all9/algos8/worst3 to bot; the summary uses null/NA for those inapplicable pool metrics."
summary["total_gate2_wall_seconds"] = round(sum(x["wall_seconds"] for x in summary["gate2"]), 2)
(root / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
lines = ["# DyIM-CHRL 2+2 实机最小验证", "", "Gate 1：3/3 通过。Gate 2：6/6 完成，每组 40,000 步、seed=101、CUDA、严格串行。", "", "| scenario / regime | all9 / 8algo / worst3 / bot | hp_margin | K_final (splits/merges) | mean_clusters | wall_seconds |", "|---|---|---:|---:|---:|---:|"]
for x in summary["gate2"]:
    rates = " / ".join("—" if x[k] is None else f"{x[k]:.2%}" for k in ["all9", "algos8", "worst3", "bot"])
    lines.append(f"| {x['scenario']} / {x['regime']} | {rates} | {x['hp_margin']:+.2f} | {x['K_final']} ({x['splits']}/{x['merges']}) | {x['mean_clusters']:.2f} | {x['wall_seconds']:.2f} |")
lines += ["", "Legacy 仅评估 bot；前三项不适用，原 report.json 对这三项复用了 bot 值。", "", "Gate 1 耗时 / 有效步速（训练步数 ÷ 训练及评估耗时）："]
for x in summary["gate1"]:
    lines.append(f"- {x['scenario']} / {x['regime']}: {x['wall_seconds']:.2f} s / {x['effective_steps_per_second']:.2f} steps/s；K: {x['K_initial']}→{x['K_final']}。")
lines += ["", "首组早停护栏通过：所有 loss 有限，验证 all9 从 0% 升至最高 27.78%，最终检查点 22.22%。", "", "定点修补：run_experiment.py 同步临时目录缓存；env_adapter.py 补齐 Legacy Bot 玩家及 disable_fog=True。原串行脚本未修改，外层包装在每个作业边界执行首组护栏。", "", "所有必需产物、6 个检查点、每组 40 次更新及最终评估局数已核对；成功运行日志未检测到非法动作或 SC2 断连异常。原训练器没有单独输出这两类事件计数。", "", "未执行 git commit / push；未修改 bch、bch/runs 或 Legacy 仓库文件，未结束既有 v11/SC2 进程。"]
(root / "VALIDATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"completed": len(summary["gate2"]), "total_gate2_wall_seconds": summary["total_gate2_wall_seconds"], "results": summary["gate2"]}, ensure_ascii=False))
