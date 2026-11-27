"""Compact, read-only monitoring of the six serial MVP arms."""
import json
import math
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parent / "mvp_2plus2_steps40000_seed101"
jobs = [("3m", "pool-all9"), ("3m", "fixed-bot"), ("2s3z", "pool-all9"),
        ("2s3z", "fixed-bot"), ("MM4m", "fixed-bot"), ("MM8", "fixed-bot")]
result = {"time": datetime.now().astimezone().isoformat(), "completed": [], "active": []}
for sc, rg in jobs:
    d = root / sc / rg / "seed_101"
    if not d.is_dir():
        continue
    if (d / "done.json").exists():
        r = json.loads((d / "report.json").read_text(encoding="utf-8-sig"))
        e = r["final_eval"]
        cb = e["codebook_diagnostics"]
        result["completed"].append({
            "job": f"{sc} / {rg}", "all9": e["all9_win_rate"], "8algo": e["algos8_win_rate"],
            "worst3": e["worst3_algos_win_rate"], "bot": e["bot_win_rate"], "hp_margin": e["mean_hp_margin"],
            "K_final": e["active_prototypes"], "splits": cb["total_splits"], "merges": cb["total_merges"],
            "mean_clusters": e["mean_clusters"], "wall_seconds": r["wall_seconds"],
        })
        continue
    status = {"job": f"{sc} / {rg}", "steps": 0}
    p = d / "codebook_evolution.jsonl"
    if p.exists():
        rows = []
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if rows:
            status.update({k: rows[-1].get(k) for k in ["steps", "train_win_rate", "active_prototypes", "total_splits", "total_merges", "total_prunes"]})
            status["losses_finite"] = all(math.isfinite(float(v)) for row in rows for k, v in row.items() if k.endswith("_loss"))
    p = d / "learning_curve.json"
    if p.exists():
        try:
            c = json.loads(p.read_text(encoding="utf-8-sig"))
            status["last_validation"] = {k: c[-1].get(k) for k in ["checkpoint_steps", "all9_win_rate", "bot_win_rate", "mean_hp_margin"]}
        except (json.JSONDecodeError, IndexError):
            pass
    result["active"].append(status)
p = root / "first_arm_guard.json"
if p.exists():
    result["first_arm_guard"] = json.loads(p.read_text(encoding="utf-8"))["status"]
result["completed_count"] = len(result["completed"])
result["completed"] = result["completed"][-1:]
print(json.dumps(result, ensure_ascii=False))
