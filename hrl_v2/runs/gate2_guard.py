"""Read-only experiment checks; writes only the gate decision in the run root."""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    first = args.root / "3m" / "pool-all9" / "seed_101"
    if not (first / "done.json").exists():
        return 0
    report = json.loads((first / "report.json").read_text(encoding="utf-8-sig"))
    curve = json.loads((first / "learning_curve.json").read_text(encoding="utf-8-sig"))
    evo = [json.loads(line) for line in (first / "codebook_evolution.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    bad = [{"steps": row.get("steps"), "update": row.get("update"), "field": key, "value": str(value)}
           for row in evo for key, value in row.items()
           if key.endswith("_loss") and not math.isfinite(float(value))]
    wins = [float(row["all9_win_rate"]) for row in curve]
    near_zero = bool(wins) and max(wins) <= 0.01 and wins[-1] <= wins[0]
    reasons = []
    if report.get("train_steps") != 40000:
        reasons.append("First arm did not complete exactly 40000 training steps")
    if bad:
        reasons.append("Non-finite PPO/auxiliary loss")
    if not evo or not evo[0].get("initialized"):
        reasons.append("Missing successful online codebook initialization")
    if not wins or not all(math.isfinite(v) for v in wins):
        reasons.append("Missing/non-finite validation win rates")
    if near_zero:
        reasons.append("All validation all9 win rates <= 1%, with no final improvement")
    denom = sum(int(row.get("episodes_finished", 0)) for row in evo)
    weighted_train_win = sum(float(row.get("train_win_rate", 0)) * int(row.get("episodes_finished", 0)) for row in evo) / max(denom, 1)
    decision = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "status": "stopped" if reasons else "passed",
        "first_arm": str(first), "reasons": reasons, "nonfinite_losses": bad,
        "validation": [{key: row.get(key) for key in ("steps", "checkpoint_steps", "all9_win_rate", "bot_win_rate", "mean_hp_margin", "active_prototypes")} for row in curve],
        "weighted_train_win_rate": weighted_train_win,
        "last_codebook_update": evo[-1] if evo else None,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "first_arm_guard.json").write_text(json.dumps(decision, indent=2, allow_nan=False), encoding="utf-8")
    print("[FIRST ARM GUARD] " + decision["status"] + (": " + "; ".join(reasons) if reasons else ""), flush=True)
    return 42 if reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
