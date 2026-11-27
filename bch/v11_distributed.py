"""SC2BA v11 paired experiments with stage barriers and explicit host ownership."""

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import psutil
import re
import signal
import subprocess
import sys
from threading import Event, Lock
from time import perf_counter
import urllib.request

from .plan import plan_digest, read_plan
from .records import digest_file, write_json
from .v10_suite import (
    ALL_OPPONENTS, CORE_METHODS, DEFAULT_PLAN, SCENARIO_GROUPS,
    _worker_env, aggregate_results, resolve_regime_opponents,
)

V11_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
V11_B1_NET_REGIMES = ("fixed-bot", "fixed-qmix", "fixed-dop", "fixed-vdn", "fixed-coma", "pool-algos8")
V11_DEFAULT_STEPS = 40000
V11_DEFAULT_EVAL_EVERY = 8000
DEFAULT_ROOT = Path(__file__).resolve().parent / "runs" / "v11_40k_10seed"
DEFAULT_ENCODERS = Path(__file__).resolve().parent / "runs" / "v10_all13" / "encoders"
RESULT_FILES = ("report.json", "policy.json", "learning_curve.json", "train_episodes.jsonl", "val_episodes.jsonl", "test_episodes.jsonl")


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def manifest_digest(manifest):
    content = {key: value for key, value in manifest.items() if key != "created_at"}
    return sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_v11_manifest(methods=CORE_METHODS, seeds=V11_SEEDS, train_steps=40000,
                       eval_every=8000, include_b1=True,
                       val_repeats=2, test_repeats=5):
    require(tuple(methods) == CORE_METHODS, "v11 requires all four paired encoders")
    require(len(seeds) == len(set(seeds)) and len(seeds) > 0, "Seeds must be distinct")
    require(train_steps > 0 and eval_every > 0, "Step budgets must be positive")
    groups = []
    for scenario in SCENARIO_GROUPS["all13"]:
        for seed in seeds:
            groups.append({"track": "v11-A1", "scenario": scenario, "regime": "pool-all9", "seed": seed})
    if include_b1:
        for scenario in SCENARIO_GROUPS["core4"]:
            for regime in V11_B1_NET_REGIMES:
                for seed in seeds:
                    groups.append({"track": "v11-B1", "scenario": scenario, "regime": regime, "seed": seed})
    for index, group in enumerate(groups):
        group.update(index=index, group_id=f"{group['track']}/{group['scenario']}/{group['regime']}/seed_{group['seed']}",
                     methods=list(methods), train_steps=train_steps, eval_every=eval_every)
    return {
        "format": "bch-sc2ba-v11-manifest-v2", "batch_id": "v11_40k_10seed", "created_at": iso_now(),
        "train_steps": train_steps, "eval_every": eval_every, "seeds": list(seeds), "methods": list(methods),
        "val_repeats": val_repeats, "test_repeats": test_repeats,
        "total_groups": len(groups),
        "total_arms": len(groups) * len(methods), "groups": groups,
    }


def arm_key(group, method):
    return f"{group['scenario']}/{group['regime']}/{method}/seed_{group['seed']}"


def write_status(arm_dir, payload):
    arm_dir.mkdir(parents=True, exist_ok=True)
    write_json(arm_dir / "status.json", payload)


def post_wnc_report(wnc_url, payload):
    if not wnc_url:
        return
    request = urllib.request.Request(
        wnc_url, data=json.dumps({"action": "distributed-report", **payload}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:4321"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=4):
            pass
    except (OSError, ValueError):
        pass


def release_gpu_hold_if_present(script, gpu_index, state_dir=None):
    require(script and Path(script).is_file(), "GPU dispatch requires a readable --gpu-hold-script")
    command = [sys.executable, str(script)]
    if state_dir:
        command.extend(["--state-dir", str(state_dir)])
    release = subprocess.run([*command, "release", "--gpu", str(gpu_index)],
                             capture_output=True, text=True, timeout=30)
    require(release.returncode == 0, f"GPU {gpu_index} hold release failed: {release.stderr[-800:]}")
    # The utility's release command can return success while a foreign process is present.
    status = subprocess.run([*command, "status"], capture_output=True, text=True, timeout=20)
    require(status.returncode == 0, f"GPU {gpu_index} status failed")
    payload = json.loads(status.stdout)
    devices = payload.get("gpus", [])
    target = next((item for item in devices if item["index"] == gpu_index), None)
    require(target is not None and target.get("idle") is True, f"GPU {gpu_index} is occupied after hold release")


@contextmanager
def runner_lock(root, host_id):
    folder = root / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{host_id}.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_worker_output(folder, group, method, manifest):
    report = read_json(folder / "report.json")
    identity = (group["scenario"], group["regime"], method, group["seed"])
    require(tuple(report[k] for k in ("scenario", "regime", "method", "seed")) == identity,
            f"Wrong report identity for {identity}")
    require(report["actual_train_steps"] == report["train_step_budget"] == group["train_steps"],
            f"Wrong step budget for {identity}")
    require(report["checkpoint_protocol"] == "exact_budget_frozen_replay_v1", "Unexpected checkpoint protocol")
    require(report["plan_sha256"] == manifest["plan_sha256"], "Plan fingerprint mismatch")
    expected_encoder = None if method == "im-hash" else manifest["encoder_sha256"][f"{group['scenario']}/{method}"]
    require(report["encoder_sha256"] == expected_encoder, "Encoder fingerprint mismatch")
    require(set(report["train_opponents"]) == set(resolve_regime_opponents(group["regime"])) and
            set(report["eval_opponents"]) == set(ALL_OPPONENTS), "Opponent protocol mismatch")
    steps = list(range(0, group["train_steps"], group["eval_every"])) + [group["train_steps"]]
    checkpoints = read_json(folder / "learning_curve.json")["checkpoints"]
    require([row["checkpoint_steps"] for row in checkpoints] == steps and
            all(row["actual_train_steps"] == step and row["episodes"] == 9 * manifest["val_repeats"]
                for row, step in zip(checkpoints, steps)), "Incomplete frozen validation checkpoints")
    for filename, expected in (("val_episodes.jsonl", len(steps) * 9 * manifest["val_repeats"]),
                               ("test_episodes.jsonl", 18 * manifest["test_repeats"])):
        with (folder / filename).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        require(len(rows) == expected, f"Incomplete {filename}")
        if filename == "val_episodes.jsonl":
            expected_keys = {(step, "diagonal", opponent, repeat)
                             for step in steps for opponent in ALL_OPPONENTS for repeat in range(manifest["val_repeats"])}
        else:
            expected_keys = {(group["train_steps"], layout, opponent, repeat)
                             for layout in ("standard", "near") for opponent in ALL_OPPONENTS
                             for repeat in range(manifest["test_repeats"])}
        actual_keys = {(row["checkpoint_steps"], row["layout"], row["opponent"], row["repeat"]) for row in rows}
        require(actual_keys == expected_keys, f"Wrong or repeated matchups in {filename}")
    require((folder / "policy.json").is_file() and (folder / "train_episodes.jsonl").is_file(),
            "Missing policy or training trajectories")
    require(read_training_steps(folder / "train_episodes.jsonl") == group["train_steps"], "Truncated training trajectory")
    require(report["test"]["episodes"] == 18 * manifest["test_repeats"] and
            all(math.isfinite(report["test"][metric]) for metric in ("all9_win_rate", "algos8_win_rate", "encode_ms_p50")),
            "Invalid final evaluation metrics")
    if method != "im-hash":
        require(report["upper_q_rows"] <= 32 and report["test"]["upper_unseen_rate"] == 0, "Invalid frozen codebook result")
    return report


class CompletionVerifier:
    """Cache fully checked results while their files and marker remain unchanged."""

    def __init__(self, root, manifest):
        self.root = root
        self.manifest = manifest
        self.digest = manifest_digest(manifest)
        self.cache = {}
        self.lock = Lock()

    def valid(self, group, method):
        folder = self.root / "arms" / arm_key(group, method)
        try:
            paths = [folder / name for name in (*RESULT_FILES, "done.json")]
            signature = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
            with self.lock:
                if self.cache.get(str(folder)) == signature:
                    return True
            marker = read_json(folder / "done.json")
            require(marker["status"] == "done" and marker["manifest_sha256"] == self.digest, "Completion protocol mismatch")
            owner = marker["host_id"]
            require(isinstance(owner, str) and bool(owner), "Missing completion host")
            require(set(marker["files_sha256"]) == set(RESULT_FILES), "Incomplete completion fingerprint list")
            for name, fingerprint in marker["files_sha256"].items():
                require(digest_file(folder / name) == fingerprint, f"Incomplete transfer of {name}")
            report = validate_worker_output(folder, group, method, self.manifest)
            require(report["host_id"] == owner, "Report and completion host differ")
            with self.lock:
                self.cache[str(folder)] = signature
            return True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return False

    def stage_complete(self, track):
        groups = [group for group in self.manifest["groups"] if group["track"] == track]
        return bool(groups) and all(self.group_complete(group) for group in groups)

    def group_complete(self, group):
        if not all(self.valid(group, method) for method in group["methods"]):
            return False
        hosts = {read_json(self.root / "arms" / arm_key(group, method) / "done.json")["host_id"]
                 for method in group["methods"]}
        return len(hosts) == 1


def stage_groups(manifest, track, host_id, direction):
    groups = [group for group in manifest["groups"] if group["track"] == track]
    return sorted(groups, key=lambda group: group["index"], reverse=direction == "backward")


def read_training_steps(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return int(json.loads(line)["cum_steps"])
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        pass
    return 0


def publish_result(work_dir, arm_dir, group, method, manifest, metadata):
    report = validate_worker_output(work_dir, group, method, manifest)
    report.update(metadata)
    write_json(work_dir / "report.json", report)
    hashes = {name: digest_file(work_dir / name) for name in RESULT_FILES}
    arm_dir.mkdir(parents=True, exist_ok=True)
    require(not (arm_dir / "done.json").exists(), f"Refusing to replace completed result {arm_dir}")
    for name in RESULT_FILES:
        (work_dir / name).replace(arm_dir / name)
    write_json(arm_dir / "done.json", {
        "status": "done", "manifest_sha256": manifest_digest(manifest),
        "host_id": metadata["host_id"], "files_sha256": hashes,
        "wall_seconds": report["wall_seconds"], "completed_at": iso_now(),
    })
    return report


def stop_owned_process(process, children=()):
    """Stop only the process tree created for this arm."""
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
        for child in children:
            try:
                if child.is_running():
                    child.terminate()
            except psutil.NoSuchProcess:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def cmd_init(args):
    root = Path(args.out_root)
    manifest = build_v11_manifest(
        methods=tuple(args.methods.split(",")), seeds=tuple(int(seed) for seed in args.seeds.split(",")),
        train_steps=args.train_steps, eval_every=args.eval_every, include_b1=not args.a1_only,
        val_repeats=args.val_repeats, test_repeats=args.test_repeats)
    document, _, _, _ = read_plan(args.plan)
    manifest["plan_sha256"] = plan_digest(document)
    encoders = Path(args.encoders_root)
    manifest["encoder_sha256"] = {
        f"{scenario}/{method}": digest_file(encoders / scenario / method / "encoder.pt")
        for scenario in SCENARIO_GROUPS["all13"] for method in CORE_METHODS[1:]
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "v11_queue_manifest.json"
    if path.exists():
        if manifest_digest(read_json(path)) == manifest_digest(manifest):
            print(f"[v11 init] Verified existing {manifest['total_groups']} groups / {manifest['total_arms']} arms", flush=True)
            return
        require(not any((root / "arms").rglob("done.json")), "Completed arms require their original manifest")
    write_json(path, manifest)
    print(f"[v11 init] Wrote {manifest['total_groups']} paired groups / {manifest['total_arms']} arms -> {path}", flush=True)


def execute_group(group, slot, args, manifest, verifier, stop):
    with runner_lock(Path(args.out_root), f"group-{group['index']}"):
        if verifier.group_complete(group):
            return
        return execute_group_owned(group, slot, args, manifest, verifier, stop)


def execute_group_owned(group, slot, args, manifest, verifier, stop):
    host_id = args.host_id
    root = Path(args.out_root)
    assigned_gpu = args.gpu_list[slot % len(args.gpu_list)] if args.gpu_list else None
    host_meta = {"host_id": host_id, "host_os": platform.platform(), "direction": args.direction,
                 "gpu_index": assigned_gpu, "sc2_version": args.sc2_version}
    foreign_partial = any(
        (root / "arms" / arm_key(group, method) / "report.json").exists()
        and read_json(root / "arms" / arm_key(group, method) / "report.json").get("host_id") != host_id
        for method in group["methods"])
    if foreign_partial:
        backup = root / ".superseded" / f"group-{group['index']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        for method in group["methods"]:
            folder = root / "arms" / arm_key(group, method)
            if folder.exists():
                backup.mkdir(parents=True, exist_ok=True)
                folder.replace(backup / method)
    for method in group["methods"]:
        if stop.is_set():
            return
        key = arm_key(group, method)
        destination = root / "arms" / key
        if verifier.valid(group, method):
            continue
        require(not (destination / "done.json").exists(), f"Invalid or partially transferred done marker at {destination}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        work = root / ".work" / host_id / key / stamp
        work.mkdir(parents=True)
        logs = root / "driver_logs"
        logs.mkdir(exist_ok=True)
        log_path = logs / f"{host_id}-{key.replace('/', '-')}-{stamp}.log"
        payload = {
            "batch_id": manifest["batch_id"], "arm_key": key, "track": group["track"],
            "scenario": group["scenario"], "regime": group["regime"], "method": method, "seed": group["seed"],
            "host_id": host_id, "direction": args.direction, "gpu": assigned_gpu, "os": host_meta["host_os"],
            "sc2_version": args.sc2_version, "target_steps": group["train_steps"], "started_at": iso_now(),
            "work_dir": str(work), "log_file": str(log_path),
        }

        def report_status(state, **values):
            payload.update(state=state, updated_at=iso_now(), **values)
            write_status(destination, payload)
            post_wnc_report(args.wnc_url, payload)

        command = [sys.executable, "-m", "bch.v10_suite", "arm-worker", "--plan", str(args.plan),
                   "--scenario", group["scenario"], "--regime", group["regime"], "--method", method,
                   "--seed", str(group["seed"]), "--train-steps", str(group["train_steps"]),
                   "--eval-every", str(group["eval_every"]), "--val-repeats", str(manifest["val_repeats"]),
                   "--test-repeats", str(manifest["test_repeats"]), "--test-layouts", "standard,near",
                   "--step-mul", "8", "--eval-init", "--output", str(work)]
        if method != "im-hash":
            command.extend(["--encoder", str(Path(args.encoders_root) / group["scenario"] / method / "encoder.pt")])
        environment = _worker_env(f"v11_{host_id}_{slot}", rollout_device="cuda" if assigned_gpu is not None else "cpu")
        environment["PYTHONUNBUFFERED"] = "1"
        if assigned_gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        started = perf_counter()
        process = None
        children = []
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name != "nt",
                                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                while True:
                    try:
                        children = psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        pass
                    elapsed = perf_counter() - started
                    steps = read_training_steps(work / "train_episodes.jsonl")
                    report_status("running", pid=process.pid, steps=steps, wall_seconds=round(elapsed, 2),
                                  steps_per_second=round(steps / max(elapsed, 0.001), 3), error="")
                    try:
                        returncode = process.wait(timeout=args.heartbeat_seconds)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            require(returncode == 0, f"Worker exited {returncode}; full log {log_path}")
            report = publish_result(work, destination, group, method, manifest, host_meta)
            report_status("completed", pid=None, steps=group["train_steps"],
                          wall_seconds=round(perf_counter() - started, 2),
                          all9_win_rate=round(report["test"]["all9_win_rate"] * 100, 2), error="")
            print(f"[arm] {host_id} {key} completed in {report['wall_seconds']:.1f}s", flush=True)
        except Exception as error:
            if process is not None:
                stop_owned_process(process, children)
            report_status("blocked", error=str(error), wall_seconds=round(perf_counter() - started, 2))
            print(f"[blocked] {host_id} {key}: {error}", flush=True)


def cmd_run(args):
    root = Path(args.out_root)
    manifest = read_json(root / "v11_queue_manifest.json")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", args.host_id), "Invalid host ID")
    require(args.workers > 0 and args.heartbeat_seconds > 0 and args.poll_seconds > 0, "Workers and intervals must be positive")
    document, _, _, _ = read_plan(args.plan)
    require(plan_digest(document) == manifest["plan_sha256"], "Run plan differs from initialized manifest")
    for scenario in SCENARIO_GROUPS["all13"]:
        for method in CORE_METHODS[1:]:
            path = Path(args.encoders_root) / scenario / method / "encoder.pt"
            require(digest_file(path) == manifest["encoder_sha256"][f"{scenario}/{method}"], f"Wrong encoder {path}")
    args.gpu_list = [int(gpu) for gpu in args.gpus.split(",") if gpu.strip()]
    require(len(args.gpu_list) == len(set(args.gpu_list)), "Duplicate GPU index")
    verifier = CompletionVerifier(root, manifest)
    state_dir = root / "runners"
    state_dir.mkdir(exist_ok=True)

    def runner_status(state, track, **values):
        write_json(state_dir / f"{args.host_id}.json", {
            "host_id": args.host_id, "pid": os.getpid(), "state": state, "track": track,
            "direction": args.direction, "workers": args.workers, "updated_at": iso_now(), **values})

    with runner_lock(root, args.host_id):
        for gpu in args.gpu_list:
            release_gpu_hold_if_present(args.gpu_hold_script, gpu, args.gpu_hold_state_dir)
        print(f"[v11 run] host={args.host_id} direction={args.direction} workers={args.workers} gpus={args.gpu_list or 'cpu'}", flush=True)
        tracks = ("v11-A1", "v11-B1") if args.track in ("all", "v11-B1") else ("v11-A1",)
        stop = Event()
        try:
            for track in tracks:
                groups = stage_groups(manifest, track, args.host_id, args.direction)
                while not verifier.stage_complete(track):
                    runner_status("running", track)
                    remaining = [group for group in groups if not verifier.group_complete(group)]
                    interrupted = False
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        pending = {pool.submit(execute_group, group, slot, args, manifest, verifier, stop): group
                                   for slot, group in enumerate(remaining)}
                        try:
                            for future in as_completed(pending):
                                try:
                                    future.result()
                                except CancelledError:
                                    continue
                                except Exception as error:
                                    print(f"[group retry] {pending[future]['group_id']}: {error}", flush=True)
                        except KeyboardInterrupt:
                            stop.set()
                            interrupted = True
                            for waiting in pending:
                                waiting.cancel()
                            print("[v11 stop] Finishing active arms; queued groups cancelled", flush=True)
                    if interrupted:
                        raise KeyboardInterrupt
                    aggregate_results(root)
                    if not verifier.stage_complete(track):
                        runner_status("running", track, reason="Retrying remaining groups locally")
                        stop.wait(args.poll_seconds)
                aggregate_results(root)
                runner_status("completed", track, scope="all groups")
                print(f"[stage] {args.host_id} {track} all groups complete", flush=True)
        except KeyboardInterrupt:
            stop.set()
            runner_status("blocked", track, error="Interrupted; active arms drained and queued groups cancelled")
            raise SystemExit(130)
        except Exception as error:
            runner_status("blocked", track, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    initialize = commands.add_parser("init", help="Create the shared ordered paired-group manifest")
    initialize.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    initialize.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    initialize.add_argument("--encoders-root", type=Path, default=DEFAULT_ENCODERS)
    initialize.add_argument("--methods", default=",".join(CORE_METHODS))
    initialize.add_argument("--seeds", default=",".join(map(str, V11_SEEDS)))
    initialize.add_argument("--train-steps", type=int, default=V11_DEFAULT_STEPS)
    initialize.add_argument("--eval-every", type=int, default=V11_DEFAULT_EVAL_EVERY)
    initialize.add_argument("--val-repeats", type=int, default=2)
    initialize.add_argument("--test-repeats", type=int, default=5)
    initialize.add_argument("--a1-only", action="store_true")
    initialize.set_defaults(func=cmd_init)
    run = commands.add_parser("run", help="Traverse all groups, completing A1 before B1")
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    run.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--encoders-root", type=Path, default=DEFAULT_ENCODERS)
    run.add_argument("--track", choices=("all", "v11-A1", "v11-B1"), default="all")
    run.add_argument("--direction", choices=("forward", "backward"), default="forward")
    run.add_argument("--workers", type=int, default=6)
    run.add_argument("--gpus", default="")
    run.add_argument("--gpu-hold-script", default="")
    run.add_argument("--gpu-hold-state-dir", default="")
    run.add_argument("--host-id", required=True)
    run.add_argument("--sc2-version", required=True, help="SC2 version confirmed by the preflight engine check")
    run.add_argument("--wnc-url", default="http://127.0.0.1:4321/studio/experiment-console.json")
    run.add_argument("--heartbeat-seconds", type=float, default=10)
    run.add_argument("--poll-seconds", type=float, default=15)
    run.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
