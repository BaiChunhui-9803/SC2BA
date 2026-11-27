"""Incrementally mirror verified v11 results and remote heartbeats through SSH.

Run locally with ``python -m bch.v11_sync --hosts gpu,cpu``. The default poll
interval is 25 seconds. ``--once`` performs one pass. An optional JSON config
contains a ``hosts`` mapping that overrides each server's target, port,
identity, repo, out_root, python, and host_id fields. Remote helper modes use
only the Python standard library and never import the training environment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request


DEFAULT_ROOT = Path(__file__).resolve().parent / "runs" / "v11_40k_10seed"
REMOTE_REPO = "/data/Share/Student/2020/BaiChunhui/SC2BA-Lab"
RESULT_FILES = ("report.json", "policy.json", "learning_curve.json",
                "train_episodes.jsonl", "val_episodes.jsonl", "test_episodes.jsonl")
TRANSFER_FILES = (*RESULT_FILES, "done.json")
DEFAULT_HOSTS = {
    "gpu": {"target": "baichunhui2202210396@10.33.101.35", "port": 23082,
            "identity": None, "host_id": "gpu-server", "repo": REMOTE_REPO,
            "out_root": REMOTE_REPO + "/bch/runs/v11_40k_10seed", "python": "python3"},
    "cpu": {"target": "baichunhui2202210396@10.33.101.42", "port": 23082,
            "identity": str(Path.home() / ".ssh" / "id_ed25519_wnc_cpu"),
            "host_id": "cpu-server", "repo": REMOTE_REPO,
            "out_root": REMOTE_REPO + "/bch/runs/v11_40k_10seed", "python": "python3"},
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(manifest):
    content = {key: value for key, value in manifest.items() if key != "created_at"}
    return sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def arm_map(manifest):
    require(manifest.get("format") == "bch-sc2ba-v11-manifest-v2", "A v2 manifest is required")
    result = {}
    for group in manifest["groups"]:
        for method in group["methods"]:
            key = f"{group['scenario']}/{group['regime']}/{method}/seed_{group['seed']}"
            require(re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/seed_[0-9]+", key),
                    "Unsafe arm path in manifest")
            require(key not in result, "Duplicate arm key")
            result[key] = (group, method)
    return result


def load_manifest(root, expected=None):
    manifest = read_json(root / "v11_queue_manifest.json")
    arm_map(manifest)
    if expected:
        require(manifest_digest(manifest) == expected, "Local and remote manifests differ")
    return manifest


def verify_bundle(folder, key, manifest, entries):
    require(key in entries, "Arm is absent from the shared manifest")
    group, method = entries[key]
    marker = read_json(folder / "done.json")
    owner = marker.get("host_id")
    require(marker.get("status") == "done" and marker.get("manifest_sha256") == manifest_digest(manifest),
            f"Invalid completion marker for {key}")
    require(isinstance(owner, str) and bool(owner), f"Missing completion host for {key}")
    hashes = marker.get("files_sha256", {})
    require(set(hashes) == set(RESULT_FILES), f"Incomplete fingerprint list for {key}")
    for name in RESULT_FILES:
        require(digest_file(folder / name) == hashes[name], f"Checksum mismatch for {key}/{name}")
    report = read_json(folder / "report.json")
    require(tuple(report.get(name) for name in ("scenario", "regime", "method", "seed")) ==
            (group["scenario"], group["regime"], method, group["seed"]), f"Wrong result identity for {key}")
    require(report.get("host_id") == owner and report.get("plan_sha256") == manifest["plan_sha256"],
            f"Wrong result provenance for {key}")
    require(report.get("actual_train_steps") == report.get("train_step_budget") == group["train_steps"],
            f"Wrong training budget for {key}")
    expected_encoder = None if method == "im-hash" else manifest["encoder_sha256"][f"{group['scenario']}/{method}"]
    require(report.get("encoder_sha256") == expected_encoder, f"Wrong frozen encoder for {key}")
    return report


def group_keys(key, entries):
    group, _ = entries[key]
    return [f"{group['scenario']}/{group['regime']}/{method}/seed_{group['seed']}"
            for method in group['methods']]


@contextmanager
def group_lock(root, index):
    folder = root / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"group-{index}.lock").open("a+b") as lock:
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


def group_priority(root, key, manifest, entries):
    markers = []
    for member in group_keys(key, entries):
        folder = root / "arms" / member
        verify_bundle(folder, member, manifest, entries)
        markers.append(read_json(folder / "done.json"))
    require(len({marker['host_id'] for marker in markers}) == 1, "A paired group must come from one host")
    return (max(marker['completed_at'] for marker in markers), markers[0]['host_id'])


def export_bundle(root, key, manifest, entries, stream):
    group_priority(root, key, manifest, entries)
    with tarfile.open(fileobj=stream, mode="w|gz") as archive:
        for member in group_keys(key, entries):
            method = entries[member][1]
            for name in TRANSFER_FILES:
                archive.add(str(root / "arms" / member / name), arcname=f"{method}/{name}", recursive=False)


def unpack_bundle(stream, stage_root, key, entries):
    keys = group_keys(key, entries)
    permitted = {f"{entries[member][1]}/{name}": (member, name)
                 for member in keys for name in TRANSFER_FILES}
    seen = set()
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            require(member.isfile() and member.name in permitted and member.name not in seen,
                    "Unsafe or duplicate result archive member")
            seen.add(member.name)
            arm, name = permitted[member.name]
            destination = stage_root / "arms" / arm / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    require(seen == set(permitted), "Paired group transfer is incomplete")


def publish_bundle(stage_root, root, key, manifest, entries, semantic_verifier=None):
    keys = group_keys(key, entries)
    incoming = group_priority(stage_root, key, manifest, entries)
    if semantic_verifier:
        verifier = semantic_verifier(stage_root, manifest)
        require(verifier.group_complete(entries[key][0]), "Full paired result validation failed")
    try:
        with group_lock(root, entries[key][0]['index']):
            try:
                current = group_priority(root, key, manifest, entries)
            except (OSError, ValueError, KeyError, RuntimeError):
                current = None
            if current is not None and current <= incoming:
                return False
            backup = root / ".superseded" / f"group-{entries[key][0]['index']}-{time.time_ns()}"
            for member in keys:
                target = root / "arms" / member
                if target.exists():
                    backup.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backup / entries[member][1])
                target.mkdir(parents=True, exist_ok=True)
                staged = stage_root / "arms" / member
                for name in RESULT_FILES:
                    os.replace(staged / name, target / name)
            for member in keys:
                atomic_bytes(root / "arms" / member / "done.json",
                             (stage_root / "arms" / member / "done.json").read_bytes())
            return True
    except (BlockingIOError, PermissionError):
        return False


def remote_snapshot(root, manifest, entries, host_id):
    done, priorities, statuses = {}, {}, []
    for key, (group, method) in entries.items():
        folder = root / "arms" / key
        if method == group['methods'][0]:
            try:
                priority = group_priority(root, key, manifest, entries)
                priorities[key] = priority
                for member in group_keys(key, entries):
                    done[member] = digest_file(root / "arms" / member / "done.json")
            except (OSError, ValueError, KeyError, RuntimeError):
                pass
        if (folder / "status.json").is_file():
            status = read_json(folder / "status.json")
            if status.get('host_id') == host_id:
                if status.get('state') == 'completed' and (folder / 'done.json').exists():
                    verify_bundle(folder, key, manifest, entries)
                    statuses.append({**status, 'verified_remote_completion': True})
                elif status.get('state') in ('running', 'blocked') and not (folder / 'done.json').exists():
                    statuses.append(status)
    return {"manifest_sha256": manifest_digest(manifest), "host_id": host_id,
            "observed_at": now(), "done": done, "priorities": priorities, "statuses": statuses}


def remote_main(args):
    root = args.out_root
    manifest = load_manifest(root, args.manifest_sha256)
    entries = arm_map(manifest)
    if args.remote_action == "snapshot":
        print(json.dumps(remote_snapshot(root, manifest, entries, args.host_id)))
    elif args.remote_action == "export":
        export_bundle(root, args.arm_key, manifest, entries, sys.stdout.buffer)
    else:
        require(args.arm_key in entries, "Unknown imported group")
        stage_dir = root / ".sync-incoming"
        stage_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="transfer-", dir=str(stage_dir)) as temporary:
            stage_root = Path(temporary)
            unpack_bundle(sys.stdin.buffer, stage_root, args.arm_key, entries)
            accepted = publish_bundle(stage_root, root, args.arm_key, manifest, entries)
        print(json.dumps({"ok": True, "accepted": accepted, "arm_key": args.arm_key}))


def load_hosts(names, config_path=None):
    overrides = read_json(config_path).get("hosts", {}) if config_path else {}
    hosts = []
    for name in names.split(","):
        name = name.strip()
        require(name in DEFAULT_HOSTS or name in overrides, f"Unknown SSH host {name}")
        host = {**DEFAULT_HOSTS.get(name, {}), **overrides.get(name, {}), "name": name}
        require(re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+", host["target"]), "Unsafe SSH target")
        require(isinstance(host["port"], int) and 1 <= host["port"] <= 65535, "Invalid SSH port")
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", host["host_id"]), "Invalid host ID")
        for key in ("repo", "out_root"):
            path = PurePosixPath(host[key])
            require(path.is_absolute() and ".." not in path.parts and not any(ord(c) < 32 for c in host[key]),
                    f"Unsafe remote {key}")
        require(isinstance(host["python"], str) and bool(host["python"]), "Missing remote Python")
        hosts.append(host)
    require(len({host["host_id"] for host in hosts}) == len(hosts), "Duplicate remote hosts")
    return hosts


class SshRemote:
    def __init__(self, host, manifest_hash):
        self.host = host
        self.manifest_hash = manifest_hash
        self.deployed = False

    def command(self, arguments):
        host = self.host
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                   "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
                   "-p", str(host["port"])]
        if host.get("identity"):
            command.extend(["-o", "IdentitiesOnly=yes", "-i", host["identity"]])
        return [*command, "--", host["target"], " ".join(shlex.quote(str(arg)) for arg in arguments)]

    def call(self, arguments, *, input_bytes=None, source=None, destination=None, timeout=90):
        require(not (input_bytes is not None and source is not None), "Choose one SSH input stream")
        result = subprocess.run(self.command(arguments), input=input_bytes, stdin=source,
                                stdout=destination or subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        require(result.returncode == 0, f"{self.host['name']} SSH failed ({result.returncode}): "
                + result.stderr.decode("utf-8", errors="replace")[-700:])
        return result.stdout

    def helper_args(self, action, key=None):
        args = [self.host["python"], str(PurePosixPath(self.host["repo"]) / "bch" / "v11_sync.py"),
                "--remote-action", action, "--out-root", self.host["out_root"],
                "--host-id", self.host["host_id"], "--manifest-sha256", self.manifest_hash]
        if key:
            args.extend(["--arm-key", key])
        return args

    def deploy(self):
        if self.deployed:
            return
        program = ("import os,sys,tempfile; from pathlib import Path; p=Path(sys.argv[1]); "
                   "p.parent.mkdir(parents=True,exist_ok=True); data=sys.stdin.buffer.read(); "
                   "fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=str(p.parent)); "
                   "f=os.fdopen(fd,'wb'); f.write(data); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(tmp,p)")
        self.call([self.host["python"], "-c", program,
                   str(PurePosixPath(self.host["repo"]) / "bch" / "v11_sync.py")],
                  input_bytes=Path(__file__).read_bytes(), timeout=30)
        self.deployed = True

    def snapshot(self):
        self.deploy()
        return json.loads(self.call(self.helper_args("snapshot"), timeout=45))

    def download(self, key, destination):
        self.call(self.helper_args("export", key), destination=destination)

    def upload(self, key, source):
        return json.loads(self.call(self.helper_args("import", key), source=source))


def completed_status(key, group, report, manifest):
    return {"batch_id": manifest["batch_id"], "arm_key": key, "track": group["track"],
            "scenario": group["scenario"], "regime": group["regime"], "method": report["method"],
            "seed": group["seed"], "state": "completed", "host_id": report["host_id"],
            "direction": report.get("direction", "backward"), "gpu": report.get("gpu_index"),
            "steps": group["train_steps"], "target_steps": group["train_steps"],
            "all9_win_rate": round(report["test"]["all9_win_rate"] * 100, 2),
            "wall_seconds": report["wall_seconds"], "updated_at": now(), "error": ""}


class Synchronizer:
    def __init__(self, root, manifest, remotes, wnc_url, stale_seconds=120):
        from .v11_distributed import CompletionVerifier, RESULT_FILES as runner_files
        require(RESULT_FILES == runner_files, "Synchronization result list differs from the runner")
        self.root, self.manifest, self.remotes = root, manifest, remotes
        self.entries = arm_map(manifest)
        self.verifier_type = CompletionVerifier
        self.verifier = CompletionVerifier(root, manifest)
        self.wnc_url, self.stale_seconds = wnc_url, stale_seconds
        self.web_error = ""

    def post(self, status):
        if not self.wnc_url:
            return
        request = urllib.request.Request(self.wnc_url,
            data=json.dumps({"action": "distributed-report", **status}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:4321"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                require(response.status == 200, "White Night City rejected a status update")
        except (OSError, RuntimeError) as error:
            self.web_error = str(error)

    def accept_status(self, status, host_id):
        key = status.get("arm_key")
        require(key in self.entries, "Remote heartbeat names an unknown arm")
        group, _ = self.entries[key]
        require(status.get("host_id") == host_id, f"Wrong heartbeat host for {key}")
        require(status.get("state") in ("running", "blocked", "completed"), "Unknown remote arm state")
        path = self.root / "arms" / key / "status.json"
        if path.exists():
            current = read_json(path)
            if current.get("host_id") == "local-win" and current.get("state") == "running":
                return
            if (current.get('host_id') == host_id and current.get('state') == 'completed'
                    and current.get('updated_at') == status.get('updated_at')):
                return
        if (path.parent / "done.json").exists():
            return
        value = {**status, "batch_id": self.manifest["batch_id"]}
        if value["state"] == "running":
            timestamp = datetime.fromisoformat(value["updated_at"].replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - timestamp).total_seconds() > self.stale_seconds:
                value.update(state="blocked", error="Remote heartbeat expired", remote_updated_at=value["updated_at"], updated_at=now())
        require(value["state"] != "completed" or value.get('verified_remote_completion') is True,
                "Completion needs verified remote result files")
        atomic_json(path, value)
        self.post(value)

    def block_active(self, host_id, error):
        for key, (group, _) in self.entries.items():
            path = self.root / "arms" / key / "status.json"
            if not path.is_file() or (path.parent / "done.json").exists():
                continue
            status = read_json(path)
            if status.get("host_id") == host_id and status.get("state") == "running":
                status.update(state="blocked", error=error, remote_updated_at=status.get("updated_at"), updated_at=now())
                atomic_json(path, status)
                self.post(status)

    def receive(self, remote, key):
        stage_dir = self.root / ".sync-incoming"
        stage_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=str(stage_dir)) as temporary:
            stage_root = Path(temporary)
            archive = stage_root / "result.tar.gz"
            with archive.open("wb") as output:
                remote.download(key, output)
            with archive.open("rb") as source:
                unpack_bundle(source, stage_root, key, self.entries)
            accepted = publish_bundle(stage_root, self.root, key, self.manifest, self.entries, self.verifier_type)
        if accepted:
            for member in group_keys(key, self.entries):
                report = read_json(self.root / "arms" / member / "report.json")
                status = completed_status(member, self.entries[member][0], report, self.manifest)
                atomic_json(self.root / "arms" / member / "status.json", status)
                self.post(status)
        return accepted

    def send(self, remote, key):
        with tempfile.TemporaryFile() as archive:
            export_bundle(self.root, key, self.manifest, self.entries, archive)
            archive.seek(0)
            response = remote.upload(key, archive)
        require(response.get("ok") is True, f"Remote publication failed for {key}")
        return response.get("accepted", False)

    def cycle(self):
        self.web_error = ""
        snapshots, summary = {}, {}
        for remote in self.remotes:
            host = remote.host
            item = {"host_id": host["host_id"], "downloaded": 0, "uploaded": 0, "state": "running", "error": ""}
            summary[host["name"]] = item
            try:
                snapshot = remote.snapshot()
                require(snapshot["manifest_sha256"] == manifest_digest(self.manifest) and snapshot["host_id"] == host["host_id"],
                        "Remote snapshot identity differs")
                snapshots[host["name"]] = snapshot
                for status in snapshot["statuses"]:
                    self.accept_status(status, host["host_id"])
                for key, remote_priority in snapshot['priorities'].items():
                    group, method = self.entries[key]
                    if self.verifier.group_complete(group):
                        local_priority = group_priority(self.root, key, self.manifest, self.entries)
                        if local_priority <= tuple(remote_priority):
                            continue
                    if self.receive(remote, key):
                        item['downloaded'] += len(group['methods'])
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError,
                    subprocess.SubprocessError, tarfile.TarError) as error:
                item.update(state="blocked", error=str(error))
                snapshots.pop(host["name"], None)
                self.block_active(host["host_id"], f"SSH synchronization unavailable: {error}")

        completed, priorities = {}, {}
        for key, (group, method) in self.entries.items():
            if method == group['methods'][0] and self.verifier.group_complete(group):
                priorities[key] = group_priority(self.root, key, self.manifest, self.entries)
                for member in group_keys(key, self.entries):
                    completed[member] = digest_file(self.root / 'arms' / member / 'done.json')
        for remote in self.remotes:
            name = remote.host['name']
            if name not in snapshots:
                continue
            try:
                inventory = snapshots[name]['priorities']
                for key, priority in priorities.items():
                    if key in inventory and tuple(inventory[key]) <= priority:
                        continue
                    if self.send(remote, key):
                        summary[name]['uploaded'] += len(self.entries[key][0]['methods'])
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError,
                    subprocess.SubprocessError, tarfile.TarError) as error:
                summary[name].update(state='blocked', error=str(error))
                self.block_active(remote.host['host_id'], f"SSH synchronization unavailable: {error}")

        if any(item["downloaded"] for item in summary.values()):
            from .v10_suite import aggregate_results
            aggregate_results(self.root)
        result = {"updated_at": now(), "pid": os.getpid(), "verified_completed": len(completed),
                  "hosts": summary, "wnc_error": self.web_error,
                  "state": "blocked" if self.web_error or any(item["state"] == "blocked" for item in summary.values()) else "running"}
        atomic_json(self.root / "sync" / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hosts", default="gpu,cpu")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--interval", type=float, default=25)
    parser.add_argument("--stale-seconds", type=float, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--wnc-url", default="http://127.0.0.1:4321/studio/experiment-console.json")
    parser.add_argument("--remote-action", choices=("snapshot", "export", "import"))
    parser.add_argument("--host-id")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--arm-key")
    args = parser.parse_args()
    if args.remote_action:
        remote_main(args)
        return
    from .v11_distributed import runner_lock
    require(args.interval > 0 and args.stale_seconds > 0, "Intervals must be positive")
    manifest = load_manifest(args.out_root)
    hosts = load_hosts(args.hosts, args.config)
    remotes = [SshRemote(host, manifest_digest(manifest)) for host in hosts]
    synchronizer = Synchronizer(args.out_root, manifest, remotes, args.wnc_url, args.stale_seconds)
    with runner_lock(args.out_root, "v11-sync"):
        while True:
            started = time.monotonic()
            result = synchronizer.cycle()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if args.once:
                if result["state"] == "blocked":
                    raise SystemExit(1)
                return
            time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
