"""Offline transfer, ownership, heartbeat, and restart regressions for v11 sync."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import v11_distributed as runner
from . import v11_sync as sync
from . import test_v11_distributed as runner_tests


class LocalRemote:
    """Exercise the real transfer format without using an SSH connection."""

    def __init__(self, name, root, manifest):
        self.host = {"name": name, "host_id": name + "-server"}
        self.root, self.manifest = root, manifest
        self.entries = sync.arm_map(manifest)
        self.downloads = 0
        self.uploads = 0
        self.disconnected = False

    def snapshot(self):
        if self.disconnected:
            raise OSError("simulated SSH disconnect")
        return sync.remote_snapshot(self.root, self.manifest, self.entries, self.host["host_id"])

    def download(self, key, stream):
        self.downloads += 1
        sync.export_bundle(self.root, key, self.manifest, self.entries, stream)

    def upload(self, key, stream):
        self.uploads += 1
        with TemporaryDirectory(dir=self.root) as temporary:
            stage = Path(temporary)
            sync.unpack_bundle(stream, stage, key)
            sync.publish_bundle(stage, self.root, key, self.manifest, self.entries)
        return {"ok": True, "done_sha256": sync.digest_file(self.root / "arms" / key / "done.json")}


class SynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runner_tests.DistributedRunnerTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.manifest = self.fixture.manifest
        self.manifest["scenario_hosts"]["3m"] = "gpu-server"
        self.manifest["scenario_hosts"]["5z_vs_3s"] = "cpu-server"
        self.entries = sync.arm_map(self.manifest)
        self.local = self.root / "local"
        self.gpu = self.root / "gpu"
        self.cpu = self.root / "cpu"
        for folder in (self.local, self.gpu, self.cpu):
            folder.mkdir()
            sync.atomic_json(folder / "v11_queue_manifest.json", self.manifest)
        self.group = self.manifest["groups"][0]
        self.key = runner.arm_key(self.group, "im-hash")

    def tearDown(self):
        self.fixture.tearDown()

    def result(self, destination, group=None):
        group = group or self.group
        self.fixture.group = group
        work = self.fixture.work_result()
        key = runner.arm_key(group, "im-hash")
        runner.publish_result(work, destination / "arms" / key, group, "im-hash", self.manifest,
                              {"host_id": self.manifest["scenario_hosts"][group["scenario"]],
                               "direction": "backward", "host_os": "fixture"})
        return key

    def synchronizer(self, remotes=()):
        return sync.Synchronizer(self.local, self.manifest, list(remotes), "")

    def status(self, key=None, host="gpu-server", updated=None):
        return {"arm_key": key or self.key, "host_id": host, "state": "running",
                "steps": 1200, "target_steps": 40000, "updated_at": updated or sync.now()}

    def test_transfer_is_fully_verified_before_done_is_published(self):
        self.result(self.gpu)
        original = sync.atomic_bytes
        publications = []

        def observe(path, content):
            if path == self.local / "arms" / self.key / "done.json":
                self.assertFalse(path.exists())
                self.assertTrue(all((path.parent / name).is_file() for name in sync.RESULT_FILES))
                publications.append(path)
            return original(path, content)

        remote = LocalRemote("gpu", self.gpu, self.manifest)
        with patch.object(sync, "atomic_bytes", side_effect=observe):
            self.synchronizer().receive(remote, self.key)
        self.assertEqual(len(publications), 1)
        self.assertTrue(runner.CompletionVerifier(self.local, self.manifest).valid(self.group, "im-hash"))

    def test_full_semantic_check_rejects_hash_consistent_bad_results(self):
        self.result(self.gpu)
        folder = self.gpu / "arms" / self.key
        (folder / "val_episodes.jsonl").write_text("{}\n", encoding="utf-8")
        marker = sync.read_json(folder / "done.json")
        marker["files_sha256"]["val_episodes.jsonl"] = sync.digest_file(folder / "val_episodes.jsonl")
        sync.atomic_json(folder / "done.json", marker)
        with self.assertRaisesRegex(RuntimeError, "Full result validation"):
            self.synchronizer().receive(LocalRemote("gpu", self.gpu, self.manifest), self.key)
        self.assertFalse((self.local / "arms" / self.key / "done.json").exists())

    def test_broadcast_is_incremental_across_restarts(self):
        self.result(self.gpu)
        local_group = next(g for g in self.manifest["groups"] if g["scenario"] == "8m")
        local_key = self.result(self.local, local_group)
        remotes = [LocalRemote("gpu", self.gpu, self.manifest), LocalRemote("cpu", self.cpu, self.manifest)]
        with patch("bch.v10_suite.aggregate_results"):
            first = self.synchronizer(remotes).cycle()
            counts = [(r.downloads, r.uploads) for r in remotes]
            second = self.synchronizer(remotes).cycle()
        self.assertEqual(first["verified_completed"], 2)
        self.assertEqual(first["state"], "running")
        self.assertEqual(second["verified_completed"], 2)
        self.assertEqual(counts, [(r.downloads, r.uploads) for r in remotes])
        for root in (self.local, self.gpu, self.cpu):
            verifier = runner.CompletionVerifier(root, self.manifest)
            self.assertTrue(verifier.valid(self.group, "im-hash"))
            self.assertTrue(verifier.valid(local_group, "im-hash"))
            self.assertEqual(sync.digest_file(root / "arms" / local_key / "done.json"),
                             sync.digest_file(self.local / "arms" / local_key / "done.json"))

    def test_disconnect_marks_only_owned_active_status_and_recovery_resumes(self):
        local_group = next(g for g in self.manifest["groups"] if g["scenario"] == "8m")
        local_key = runner.arm_key(local_group, "im-hash")
        local_status = self.status(local_key, "local-win")
        sync.atomic_json(self.local / "arms" / local_key / "status.json", local_status)
        remote_status = self.status()
        sync.atomic_json(self.gpu / "arms" / self.key / "status.json", remote_status)
        remote = LocalRemote("gpu", self.gpu, self.manifest)
        daemon = self.synchronizer([remote])
        daemon.cycle()
        remote.disconnected = True
        blocked = daemon.cycle()
        self.assertEqual(blocked["state"], "blocked")
        self.assertEqual(sync.read_json(self.local / "arms" / self.key / "status.json")["state"], "blocked")
        self.assertEqual(sync.read_json(self.local / "arms" / local_key / "status.json"), local_status)
        self.assertFalse((self.local / "arms" / self.key / "done.json").exists())
        remote.disconnected = False
        daemon.cycle()
        self.assertEqual(sync.read_json(self.local / "arms" / self.key / "status.json")["state"], "running")

    def test_wrong_owner_and_stale_heartbeat(self):
        daemon = self.synchronizer()
        with self.assertRaisesRegex(RuntimeError, "another scenario owner"):
            daemon.accept_status(self.status(host="cpu-server"), "cpu-server")
        daemon.accept_status(self.status(updated="2000-01-01T00:00:00+00:00"), "gpu-server")
        self.assertEqual(sync.read_json(self.local / "arms" / self.key / "status.json")["state"], "blocked")

    def test_archive_rejects_path_traversal(self):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as output:
            item = tarfile.TarInfo("../escaped")
            item.size = 1
            output.addfile(item, io.BytesIO(b"x"))
        archive.seek(0)
        with TemporaryDirectory(dir=self.root) as temporary:
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                sync.unpack_bundle(archive, Path(temporary), self.key)
        self.assertFalse((self.root / "escaped").exists())

    def test_interrupted_archive_is_retryable_and_never_publishes_done(self):
        self.result(self.gpu)
        remote = LocalRemote("gpu", self.gpu, self.manifest)
        daemon = self.synchronizer([remote])
        with patch.object(remote, "download", side_effect=lambda key, stream: stream.write(b"partial archive")):
            result = daemon.cycle()
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(result["verified_completed"], 0)
        self.assertFalse((self.local / "arms" / self.key / "done.json").exists())
        with patch("bch.v10_suite.aggregate_results"):
            result = daemon.cycle()
        self.assertEqual(result["state"], "running")
        self.assertEqual(result["verified_completed"], 1)

    def test_remote_helper_runs_without_site_packages(self):
        result = subprocess.run([sys.executable, "-I", "-S", str(Path(sync.__file__).resolve()),
                                 "--remote-action", "snapshot", "--out-root", str(self.gpu),
                                 "--host-id", "gpu-server", "--manifest-sha256", sync.manifest_digest(self.manifest)],
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["host_id"], "gpu-server")


if __name__ == "__main__":
    unittest.main()
