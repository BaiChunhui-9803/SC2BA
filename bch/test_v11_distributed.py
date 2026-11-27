"""Offline regressions for v11 dispatch and result publication; no SC2 launch."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from threading import Event

from . import v11_distributed as runner


class DistributedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="sc2ba_v11_regression_")
        self.root = Path(self.temp.name)
        self.hosts = {scenario: "local-win" for scenario in runner.SCENARIO_GROUPS["all13"]}
        self.hosts["5z_vs_3s"] = "gpu-server"
        self.manifest = runner.build_v11_manifest(scenario_hosts=self.hosts)
        self.manifest["plan_sha256"] = "plan-fingerprint"
        self.manifest["encoder_sha256"] = {
            f"{scenario}/{method}": "encoder-fingerprint"
            for scenario in runner.SCENARIO_GROUPS["all13"] for method in runner.CORE_METHODS[1:]
        }
        self.group = self.manifest["groups"][0]

    def tearDown(self):
        self.temp.cleanup()

    def work_result(self, method="im-hash"):
        work = self.root / "work"
        work.mkdir(exist_ok=True)
        report = {
            "scenario": self.group["scenario"], "regime": self.group["regime"], "method": method,
            "seed": self.group["seed"], "actual_train_steps": 40000, "train_step_budget": 40000,
            "checkpoint_protocol": "exact_budget_frozen_replay_v1", "plan_sha256": "plan-fingerprint",
            "encoder_sha256": None if method == "im-hash" else "encoder-fingerprint",
            "train_opponents": list(runner.ALL_OPPONENTS), "eval_opponents": list(runner.ALL_OPPONENTS),
            "upper_q_rows": 32, "wall_seconds": 10,
            "test": {"upper_unseen_rate": 0, "all9_win_rate": 0.5, "episodes": 90,
                     "algos8_win_rate": 0.5, "encode_ms_p50": 1.0},
        }
        runner.write_json(work / "report.json", report)
        runner.write_json(work / "policy.json", {"q": {}})
        runner.write_json(work / "learning_curve.json", {"checkpoints": [
            {"checkpoint_steps": step, "actual_train_steps": step, "episodes": 18}
            for step in range(0, 40001, 8000)]})
        (work / "train_episodes.jsonl").write_text('{"cum_steps":40000}\n', encoding="utf-8")
        validation = [{"checkpoint_steps": step, "layout": "diagonal", "opponent": opponent, "repeat": repeat}
                      for step in range(0, 40001, 8000) for opponent in runner.ALL_OPPONENTS for repeat in range(2)]
        evaluation = [{"checkpoint_steps": 40000, "layout": layout, "opponent": opponent, "repeat": repeat}
                      for layout in ("standard", "near") for opponent in runner.ALL_OPPONENTS for repeat in range(5)]
        (work / "val_episodes.jsonl").write_text("\n".join(map(json.dumps, validation)) + "\n", encoding="utf-8")
        (work / "test_episodes.jsonl").write_text("\n".join(map(json.dumps, evaluation)) + "\n", encoding="utf-8")
        # The v10 worker itself creates this marker. It stays private until v11 commits.
        runner.write_json(work / "done.json", {"status": "done"})
        return work

    def test_opposite_directions_respect_stage_and_scenario_owner(self):
        self.assertEqual((self.manifest["total_groups"], self.manifest["total_arms"]), (370, 1480))
        a1 = [group for group in self.manifest["groups"] if group["track"] == "v11-A1"]
        b1 = [group for group in self.manifest["groups"] if group["track"] == "v11-B1"]
        self.assertEqual((len(a1) * 4, len(b1) * 4), (520, 960))
        backward = runner.stage_groups(self.manifest, "v11-A1", "gpu-server", "backward")
        self.assertEqual(backward[0]["index"], 129)
        self.assertTrue(all(group["track"] == "v11-A1" for group in backward))
        local = runner.stage_groups(self.manifest, "v11-A1", "local-win", "forward")
        self.assertFalse(set(group["index"] for group in local) & set(group["index"] for group in backward))
        self.assertEqual(len(local) + len(backward), 130)
        self.assertEqual(runner.stage_groups(self.manifest, "v11-B1", "gpu-server", "backward"), [])

    def test_done_is_published_after_all_verified_files(self):
        work = self.work_result()
        destination = self.root / "arms" / runner.arm_key(self.group, "im-hash")
        original = runner.write_json
        seen = []

        def observe(path, value):
            if path == destination / "done.json":
                self.assertTrue(all((destination / name).is_file() for name in runner.RESULT_FILES))
                self.assertFalse(path.exists())
                self.assertEqual(value["host_id"], "local-win")
                seen.append(path)
            return original(path, value)

        with patch.object(runner, "write_json", side_effect=observe):
            runner.publish_result(work, destination, self.group, "im-hash", self.manifest,
                                  {"host_id": "local-win", "host_os": "Windows", "sc2_version": "5.0.16"})
        self.assertEqual(len(seen), 1)
        verifier = runner.CompletionVerifier(self.root, self.manifest)
        self.assertTrue(verifier.valid(self.group, "im-hash"))
        self.assertFalse(verifier.stage_complete("v11-A1"))
        (destination / "policy.json").write_text("incomplete transfer", encoding="utf-8")
        self.assertFalse(verifier.valid(self.group, "im-hash"))

    def test_wrong_budget_never_creates_public_done(self):
        work = self.work_result()
        report = runner.read_json(work / "report.json")
        report["actual_train_steps"] = 8000
        runner.write_json(work / "report.json", report)
        destination = self.root / "arms" / runner.arm_key(self.group, "im-hash")
        with self.assertRaisesRegex(RuntimeError, "Wrong step budget"):
            runner.publish_result(work, destination, self.group, "im-hash", self.manifest, {"host_id": "local-win"})
        self.assertFalse((destination / "done.json").exists())

    def test_foreign_host_result_cannot_satisfy_barrier(self):
        work = self.work_result()
        destination = self.root / "arms" / runner.arm_key(self.group, "im-hash")
        with self.assertRaisesRegex(RuntimeError, "another scenario owner"):
            runner.publish_result(work, destination, self.group, "im-hash", self.manifest, {"host_id": "gpu-server"})
        self.assertFalse((destination / "done.json").exists())
        self.assertFalse(runner.CompletionVerifier(self.root, self.manifest).valid(self.group, "im-hash"))

    def test_heartbeat_reads_last_complete_json_line(self):
        path = self.root / "train_episodes.jsonl"
        self.assertEqual(runner.read_training_steps(path), 0)
        path.write_bytes(b'{"cum_steps":1234}\n{"cum_steps":')
        self.assertEqual(runner.read_training_steps(path), 1234)

    def test_foreign_later_method_blocks_entire_group_before_launch(self):
        destination = self.root / "arms" / runner.arm_key(self.group, "im-grid")
        runner.write_status(destination, {"host_id": "gpu-server", "state": "running"})
        args = argparse.Namespace(out_root=self.root, host_id="local-win", gpu_list=[],
                                  direction="forward", sc2_version="test")
        verifier = runner.CompletionVerifier(self.root, self.manifest)
        with patch.object(runner.platform, "platform", return_value="Windows"), \
             patch.object(runner.subprocess, "Popen") as process:
            with self.assertRaisesRegex(RuntimeError, "cross-host partial group"):
                runner.execute_group(self.group, 0, args, self.manifest, verifier, Event())
            process.assert_not_called()

    def test_gpu_release_failure_and_foreign_gpu_block_launch(self):
        script = self.root / "gpu_hold.py"
        script.touch()
        failed = argparse.Namespace(returncode=1, stdout="", stderr="ownership mismatch")
        with patch.object(runner.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "hold release failed"):
                runner.release_gpu_hold_if_present(script, 0)
        release = argparse.Namespace(returncode=0, stdout='{"ok":true}', stderr="")
        occupied = argparse.Namespace(returncode=0, stdout=json.dumps({"gpus": [{"index": 0, "idle": False}]}))
        with patch.object(runner.subprocess, "run", side_effect=[release, occupied]):
            with self.assertRaisesRegex(RuntimeError, "occupied after hold release"):
                runner.release_gpu_hold_if_present(script, 0)

    def test_init_is_idempotent_and_rejects_changed_configuration(self):
        assignment = self.root / "hosts.json"
        runner.write_json(assignment, self.hosts)
        args = argparse.Namespace(out_root=self.root / "output", scenario_hosts=assignment,
                                  methods=",".join(runner.CORE_METHODS), seeds=",".join(map(str, runner.V11_SEEDS)),
                                  train_steps=40000, eval_every=8000, a1_only=False, val_repeats=2, test_repeats=5,
                                  plan=self.root / "plan.json", encoders_root=self.root / "encoders")
        with patch.object(runner, "read_plan", return_value=({}, {}, {}, {})), \
             patch.object(runner, "plan_digest", return_value="plan-fingerprint"), \
             patch.object(runner, "digest_file", return_value="encoder-fingerprint"):
            runner.cmd_init(args)
            path = args.out_root / "v11_queue_manifest.json"
            content = path.read_bytes()
            runner.cmd_init(args)
            self.assertEqual(content, path.read_bytes())
            args.train_steps = 8000
            with self.assertRaisesRegex(RuntimeError, "Existing manifest differs"):
                runner.cmd_init(args)
            self.assertEqual(content, path.read_bytes())

    def test_b1_waits_for_complete_a1_before_dispatch(self):
        runner.write_json(self.root / "v11_queue_manifest.json", self.manifest)
        args = argparse.Namespace(out_root=self.root, encoders_root=self.root, plan=self.root / "plan.json",
                                  host_id="local-win", direction="backward", track="v11-B1", workers=2,
                                  heartbeat_seconds=1, poll_seconds=0.001, gpus="", gpu_hold_script="",
                                  sc2_version="test", wnc_url="")
        order = []
        barrier = iter([False, True])

        def ready(track):
            value = next(barrier)
            order.append(("barrier", value))
            return value

        def execute(group, *args):
            order.append(("group", group["track"]))

        with patch.object(runner, "read_plan", return_value=({}, {}, {}, {})), \
             patch.object(runner, "plan_digest", return_value="plan-fingerprint"), \
             patch.object(runner, "digest_file", return_value="encoder-fingerprint"), \
             patch.object(runner, "runner_lock", return_value=nullcontext()), \
             patch.object(runner.CompletionVerifier, "stage_complete", side_effect=ready), \
             patch.object(runner.CompletionVerifier, "valid", return_value=True), \
             patch.object(runner, "execute_group", side_effect=execute), \
             patch.object(runner, "aggregate_results"):
            runner.cmd_run(args)
        self.assertEqual(order[:2], [("barrier", False), ("barrier", True)])
        self.assertTrue(all(item == ("group", "v11-B1") for item in order[2:]))

    def test_interrupt_sets_stop_drains_executor_and_returns_130(self):
        runner.write_json(self.root / "v11_queue_manifest.json", self.manifest)
        args = argparse.Namespace(out_root=self.root, encoders_root=self.root, plan=self.root / "plan.json",
                                  host_id="local-win", direction="forward", track="v11-A1", workers=2,
                                  heartbeat_seconds=1, poll_seconds=0.001, gpus="", gpu_hold_script="",
                                  sc2_version="test", wnc_url="")
        stops = []

        def execute(group, slot, config, manifest, verifier, stop):
            stops.append(stop)
            stop.wait(0.2)

        with patch.object(runner, "read_plan", return_value=({}, {}, {}, {})), \
             patch.object(runner, "plan_digest", return_value="plan-fingerprint"), \
             patch.object(runner, "digest_file", return_value="encoder-fingerprint"), \
             patch.object(runner, "runner_lock", return_value=nullcontext()), \
             patch.object(runner, "execute_group", side_effect=execute), \
             patch.object(runner, "as_completed", side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as result:
                runner.cmd_run(args)
        self.assertEqual(result.exception.code, 130)
        self.assertTrue(stops and all(stop.is_set() for stop in stops))
        status = runner.read_json(self.root / "runners" / "local-win.json")
        self.assertEqual(status["state"], "blocked")
        self.assertIn("Interrupted", status["error"])


if __name__ == "__main__":
    unittest.main()
