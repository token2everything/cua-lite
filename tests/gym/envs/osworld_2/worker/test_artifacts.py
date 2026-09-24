"""Concurrent local run writers must keep every result without leaking credentials."""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from lite.gym.envs.osworld_2 import artifacts
from lite.gym.envs.osworld_2.artifacts import Experiment


def _process_writer(root, number, barrier):
    experiment = Experiment("multiprocess", root, existing=True)
    counter = experiment.path / "counter.txt"
    barrier.wait(timeout=10)
    for _ in range(5):
        with experiment.lock():
            value = int(counter.read_text())
            time.sleep(0.005)  # Competing processes must not read the same value.
            counter.write_text(str(value + 1))
    run = experiment.new_run("054" if number % 2 else "004", run_id=f"run-{number}")
    run.finish("completed", score=number / 10)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_parallel_writers_and_repeated_tasks_keep_all_runs_and_scores(self):
        def worker(number):
            experiment = Experiment("parallel", self.root, {"kind": "smoke"})
            run = experiment.new_run("054" if number % 2 else "004")
            run.finish("completed", score=number / 10)
            return run.run_id

        with ThreadPoolExecutor(max_workers=6) as executor:
            run_ids = list(executor.map(worker, range(10)))
        summary = json.loads((self.root / "parallel/summary.json").read_text())
        self.assertEqual(len(set(run_ids)), 10)
        self.assertEqual(summary["run_count"], 10)
        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["counts"]["completed"], 10)
        self.assertAlmostEqual(sum(run["score"] for run in summary["runs"]), 4.5)
        with (self.root / "parallel/summary.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 10)

    def test_multiprocess_lock_serializes_updates_and_keeps_all_results(self):
        experiment = Experiment("multiprocess", self.root)
        counter = experiment.path / "counter.txt"
        counter.write_text("0")
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(4)
        processes = [
            context.Process(target=_process_writer, args=(str(self.root), number, barrier))
            for number in range(4)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        self.assertEqual(counter.read_text(), "20")
        summary = json.loads((experiment.path / "summary.json").read_text())
        self.assertEqual(summary["run_count"], 4)
        self.assertEqual(summary["counts"]["completed"], 4)
        self.assertEqual({run["run_id"] for run in summary["runs"]},
                         {f"run-{number}" for number in range(4)})
        self.assertAlmostEqual(sum(run["score"] for run in summary["runs"]), 0.6)
        self.assertFalse((experiment.path / ".experiment.lock").exists())

    def test_lock_releases_after_exception_and_can_be_acquired_again(self):
        experiment = Experiment("release", self.root)
        lock_path = experiment.path / ".experiment.lock"
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            with experiment.lock():
                self.assertTrue(lock_path.is_dir())
                raise RuntimeError("writer failed")
        self.assertFalse(lock_path.exists())
        with experiment.lock(timeout=0):
            self.assertTrue(lock_path.is_dir())
        self.assertFalse(lock_path.exists())

    def test_lock_timeout_does_not_remove_another_owner_or_steal_stale_lock(self):
        experiment = Experiment("stale", self.root)
        lock_path = experiment.path / ".experiment.lock"
        lock_path.mkdir()
        owner = lock_path / "owner.txt"
        owner.write_text("another writer")
        with patch.object(artifacts.time, "sleep", side_effect=AssertionError("must not wait")):
            with self.assertRaisesRegex(TimeoutError, "Verify whether a writer is active"):
                with experiment.lock(timeout=0):
                    self.fail("A contender entered an already locked experiment")
        self.assertEqual(owner.read_text(), "another writer")

    def test_old_file_lock_is_rejected_without_modification(self):
        folder = self.root / "old-format"
        folder.mkdir()
        lock_path = folder / ".experiment.lock"
        lock_path.write_text("old file lock")
        with self.assertRaisesRegex(FileExistsError, "File-based locks are unsupported"):
            Experiment("old-format", self.root)
        self.assertEqual(lock_path.read_text(), "old file lock")
        self.assertFalse((folder / "experiment.json").exists())

    def test_progress_is_live_without_rewriting_aggregate_snapshots(self):
        experiment = Experiment("progress", self.root)
        run = experiment.new_run("054")
        summary_path = experiment.path / "summary.json"
        csv_path = experiment.path / "summary.csv"
        before = (summary_path.read_bytes(), csv_path.read_bytes())

        run.update(env_steps=2)
        self.assertEqual(json.loads((run.path / "run.json").read_text())["env_steps"], 2)
        self.assertEqual((summary_path.read_bytes(), csv_path.read_bytes()), before)

        refreshed = experiment.summarize()
        self.assertEqual(refreshed["runs"][0]["env_steps"], 2)
        run.update(env_steps=3)
        record = run.finish("completed", score=1.0)
        final = json.loads(summary_path.read_text())
        self.assertEqual(final["runs"], [record])
        self.assertEqual(record["env_steps"], 3)
        self.assertEqual(final["counts"]["completed"], 1)
        with csv_path.open() as stream:
            (row,) = csv.DictReader(stream)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["score"], "1.0")

    def test_summary_cli_uses_the_existing_experiment_release(self):
        experiment = Experiment("june", self.root, release="osworld-v2-2026.06.24")
        run = experiment.new_run("054")
        run.update(env_steps=2)
        manifest = (experiment.path / "experiment.json").read_bytes()
        output = io.StringIO()

        with patch("sys.argv", ["artifacts", "june", "--results-root", str(self.root)]):
            with redirect_stdout(output):
                artifacts.main()

        self.assertEqual(json.loads(output.getvalue())["run_count"], 1)
        self.assertEqual((experiment.path / "experiment.json").read_bytes(), manifest)
        (record,) = json.loads((experiment.path / "summary.json").read_text())["runs"]
        self.assertEqual(record["release"], "osworld-v2-2026.06.24")
        self.assertEqual(record["env_steps"], 2)

    def test_invalid_names_conflicting_config_and_unowned_directory_are_rejected(self):
        for name in ("../escape", "/absolute", "two/parts", ".", "a" * 129):
            with self.assertRaises(ValueError):
                Experiment(name, self.root)
        experiment = Experiment("existing", self.root, {"model": "first"})
        with self.assertRaises(ValueError):
            Experiment("existing", self.root, {"model": "different"})
        with self.assertRaises(ValueError):
            experiment.new_run("../054")
        folder = self.root / "unowned"
        folder.mkdir()
        (folder / "keep.txt").write_text("original")
        with self.assertRaises(FileExistsError):
            Experiment("unowned", self.root)
        self.assertEqual((folder / "keep.txt").read_text(), "original")

    def test_same_release_name_cannot_mix_different_manifests(self):
        experiment = Experiment("immutable", self.root)
        manifest = experiment.path / "experiment.json"
        saved = json.loads(manifest.read_text())
        saved["release_manifest_sha256"] = "different-release-content"
        manifest.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "release manifest differs"):
            Experiment("immutable", self.root)

    def test_observations_keep_redacted_metadata_without_copying_images(self):
        experiment = Experiment("metadata-only", self.root)
        run = experiment.new_run("054", secrets=["private-observation"])
        stream = io.BytesIO()
        Image.new("RGB", (2, 3), "red").save(stream, format="PNG")
        screenshot = stream.getvalue()
        observation = {
            "run_id": run.run_id,
            "observation_id": "observation-1",
            "step": 1,
            "instruction": "Open private-observation.",
            "screenshot": screenshot,
            "screenshot_encoding": "base64/png",
        }

        run.observation(observation)
        run.event("close", status="succeeded")

        events_text = (run.path / "events.jsonl").read_text()
        events = [json.loads(line) for line in events_text.splitlines()]
        self.assertEqual([event["method"] for event in events], ["observation", "close"])
        self.assertEqual(
            events[0]["observation"],
            {
                "run_id": run.run_id,
                "observation_id": "observation-1",
                "step": 1,
                "instruction": "Open [redacted].",
            },
        )
        self.assertIs(observation["screenshot"], screenshot)
        self.assertEqual(observation["instruction"], "Open private-observation.")
        self.assertNotIn("private-observation", events_text)
        self.assertEqual({path.name for path in run.path.iterdir()}, {"run.json", "events.jsonl"})

    def test_invalid_screenshots_do_not_publish_observation_events(self):
        run = Experiment("invalid-image", self.root).new_run("054")
        for screenshot in (None, "base64-image", b"not an image", b"\xff\xd8\xff"):
            with self.subTest(screenshot=screenshot):
                with self.assertRaisesRegex(ValueError, "Expected PNG screenshot bytes"):
                    run.observation({"screenshot": screenshot, "step": 1})
        self.assertEqual({path.name for path in run.path.iterdir()}, {"run.json"})
