import argparse
import json
import tempfile
import unittest
import importlib.util
from unittest.mock import MagicMock, patch
from pathlib import Path

from PIL import Image

from baselines.metrics import METRIC_NAMES
from baselines.tracking import ExperimentHistory, WandbTracker


class TrackingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("wandb"), "W&B SDK not installed")
    def test_wandb_init_excludes_local_paths_and_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            history = ExperimentHistory(temporary, dict(model="UNet", dataset="HRC_WHU",
                selection_metric="mIoU", data_root="/private/dataset", work_dir="/private/run", cafbr_start_ratio=0.5))
            args = argparse.Namespace(wandb=True, wandb_project="Cloud-Adapter", wandb_entity=None,
                wandb_name="test", wandb_mode="offline")
            run = MagicMock(id="test1234")
            with patch("wandb.init", return_value=run) as initialize:
                tracker = WandbTracker(args, history)
            cloud_config = initialize.call_args.kwargs["config"]
            self.assertEqual(cloud_config["model"], "UNet")
            self.assertEqual(cloud_config["cafbr_start_ratio"], 0.5)
            self.assertNotIn("data_root", cloud_config)
            self.assertNotIn("work_dir", cloud_config)
            self.assertEqual(len(initialize.call_args.kwargs["id"]), 8)

    def test_histories_survive_resume_and_render_all_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = dict(model="UNet", dataset="HRC_WHU", selection_metric="mIoU")
            history = ExperimentHistory(temporary, config)
            run_id = history.data["run_id"]
            metrics = dict(loss=1.25, ce_loss=1.25, samples=4, per_class={},
                           **{key: 75.0 for key in METRIC_NAMES})
            history.record("training", 2, metrics, first_iteration=1, batches=2)
            for split in ("train", "validation"):
                history.record(split, 2, metrics, checkpoint="checkpoints/iter_0000002.pth")
            history.record("test", 2, metrics, checkpoint="checkpoints/best.pth", checkpoint_sha256="verified-hash")
            restored = ExperimentHistory(temporary, config)
            self.assertEqual(restored.data["run_id"], run_id)
            self.assertEqual(len(restored.data["sessions"]), 2)
            self.assertEqual(len(restored.data["test"]), 1)
            tracker = WandbTracker(argparse.Namespace(wandb=False), restored)
            tracker.update_checkpoints([], 2)
            self.assertTrue(restored.data["validation"][0]["checkpoint_retained"])
            self.assertEqual(restored.data["validation"][0]["checkpoint"], "checkpoints/last.pth")
            tracker.update_checkpoints([], 3)
            self.assertFalse(restored.data["validation"][0]["checkpoint_retained"])
            self.assertIsNone(restored.data["validation"][0]["checkpoint"])
            restored.render()
            for filename in ("loss.png", "metrics.png"):
                with Image.open(Path(temporary) / "curves" / filename) as image:
                    image.verify()
            saved = json.loads((Path(temporary) / "results.json").read_text())
            self.assertTrue(saved["sessions"][0]["code_sha256"])
            self.assertEqual(saved["protocol"]["validation"], "original test split")

    def test_wandb_scalars_use_iteration_axis_and_skip_undefined_metrics(self):
        class RecordingRun:
            def log(self, payload):
                self.payload = payload
        tracker = object.__new__(WandbTracker)
        tracker.run = RecordingRun()
        tracker.log("test", dict(iteration=3, loss=0.7, mIoU=80, mPrecision=None,
                                 cafbr_enabled=False, cafbr_start_iteration=20001,
                                 per_class={"cloud": {"IoU": 79, "Precision": None}}))
        self.assertEqual(tracker.run.payload["iteration"], 3)
        self.assertEqual(tracker.run.payload["test/mIoU"], 80)
        self.assertFalse(tracker.run.payload["test/cafbr_enabled"])
        self.assertEqual(tracker.run.payload["test/cafbr_start_iteration"], 20001)
        self.assertEqual(tracker.run.payload["test/per_class/cloud/IoU"], 79)
        self.assertNotIn("test/mPrecision", tracker.run.payload)


if __name__ == "__main__":
    unittest.main()
