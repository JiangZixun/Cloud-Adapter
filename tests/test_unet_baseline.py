import json
import tempfile
import unittest
from pathlib import Path

import torch

from baselines.checkpoints import CheckpointManager
from baselines.data import IterationBatchSampler
from baselines.metrics import confusion_matrix, segmentation_metrics
from baselines.unet import UNet


class BaselineTests(unittest.TestCase):
    def test_metrics_against_hand_calculated_confusion(self):
        # Rows = actual, columns = predicted; class 0: TP=8, FP=1, FN=2.
        result = segmentation_metrics(torch.tensor([[8, 2], [1, 9]]), ["clear", "cloud"])
        self.assertAlmostEqual(result["aAcc"], 85)
        self.assertAlmostEqual(result["mIoU"], (8 / 11 + 9 / 12) * 50)
        self.assertAlmostEqual(result["mAcc"], 85)
        self.assertAlmostEqual(result["mPrecision"], (8 / 9 + 9 / 11) * 50)
        self.assertAlmostEqual(result["mDice"], (16 / 19 + 18 / 21) * 50)
        self.assertAlmostEqual(result["mFscore"], result["mDice"])
        self.assertEqual(result["mRecall"], result["mAcc"])
        matrix = confusion_matrix(torch.tensor([0, 1, 0]), torch.tensor([0, 1, 255]))
        self.assertTrue(torch.equal(matrix, torch.eye(2, dtype=torch.int64)))
        empty_class = segmentation_metrics(torch.tensor([[4, 0], [0, 0]]), ["clear", "cloud"])
        self.assertIsNone(empty_class["per_class"]["cloud"]["IoU"])
        json.dumps(empty_class, allow_nan=False)

    def test_ranking_pruning_last_best_and_reopen(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(temporary)
            for iteration, score in enumerate([60, 90, 70, 80, 50], 1):
                manager.update({"iteration": iteration}, {"mIoU": score})
            directory = Path(temporary)
            self.assertEqual(len(list(directory.glob("iter_*.pth"))), 3)
            self.assertEqual([row["iteration"] for row in manager.top], [2, 4, 3])
            self.assertEqual(torch.load(directory / "best.pth", weights_only=True)["iteration"], 2)
            self.assertEqual(torch.load(directory / "last.pth", weights_only=True)["iteration"], 5)
            restored = CheckpointManager(temporary)
            restored.update({"iteration": 6}, {"mIoU": 95})
            self.assertEqual([row["iteration"] for row in restored.top], [6, 2, 4])
            self.assertEqual(len(list(directory.glob("iter_*.pth"))), 3)

    def test_resumed_sampler_matches_uninterrupted_samples_and_augmentations(self):
        batches = list(IterationBatchSampler(7, 4, 0, 8, 42))
        self.assertEqual(list(IterationBatchSampler(7, 4, 3, 8, 42)), batches[3:])
        self.assertEqual(len({index for batch in batches[:2] for index, _ in batch}), 7)

    def test_unet_preserves_resolution_and_backward(self):
        torch.set_num_threads(2)
        model = UNet(base_channels=4)
        images = torch.randn(2, 3, 35, 37)
        logits = model(images)
        self.assertEqual(logits.shape, (2, 2, 35, 37))
        torch.nn.functional.cross_entropy(logits, torch.zeros(2, 35, 37, dtype=torch.long)).backward()
        self.assertTrue(torch.isfinite(model.encoder[0][0].weight.grad).all())
        self.assertGreater(model.encoder[0][0].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
