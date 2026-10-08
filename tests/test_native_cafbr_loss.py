import math
import unittest

import torch
from torch import nn

from baselines.model_factory import training_loss


class NativeLossTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(model="FMamba_CAFBR", ignore_index=255,
                           loss=dict(ce_weight=5.0, dice_weight=5.0, dice_eps=1e-5))

    def test_weighted_loss_against_hand_calculation_and_ignore_gradient(self):
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        labels = torch.tensor([[[0, 0], [1, 255]]])
        eps = 1e-5
        expected_dice = 1 - ((2 + eps) / (3.5 + eps) + (1 + eps) / (2.5 + eps)) / 2
        actual = training_loss(nn.Identity(), logits, labels, self.config)
        self.assertAlmostEqual(actual.item(), 5 * math.log(2) + 5 * expected_dice, places=5)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, :, 1, 1].abs().sum().item(), 0)
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_all_ignored_pixels_and_unchanged_unet_loss(self):
        logits = torch.randn(1, 2, 2, 2, requires_grad=True)
        ignored = torch.full((1, 2, 2), 255, dtype=torch.long)
        loss = training_loss(nn.Identity(), logits, ignored, self.config)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)
        labels = torch.tensor([[[0, 1], [1, 0]]])
        actual = training_loss(nn.Identity(), logits, labels, dict(model="UNet", ignore_index=255))
        torch.testing.assert_close(actual, nn.functional.cross_entropy(logits, labels))


if __name__ == "__main__":
    unittest.main()
