"""CUDA checks for the real scan kernel and the two full FMamba head variants."""

import json
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "FMamba tests require CUDA")
class CAFBRTests(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_scan_cuda_matches_recurrence_and_gradients(self):
        from baselines.fmamba_cafbr.selective_scan_interface import selective_scan_fn
        torch.manual_seed(42)
        tensors = [torch.randn(shape, device="cuda", requires_grad=True) for shape in
                   [(1, 4, 8), (1, 4, 8), (4, 3), (1, 1, 3, 8),
                    (1, 1, 3, 8), (4,), (4,)]]
        # Negative A gives stable state dynamics; each independent tensor gets gradients.
        tensors[2] = (-tensors[2].detach().abs()).requires_grad_()
        u, delta, A, B, C, D, bias = tensors
        actual = selective_scan_fn(u, delta, A, B, C, D, delta_bias=bias, delta_softplus=True)
        state = torch.zeros(1, 4, 3, device="cuda")
        outputs = []
        dt = F.softplus(delta + bias[None, :, None])
        for step in range(8):
            state = torch.exp(dt[:, :, step, None] * A[None]) * state + (
                dt[:, :, step, None] * B[:, 0, :, step][:, None] * u[:, :, step, None]
            )
            outputs.append((state * C[:, 0, :, step][:, None]).sum(-1) + D[None] * u[:, :, step])
        expected = torch.stack(outputs, -1)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        actual_grads = torch.autograd.grad(actual.square().mean(), tensors, retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().mean(), tensors)
        for actual_gradient, expected_gradient in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-4, rtol=2e-4)

    def test_native_and_mask2former_cafbr_and_scan_receive_gradients(self):
        from baselines.model_factory import build_model, prediction_logits, training_loss
        from baselines.cafbr_schedule import apply_cafbr_schedule
        for filename in ("hrc_whu_native.json", "hrc_whu_mask2former.json"):
            with self.subTest(config=filename):
                torch.manual_seed(42)
                config = json.loads((ROOT / "configs/fmamba" / filename).read_text())
                model = build_model(config).cuda().train()
                images = torch.randn(1, 3, 64, 64, device="cuda")
                labels = torch.zeros(1, 64, 64, dtype=torch.long, device="cuda")
                labels[:, :, 32:] = 1
                backbone = model.backbone if hasattr(model, "backbone") else model
                calls = []
                hooks = [refiner.register_forward_hook(lambda *args: calls.append(True))
                         for refiner in backbone.skip_refiners]
                apply_cafbr_schedule(model, config, config["max_iters"] // 2)
                warmup_output = model(images)
                warmup_loss = training_loss(model, warmup_output, labels, config)
                warmup_loss.backward()
                self.assertEqual(calls, [])
                self.assertTrue(all(p.grad is None and not p.requires_grad
                                    for p in backbone.skip_refiners.parameters()))
                model.zero_grad(set_to_none=True)
                del warmup_output, warmup_loss
                apply_cafbr_schedule(model, config, config["max_iters"] // 2 + 1)
                output = model(images)
                self.assertEqual(len(calls), 4)
                for hook in hooks:
                    hook.remove()
                self.assertEqual(prediction_logits(output, labels.shape[-2:]).shape, (1, 2, 64, 64))
                loss = training_loss(model, output, labels, config)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                backbone = model.backbone if hasattr(model, "backbone") else model
                refiners = [module for module in backbone.skip_refiners]
                self.assertEqual(len(refiners), 4)
                for refiner in refiners:
                    gradients = [p.grad for p in refiner.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                    self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)
                scans = [p.grad for name, p in backbone.named_parameters() if name.endswith("A_logs")]
                self.assertEqual(len(scans), 8)
                self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in scans))
                if isinstance(output, dict):
                    self.assertEqual(len(output["auxiliary_predictions"]), 9)
                    self.assertIsNotNone(model.class_predictor.weight.grad)
                del model, output, backbone, refiners, scans, loss


if __name__ == "__main__":
    unittest.main()
