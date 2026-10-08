import json
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from baselines.losses import native_lsmamba_loss

ROOT = Path(__file__).resolve().parents[1]


class LSMambaLossTests(unittest.TestCase):
    def test_source_squared_dice_formula_and_ignored_gradient(self):
        logits = torch.tensor([[[[1.0, -0.4, 0.1]], [[0.2, 0.8, 0.5]]]], requires_grad=True)
        labels = torch.tensor([[[0, 1, 255]]])
        config = json.loads((ROOT / 'configs/lsmamba/hrc_whu_native.json').read_text())
        self.assertEqual(config['loss']['ce_weight'], 5.0)
        self.assertEqual(config['loss']['dice_weight'], 5.0)
        actual = native_lsmamba_loss(logits, labels, config)
        probabilities = logits[..., :2].softmax(1)
        targets = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        terms = []
        for channel in range(2):
            score, target = probabilities[:, channel], targets[:, channel]
            terms.append(1 - (2 * (score * target).sum() + 1e-5) /
                         (score.square().sum() + target.square().sum() + 1e-5))
        expected = 5 * F.cross_entropy(logits[..., :2], labels[..., :2]) + 5 * torch.stack(terms).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(logits.grad[..., 2].abs().sum().item(), 0)
        self.assertEqual(native_lsmamba_loss(logits, torch.full_like(labels, 255), config).item(), 0)


@unittest.skipUnless(torch.cuda.is_available(), "LS-Mamba requires CUDA")
class LSMambaCUDATests(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_spectral_mamba_matches_direct_recurrence_and_gradients(self):
        from baselines.lsmamba.mamba_simple import Mamba
        torch.manual_seed(42)
        model = Mamba(d_model=8, d_state=4, d_conv=3).cuda()
        inputs = torch.randn(2, 8, 8, device='cuda', requires_grad=True)
        actual = model(inputs)
        x, z = model.in_proj(inputs).transpose(1, 2).chunk(2, dim=1)
        x = F.silu(model.conv1d(x)[..., :inputs.shape[1]])
        delta, B, C = torch.split(model.x_proj(x.transpose(1, 2)), [model.dt_rank, model.d_state, model.d_state], -1)
        delta = F.softplus(F.linear(delta, model.dt_proj.weight, model.dt_proj.bias)).transpose(1, 2)
        A = -model.A_log.exp()
        state = torch.zeros(2, model.d_inner, model.d_state, device='cuda')
        values = []
        for step in range(inputs.shape[1]):
            dt = delta[:, :, step, None]
            state = torch.exp(dt * A[None]) * state + dt * B[:, step, None] * x[:, :, step, None]
            values.append((state * C[:, step, None]).sum(-1) + model.D[None] * x[:, :, step])
        expected = model.out_proj((torch.stack(values, -1) * F.silu(z)).transpose(1, 2))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        tensors = [inputs, *model.parameters()]
        actual_gradients = torch.autograd.grad(actual.square().mean(), tensors, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected.square().mean(), tensors)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-4)

    def test_local_scan_roundtrip_and_gradient_with_padding_and_directions(self):
        from baselines.lsmamba.mamba2.local_scan import local_scan_bchw, local_reverse
        image = torch.randn(2, 3, 3, 5, device='cuda', requires_grad=True)
        for flip in (False, True):
            for column_first in (False, True):
                scanned = local_scan_bchw(image, 2, 3, 5, flip=flip, column_first=column_first)
                restored = local_reverse(scanned, 2, 3, 5, flip=flip, column_first=column_first)
                torch.testing.assert_close(restored, image.flatten(2))
                gradient, = torch.autograd.grad(restored.sum(), image)
                torch.testing.assert_close(gradient, torch.ones_like(image))

    def test_both_heads_train_all_spectral_local_and_global_branches(self):
        from baselines.model_factory import build_model, prediction_logits, training_loss
        for filename in ('hrc_whu_native.json', 'hrc_whu_mask2former.json'):
            with self.subTest(config=filename):
                config = json.loads((ROOT / 'configs/lsmamba' / filename).read_text())
                # A single sample can legitimately drop an entire spatial branch;
                # disable stochastic depth only for this gradient connectivity check.
                config['lsmamba']['drop_path_rate'] = 0.0
                model = build_model(config).cuda().train()
                images = torch.randn(1, 3, 64, 64, device='cuda')
                labels = torch.zeros(1, 64, 64, dtype=torch.long, device='cuda')
                labels[:, :, 32:] = 1
                output = model(images)
                self.assertEqual(prediction_logits(output, (64, 64)).shape, (1, 2, 64, 64))
                loss = training_loss(model, output, labels, config)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                backbone = model.backbone if hasattr(model, 'backbone') else model
                blocks = [block for layer in [*backbone.vmunet.layers, *backbone.vmunet.layers_up] for block in layer.blocks]
                self.assertEqual(len(blocks), 16)
                for block in blocks:
                    for branch in (block.spemamba, block.vss_block, block.vss_gl_block):
                        gradients = [p.grad for p in branch.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                if isinstance(output, dict):
                    self.assertEqual(len(output['auxiliary_predictions']), 9)
                    self.assertIsNotNone(model.class_predictor.weight.grad)
                del model, output, loss, backbone, blocks, branch, gradients


if __name__ == '__main__':
    unittest.main()
