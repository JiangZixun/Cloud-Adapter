"""Verify the configurable native FMamba hierarchy and legacy checkpoints."""

import json
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class FMambaScalingTests(unittest.TestCase):
    def test_legacy_checkpoint_loads_strictly(self):
        from baselines.fmamba_cafbr.models import native_model
        config = json.loads((ROOT / 'configs/fmamba/cloudsen12_l1c_native.json').read_text())
        model = native_model(config)
        checkpoint = ROOT / 'experiments/CloudSEN12_L1C/FMamba_CAFBR/checkpoints/best.pth'
        if not checkpoint.exists():
            self.skipTest('Original completed-run checkpoint is unavailable')
        state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 63768409)
        names = list(dict(model.named_parameters()))
        # Optimizer checkpoints associate moments by parameter order.
        self.assertLess(names.index('upsample_1.0.weight'),
                        names.index('stage_up_4.0.weight'))

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires the CUDA scan kernel')
    def test_five_downsamples_at_512_and_all_branch_gradients(self):
        from baselines.fmamba_cafbr.models import native_model
        from baselines.model_factory import training_loss
        for base in (16, 32):
            with self.subTest(base=base):
                config = json.loads((ROOT / f'configs/fmamba/cloudsen12_l1c_native_base{base}_down5.json').read_text())
                model = native_model(config).cuda().train()
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 {16: 4199547, 32: 16260539}[base])
                self.assertEqual(sum(isinstance(m, torch.nn.ConvTranspose2d)
                                     for m in model.modules()), 5)
                self.assertEqual(model.stage_1[0].stride, (2, 2))
                sizes = []
                hooks = [getattr(model, f'stage_{stage}').register_forward_hook(
                    lambda module, inputs, output: sizes.append(tuple(output.shape[1:])))
                    for stage in range(1, 6)]
                output = model(torch.randn(1, 3, 512, 512, device='cuda'))
                for hook in hooks:
                    hook.remove()
                self.assertEqual(sizes, [(base * 2 ** i, 256 // 2 ** i, 256 // 2 ** i)
                                         for i in range(5)])
                self.assertEqual(tuple(output.shape), (1, 4, 512, 512))
                labels = torch.randint(0, 4, (1, 512, 512), device='cuda')
                loss = training_loss(model, output, labels, config)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                scans = [p.grad for name, p in model.named_parameters() if name.endswith('A_logs')]
                self.assertEqual(len(scans), 9)
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in scans))
                self.assertEqual(len(model.skip_refiners), 4)
                for refiner in model.skip_refiners:
                    gradients = [p.grad for p in refiner.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                del model, output, loss, scans, gradients
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
