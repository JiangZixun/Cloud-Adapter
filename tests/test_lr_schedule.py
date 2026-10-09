import unittest
import json
import runpy
from pathlib import Path

import torch

from baselines.lr_schedule import lr_multiplier, warmup_steps, validate_resume_schedule


class LearningRateScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(max_iters=40000, poly_power=0.9)

    def test_legacy_schedule_is_preserved(self):
        for step in (0, 1, 4000, 20000, 39999, 40000):
            self.assertEqual(lr_multiplier(step, self.config),
                             max(0, 1 - step / 40000) ** 0.9)

    def test_ten_percent_warmup_boundaries(self):
        config = dict(self.config, warmup_ratio=0.1, warmup_start_factor=0.01)
        self.assertEqual(warmup_steps(config), 4000)
        self.assertAlmostEqual(lr_multiplier(0, config) * 1e-4, 1e-6)
        self.assertAlmostEqual(lr_multiplier(2000, config), 0.505)
        self.assertEqual(lr_multiplier(4000, config), 1)
        self.assertAlmostEqual(lr_multiplier(22000, config), 0.5 ** 0.9)
        self.assertEqual(lr_multiplier(40000, config), 0)

    def test_cosine_warmup_endpoints_and_monotonic_decay(self):
        config = dict(max_iters=40000, lr_schedule="cosine", warmup_ratio=0.1,
                      warmup_start_factor=0.01, min_lr_ratio=0.0)
        self.assertAlmostEqual(lr_multiplier(0, config), 0.01)
        self.assertAlmostEqual(lr_multiplier(2000, config), 0.505)
        self.assertEqual(lr_multiplier(4000, config), 1.0)
        self.assertAlmostEqual(lr_multiplier(22000, config), 0.5)
        self.assertEqual(lr_multiplier(40000, config), 0)
        self.assertEqual(lr_multiplier(50000, config), 0)
        values = [lr_multiplier(step, config) for step in range(4000, 40001, 1000)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        config["min_lr_ratio"] = 0.1
        self.assertAlmostEqual(lr_multiplier(22000, config), 0.55)
        self.assertEqual(lr_multiplier(40000, config), 0.1)

    def test_resume_rejects_schedule_change_but_accepts_legacy_defaults(self):
        saved = dict(self.config, lr=1e-4, weight_decay=0.05)
        validate_resume_schedule(saved, dict(saved, lr_schedule="poly", warmup_ratio=0))
        for override in (dict(lr_schedule="cosine"), dict(warmup_ratio=0.1),
                         dict(lr=5e-5), dict(poly_power=1), dict(min_lr_ratio=0.1)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_resume_schedule(saved, dict(saved, **override))

    def test_cafbr_phase_schedule_restarts_at_activation(self):
        config = dict(max_iters=40000, model="FMamba_CAFBR", cafbr_start_ratio=0.5,
                      fmamba=dict(skip_refinement=dict(enabled=True)),
                      lr_schedule="cosine_cafbr_phase", warmup_ratio=0.1,
                      warmup_start_factor=0.01, min_lr_ratio=0.0)
        self.assertEqual(warmup_steps(config), 2000)
        for phase_offset in (0, 20000):
            for local_step, expected in ((0, 0.01), (1000, 0.505), (2000, 1), (11000, 0.5)):
                self.assertAlmostEqual(lr_multiplier(phase_offset + local_step, config), expected)
        self.assertLess(lr_multiplier(19999, config), 1e-7)
        self.assertAlmostEqual(lr_multiplier(20000, config), 0.01)
        self.assertEqual(lr_multiplier(40000, config), 0)
        self.assertEqual(lr_multiplier(40001, config), 0)
        # A non-even split follows the actual CAFBR boundary, not a hardcoded midpoint.
        uneven = dict(config, max_iters=30000, cafbr_start_ratio=0.25)
        self.assertEqual(warmup_steps(uneven), 750)
        self.assertEqual(lr_multiplier(750, uneven), 1)
        self.assertAlmostEqual(lr_multiplier(7500, uneven), 0.01)
        self.assertEqual(lr_multiplier(9750, uneven), 1)
        self.assertEqual(lr_multiplier(30000, uneven), 0)

    def test_phase_scheduler_optimizer_rates_and_resume_across_boundary(self):
        config = dict(max_iters=20, model="FMamba_CAFBR", cafbr_start_ratio=0.5,
                      fmamba=dict(skip_refinement=dict(enabled=True)),
                      lr_schedule="cosine_cafbr_phase", warmup_ratio=0.1)
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=3e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, config))
        for _ in range(9):
            optimizer.step()
            scheduler.step()
        optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
        expected = []
        for _ in range(11):
            expected.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(expected[1], 3e-7)  # update 11: CAFBR's first update
        self.assertAlmostEqual(expected[2], 3e-5)  # update 12: warmup completed
        restored = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=3e-5)
        restored_schedule = torch.optim.lr_scheduler.LambdaLR(restored, lambda step: lr_multiplier(step, config))
        restored.load_state_dict(optimizer_state)
        restored_schedule.load_state_dict(scheduler_state)
        actual = []
        for _ in range(11):
            actual.append(restored.param_groups[0]["lr"])
            restored.step()
            restored_schedule.step()
        self.assertEqual(actual, expected)
        for override in (dict(max_iters=30), dict(cafbr_start_ratio=0.25), dict(lr_schedule="cosine")):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_resume_schedule(config, dict(config, **override))

    def test_phase_schedule_rejects_empty_phases_and_disabled_cafbr(self):
        config = dict(max_iters=20, model="FMamba_CAFBR", cafbr_start_ratio=0.5,
                      fmamba=dict(skip_refinement=dict(enabled=True)),
                      lr_schedule="cosine_cafbr_phase", warmup_ratio=0.1)
        for override in (dict(cafbr_start_ratio=0), dict(cafbr_start_ratio=1),
                         dict(cafbr_start_ratio=-0.1), dict(cafbr_start_ratio=float("nan")),
                         dict(max_iters=1), dict(max_iters=2), dict(warmup_ratio=0.99),
                         dict(model="UNet"), dict(fmamba=dict(skip_refinement=dict(enabled=False)))):
            with self.subTest(override=override), self.assertRaises(ValueError):
                warmup_steps(dict(config, **override))

    def test_hrc_baselines_use_cosine_and_other_models_keep_original_poly(self):
        root = Path(__file__).resolve().parents[1]
        json_paths = list((root / "configs").glob("*/hrc_whu*.json"))
        python_paths = list((root / "configs").glob("*/*hrc_whu.py"))
        self.assertEqual(len(json_paths), 7)
        self.assertEqual(len(python_paths), 7)
        for path in json_paths:
            with self.subTest(path=path):
                config = json.loads(path.read_text())
                self.assertEqual(config["lr_schedule"], "cosine")
                self.assertEqual(warmup_steps(config), config["max_iters"] // 10)
                self.assertEqual(config["warmup_start_factor"], 0.01)
                self.assertEqual(config["min_lr_ratio"], 0)
                self.assertEqual(config["val_interval"], 2000)
                self.assertIn("warmup_cosine", config["work_dir"])
        for path in python_paths:
            with self.subTest(path=path):
                config = runpy.run_path(str(path))
                total = config["train_cfg"]["max_iters"]
                self.assertEqual(config["param_scheduler"], [dict(
                    type="PolyLR", eta_min=0, power=0.9,
                    begin=0, end=total, by_epoch=False)])
                interval = 15 if path.parent.name == "clip" else 4000
                self.assertEqual(config["train_cfg"]["val_interval"], interval)
                self.assertEqual(config["default_hooks"]["logger"]["interval"], interval)
                self.assertEqual(config["default_hooks"]["checkpoint"]["interval"], interval)
                self.assertNotIn("work_dir", config)

    def test_scheduler_resume_matches_uninterrupted_learning_rates(self):
        config = dict(max_iters=20, lr_schedule="cosine", warmup_ratio=0.1)
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, config))
        for _ in range(7):
            optimizer.step()
            scheduler.step()
        optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
        expected = []
        for _ in range(13):
            expected.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        restored = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
        restored_schedule = torch.optim.lr_scheduler.LambdaLR(restored, lambda step: lr_multiplier(step, config))
        restored.load_state_dict(optimizer_state)
        restored_schedule.load_state_dict(scheduler_state)
        actual = []
        for _ in range(13):
            actual.append(restored.param_groups[0]["lr"])
            restored.step()
            restored_schedule.step()
        self.assertEqual(actual, expected)

    def test_invalid_settings_and_short_smoke_schedule(self):
        for settings in (dict(warmup_ratio=-0.1), dict(warmup_ratio=1),
                         dict(warmup_ratio=float("nan")), dict(warmup_start_factor=0),
                         dict(poly_power=0), dict(lr_schedule="unknown"),
                         dict(min_lr_ratio=-0.1), dict(min_lr_ratio=float("nan")),
                         dict(max_iters=1, warmup_ratio=0.1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                warmup_steps(dict(self.config, **settings))
        self.assertEqual(warmup_steps(dict(self.config, max_iters=5, warmup_ratio=0.1)), 1)


if __name__ == "__main__":
    unittest.main()
