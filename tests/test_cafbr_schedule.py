import unittest

from torch import nn

from baselines.cafbr_schedule import apply_cafbr_schedule, cafbr_start_iteration, restore_cafbr_state


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip_refiners = nn.ModuleList([nn.Linear(2, 2)])
        self.skip_refinement_enabled = True
        self.skip_refinement_active = True

    def set_skip_refinement_active(self, active):
        self.skip_refinement_active = active


class CAFBRScheduleTests(unittest.TestCase):
    def test_midpoint_freezes_and_then_enables_parameters_for_both_wrappers(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                backbone = Backbone()
                model = nn.Module() if wrapped else backbone
                if wrapped:
                    model.backbone = backbone
                config = dict(model="FMamba_CAFBR", max_iters=40000, cafbr_start_ratio=0.5)
                state = apply_cafbr_schedule(model, config, 20000)
                self.assertFalse(state['cafbr_enabled'])
                self.assertTrue(all(not p.requires_grad for p in backbone.skip_refiners.parameters()))
                state = apply_cafbr_schedule(model, config, 20001)
                self.assertTrue(state['cafbr_enabled'])
                self.assertTrue(all(p.requires_grad for p in backbone.skip_refiners.parameters()))
                self.assertEqual(cafbr_start_iteration(dict(config, max_iters=5)), 3)

    def test_best_restores_saved_phase_and_resume_uses_next_absolute_step(self):
        model = Backbone()
        config = dict(model="FMamba_CAFBR", max_iters=4, cafbr_start_ratio=0.5)
        apply_cafbr_schedule(model, config, 4)
        checkpoint = dict(config=config, iteration=2, cafbr_enabled=False)
        self.assertFalse(restore_cafbr_state(model, checkpoint)['cafbr_enabled'])
        self.assertTrue(apply_cafbr_schedule(model, config, 3)['cafbr_enabled'])
        # Explicit phase survives a later change to the training horizon.
        checkpoint.update(iteration=4, cafbr_enabled=True, config=dict(config, max_iters=100))
        self.assertTrue(restore_cafbr_state(model, checkpoint)['cafbr_enabled'])
        # Legacy models used CAFBR from their first iteration.
        checkpoint = dict(config=dict(model="FMamba_CAFBR", max_iters=40000), iteration=1)
        self.assertTrue(restore_cafbr_state(model, checkpoint)['cafbr_enabled'])

    def test_invalid_ratios_and_unet(self):
        for ratio in (-0.1, 1.1, float('nan'), 'half'):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                cafbr_start_iteration(dict(max_iters=4, cafbr_start_ratio=ratio))
        self.assertEqual(apply_cafbr_schedule(nn.Linear(2, 2), dict(model="UNet"), 1), {})


if __name__ == '__main__':
    unittest.main()
