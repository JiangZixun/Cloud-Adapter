import unittest

from baselines.data import EpochBatchSampler
from baselines.epoch_schedule import resolve_epoch_schedule, epoch_context, validate_resume_epochs
from baselines.cafbr_schedule import cafbr_start_iteration
from baselines.lr_schedule import warmup_steps, lr_multiplier


class EpochTrainingTests(unittest.TestCase):
    def test_cloudsen_counts_and_schedule_follow_world_size(self):
        for world, expected in [(1, 2123), (2, 1062)]:
            config = dict(max_epochs=20, val_interval_epochs=1, batch_size=4,
                          grad_accum_steps=1, world_size=world, warmup_ratio=.1,
                          warmup_start_factor=.01, lr_schedule='cosine', cafbr_start_ratio=.5)
            resolve_epoch_schedule(config, 8490)
            self.assertEqual(config['steps_per_epoch'], expected)
            self.assertEqual(config['max_iters'], 20 * expected)
            self.assertEqual(config['val_interval'], expected)
            self.assertEqual(warmup_steps(config), 2 * expected)
            self.assertEqual(cafbr_start_iteration(config), 10 * expected + 1)
            self.assertEqual(lr_multiplier(config['max_iters'], config), 0)
            self.assertEqual(epoch_context(3 * expected, config)['epoch'], 3)

    def test_each_epoch_covers_dataset_with_only_final_batch_padding(self):
        samplers = [list(EpochBatchSampler(27, 4, 0, 3, 42, rank, 2)) for rank in range(2)]
        orders = []
        for epoch in range(3):
            samples = [item for step in range(epoch * 4, (epoch + 1) * 4)
                       for rank in range(2) for item in samplers[rank][step]]
            self.assertEqual(len(samples), 32)
            self.assertEqual(len({seed for _, seed in samples}), 32)
            self.assertEqual(set(i for i, _ in samples[:27]), set(range(27)))
            self.assertEqual([i for i, _ in samples[27:]], [i for i, _ in samples[:5]])
            orders.append([i for i, _ in samples[:27]])
        self.assertNotEqual(orders[0], orders[1])
        self.assertTrue(all(len(batch) == 4 for rank in samplers for batch in rank))

    def test_resume_matches_uninterrupted_sampler_mid_epoch(self):
        for rank in (0, 1):
            full = list(EpochBatchSampler(27, 4, 0, 3, 42, rank, 2))
            for start in (3, 4, 7):
                resumed = list(EpochBatchSampler(27, 4, start, 3, 42, rank, 2))
                self.assertEqual(resumed, full[start:])

    def test_resume_rejects_dataset_size_or_training_unit_changes(self):
        saved = dict(max_epochs=20, batch_size=4, grad_accum_steps=1, world_size=2)
        resolve_epoch_schedule(saved, 8490)
        validate_resume_epochs(saved, dict(saved))
        with self.assertRaises(ValueError):
            validate_resume_epochs({}, saved)
        changed = dict(saved)
        resolve_epoch_schedule(changed, 8500)
        with self.assertRaises(ValueError):
            validate_resume_epochs(saved, changed)

    def test_invalid_epoch_targets_and_accumulation(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                resolve_epoch_schedule(dict(max_epochs=value, batch_size=4), 27)
        with self.assertRaises(ValueError):
            resolve_epoch_schedule(dict(max_epochs=20, batch_size=4, grad_accum_steps=2), 27)


if __name__ == '__main__':
    unittest.main()
