import copy
import unittest

import torch

from baselines.data import IterationBatchSampler
from baselines.optimization import accumulation_steps, accumulated_update


class GradientAccumulationTests(unittest.TestCase):
    def test_accumulated_adamw_matches_physical_batch_and_unscaled_loss(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        images, targets = torch.randn(8, 3), torch.randn(8, 2)
        iterator = iter([(images[i:i+1], targets[i:i+1]) for i in range(8)])
        objective = lambda model, output, labels, config: torch.nn.functional.mse_loss(output, labels)
        for step in range(2):
            value = accumulated_update(model, iterator, optimizer, scaler, torch.device('cpu'),
                                       dict(grad_accum_steps=4, amp=False), objective)
            reference_optimizer.zero_grad(set_to_none=True)
            loss = objective(reference, reference(images[4*step:4*step+4]), targets[4*step:4*step+4], {})
            loss.backward()
            reference_optimizer.step()
            self.assertAlmostEqual(value, loss.item(), places=6)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected)
        self.assertEqual(optimizer.state[model.weight]['step'].item(), 2)

    def test_resume_sampler_starts_after_completed_micro_batches(self):
        accumulation = 4
        complete = list(IterationBatchSampler(7, 1, 0, 5 * accumulation, 42))
        resumed = list(IterationBatchSampler(7, 1, 2 * accumulation, 5 * accumulation, 42))
        self.assertEqual(resumed, complete[8:])
        self.assertEqual(len(resumed), 12)

    def test_distributed_sampler_partitions_global_batches_and_resumes(self):
        reference = list(IterationBatchSampler(17, 6, 0, 8, 42))
        ranks = [list(IterationBatchSampler(17, 2, 0, 8, 42, rank, 3)) for rank in range(3)]
        for step, batch in enumerate(reference):
            self.assertEqual([sample for rank in ranks for sample in rank[step]], batch)
            self.assertEqual(len({seed for _, seed in batch}), 6)
        for rank in range(3):
            resumed = list(IterationBatchSampler(17, 2, 3, 8, 42, rank, 3))
            self.assertEqual(resumed, ranks[rank][3:])
        for rank, world_size in ((-1, 2), (2, 2), (0, 0)):
            with self.assertRaises(ValueError):
                IterationBatchSampler(17, 2, 0, 8, 42, rank, world_size)

    def test_default_and_invalid_accumulation(self):
        self.assertEqual(accumulation_steps({}), 1)
        for value in (0, -1, 1.5, True, '4'):
            with self.assertRaises(ValueError):
                accumulation_steps(dict(grad_accum_steps=value))


if __name__ == '__main__':
    unittest.main()
