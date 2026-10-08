import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from baselines.data import CloudDataset, dataset_splits
from baselines.tracking import ExperimentHistory
from baselines.lr_schedule import lr_multiplier, warmup_steps

ROOT = Path(__file__).resolve().parents[1]


class CloudSENPipelineTests(unittest.TestCase):
    def test_explicit_val_split_palette_and_history_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(data_root=str(root), classes=['clear', 'thick', 'thin', 'shadow'],
                          mean=[0, 0, 0], std=[1, 1, 1], image_size=[8, 8], ignore_index=255,
                          validation_split='val', test_split='test', selection_metric='mIoU')
            for split, value in [('train', 0), ('val', 3), ('test', 2)]:
                for folder in ('img_dir', 'ann_dir'):
                    (root / folder / split).mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(root / 'img_dir' / split / 'sample.png')
                mask = Image.fromarray(np.full((8, 8), value, dtype=np.uint8)).convert('P')
                mask.save(root / 'ann_dir' / split / 'sample.png')
            splits = dataset_splits(config)
            self.assertEqual(splits['validation'], 'val')
            self.assertEqual(CloudDataset(config, splits['validation'])[0][1].unique().tolist(), [3])
            self.assertEqual(CloudDataset(config, splits['test'])[0][1].unique().tolist(), [2])
            history = ExperimentHistory(root / 'run', config)
            self.assertEqual(history.data['protocol']['validation'], 'original val split')
            self.assertEqual(dataset_splits({})['validation'], 'test')

    def test_all_four_cloudsen_configs_have_matching_protocol_and_loss(self):
        for level in ('l1c', 'l2a'):
            for variant in ('native', 'mask2former'):
                config = json.loads((ROOT / f'configs/fmamba/cloudsen12_{level}_{variant}.json').read_text())
                self.assertEqual(config['validation_split'], 'test')
                self.assertEqual(config['test_split'], 'test')
                self.assertTrue(config['train_loss_only'])
                self.assertEqual(config['image_size'], [512, 512])
                self.assertEqual(len(config['classes']), 4)
                self.assertEqual(config['cafbr_start_ratio'], 0.5)
                self.assertEqual(config['max_iters'], 40000)
                self.assertEqual(config['batch_size'], 1)
                self.assertEqual(config['grad_accum_steps'], 4)
                self.assertEqual(config['effective_batch_size'], 4)
                self.assertEqual(config['val_interval'], 2000)
                self.assertEqual(config['lr_schedule'], 'cosine')
                self.assertEqual(config['warmup_ratio'], 0.1)
                self.assertEqual(warmup_steps(config), 4000)
                self.assertAlmostEqual(lr_multiplier(0, config) * config['lr'], 1e-6)
                self.assertAlmostEqual(lr_multiplier(4000, config), 1)
                self.assertAlmostEqual(lr_multiplier(22000, config), 0.5)
                self.assertEqual(lr_multiplier(40000, config), 0)
                if variant == 'native':
                    self.assertEqual(config['loss']['ce_weight'], 5)
                    self.assertEqual(config['loss']['dice_weight'], 5)
                else:
                    self.assertEqual(config['mask2former']['class_weight'], 2)
                    self.assertEqual(config['mask2former']['mask_weight'], 5)
                    self.assertEqual(config['mask2former']['dice_weight'], 5)


if __name__ == '__main__':
    unittest.main()
