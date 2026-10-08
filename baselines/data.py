"""Read original HRC-WHU PNG pairs; validation always uses the test split."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, Sampler


class CloudDataset(Dataset):
    def __init__(self, config, split, limit=None):
        self.config, self.split = config, split
        root = Path(config["data_root"])
        image_dir, mask_dir = root / "img_dir" / split, root / "ann_dir" / split
        images, masks = sorted(image_dir.glob("*.png")), sorted(mask_dir.glob("*.png"))
        if not images or {p.name for p in images} != {p.name for p in masks}:
            raise ValueError(f"Missing or unpaired PNG files in {image_dir} / {mask_dir}")
        self.pairs = [(image, mask_dir / image.name) for image in images]
        if limit is not None:
            self.pairs = self.pairs[:limit]
        self.mean = np.array(config["mean"], dtype=np.float32)
        self.std = np.array(config["std"], dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        index, seed = item if isinstance(item, tuple) else (item, item)
        rng = random.Random(seed)
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            # Preserve palette indices (0/1), rather than converting palette colors.
            mask = Image.fromarray(np.array(source))
        if image.size != mask.size:
            raise ValueError(f"Image/mask sizes differ: {image_path}")
        labels = np.array(mask)
        if labels.ndim != 2 or not np.isin(
            labels, list(range(len(self.config["classes"]))) + [self.config["ignore_index"]]
        ).all():
            raise ValueError(f"Invalid class indices: {mask_path}")
        height, width = self.config["image_size"]
        if self.split == "train" and image.width >= width and image.height >= height:
            left, top = rng.randrange(image.width - width + 1), rng.randrange(image.height - height + 1)
            box = (left, top, left + width, top + height)
            image, mask = image.crop(box), mask.crop(box)
        else:
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        if self.split == "train":
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                if rng.random() < 0.5:
                    image = enhancer(image).enhance(rng.uniform(0.5, 1.5))
        pixels = (np.array(image, dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(pixels.transpose(2, 0, 1).copy()), torch.from_numpy(
            np.array(mask, dtype=np.int64).copy()
        )


class IterationBatchSampler(Sampler):
    """Shuffled infinite sampling reproducible across resume and worker prefetch."""

    def __init__(self, size, batch_size, start_iter, max_iters, seed):
        self.size, self.batch_size = size, batch_size
        self.start_iter, self.max_iters, self.seed = start_iter, max_iters, seed

    def __len__(self):
        return self.max_iters - self.start_iter

    def __iter__(self):
        previous_epoch, order = None, None
        for step in range(self.start_iter, self.max_iters):
            batch = []
            for position in range(step * self.batch_size, (step + 1) * self.batch_size):
                epoch, offset = divmod(position, self.size)
                if epoch != previous_epoch:
                    order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                    previous_epoch = epoch
                batch.append((order[offset], self.seed + position))
            yield batch
