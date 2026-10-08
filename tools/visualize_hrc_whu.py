"""Visualize HRC-WHU using the same CloudDataset as the UNet trainer."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.data import CloudDataset
from baselines.checkpoints import write_json


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def denormalize(tensor, config):
    pixels = tensor.numpy().transpose(1, 2, 0)
    return np.rint(np.clip(pixels * np.array(config["std"]) + np.array(config["mean"]), 0, 255)).astype(np.uint8)


def color_mask(mask):
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask == 1] = [255, 190, 0]
    result[mask == 255] = [255, 0, 255]
    return result


def overlay(rgb, mask):
    result = rgb.copy()
    selected = mask == 1
    result[selected] = np.rint(0.55 * rgb[selected] + 0.45 * np.array([255, 190, 0])).astype(np.uint8)
    return result


def representative_indices(dataset, count):
    groups = {}
    for index, (image, _) in enumerate(dataset.pairs):
        groups.setdefault(image.stem.rsplit("_", 1)[0], []).append(index)
    selected = [indices[len(indices) // 2] for indices in groups.values()]
    if len(selected) < count:
        selected.extend(i for i in range(len(dataset)) if i not in selected)
    return selected[:count]


def draw_grid(dataset, config, indices, output, split):
    width, height = 256, 256
    columns = ["Original RGB", "Original ground truth", "Original overlay"]
    if split == "train":
        columns += ["Loader RGB (augmented)", "Loader ground truth", "Loader overlay"]
    else:
        columns = ["Loader RGB", "Loader ground truth", "Loader overlay"]
    margin, gap, header, row_height = 16, 12, 88, height + 45
    canvas = Image.new("RGB", (margin * 2 + (width + gap) * len(columns) - gap,
                               header + row_height * len(indices) + 20), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), f"HRC-WHU / {split} ({len(dataset)} samples)", fill="#152030", font=font(22))
    draw.text((margin, 40), "Ground truth: clear sky = black (0), cloud = gold (1). Overlay: cloud highlighted.",
              fill="#253040", font=font(14))
    for col, name in enumerate(columns):
        draw.text((margin + col * (width + gap), 65), name, fill="#152030", font=font(14))
    metadata = []
    for row, index in enumerate(indices):
        image_path, mask_path = dataset.pairs[index]
        seed = config["seed"] + index
        loaded, labels = dataset[(index, seed)]
        rgb, mask = denormalize(loaded, config), labels.numpy()
        panels = [rgb, color_mask(mask), overlay(rgb, mask)]
        if split == "train":
            with Image.open(image_path) as source:
                original = np.array(source.convert("RGB"))
            with Image.open(mask_path) as source:
                original_mask = np.array(source)
            panels = [original, color_mask(original_mask), overlay(original, original_mask)] + panels
        y = header + row * row_height
        draw.text((margin, y), f"{image_path.name} | cloud pixels: {(mask == 1).mean() * 100:.1f}%",
                  fill="#152030", font=font(15))
        for col, pixels in enumerate(panels):
            panel = Image.fromarray(pixels)
            panel = panel.resize((width, height), Image.Resampling.NEAREST if col % 3 == 1 else Image.Resampling.BILINEAR)
            canvas.paste(panel, (margin + col * (width + gap), y + 26))
        metadata.append(dict(file=image_path.name, image_path=str(image_path), mask_path=str(mask_path),
                             augmentation_seed=seed if split == "train" else None,
                             cloud_percent=float((mask == 1).mean() * 100), labels=np.unique(mask).tolist()))
    canvas.save(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/unet/hrc_whu.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/HRC_WHU/UNet_data_preview/visualizations")
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "test"):
        dataset = CloudDataset(config, split)
        histogram = np.zeros(len(config["classes"]), dtype=np.int64)
        for _, path in dataset.pairs:
            with Image.open(path) as source:
                mask = np.array(source)
            values, counts = np.unique(mask, return_counts=True)
            if any(value not in (0, 1, config["ignore_index"]) for value in values):
                raise ValueError(f"Invalid labels: {path}")
            for value, count in zip(values, counts):
                if value != config["ignore_index"]:
                    histogram[value] += count
        selected = representative_indices(dataset, min(args.samples, len(dataset)))
        output = args.output_dir / f"{split}_samples.png"
        samples = draw_grid(dataset, config, selected, output, split)
        report[split] = dict(count=len(dataset), pixel_counts=histogram.tolist(),
                             class_percent=(histogram / histogram.sum() * 100).tolist(),
                             figure=str(output), selected_samples=samples)
        print(f"Saved {output}")
    write_json(args.output_dir / "data_summary.json", {
        "data_root": config["data_root"], "classes": config["classes"],
        "validation_split": "test", "source": "baselines.data.CloudDataset", "splits": report,
    })


if __name__ == "__main__":
    main()
