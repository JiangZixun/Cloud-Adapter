"""Preview CloudSEN12 L1C/L2A and GF1/GF2 through the shared PNG dataset loader."""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.checkpoints import write_json
from baselines.data import CloudDataset
from tools.visualize_hrc_whu import denormalize, font

DATASETS = {
    "cloudsen12_high_l1c": ("CloudSEN12_L1C", 512),
    "cloudsen12_high_l2a": ("CloudSEN12_L2A", 512),
    "gf12ms_whu_gf1": ("GF1", 256),
    "gf12ms_whu_gf2": ("GF2", 256),
}


def class_metadata(name):
    # Read actual dataset metadata without importing the MMSeg dependency stack.
    source = ROOT / "cloud_adapter/datasets" / f"{name}.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "METAINFO" for target in node.targets):
            return {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.value.keywords}
    raise ValueError(f"No METAINFO found in {source}")


def select_samples(dataset, count, seed, num_classes):
    rng = np.random.default_rng(seed)
    pool = np.sort(rng.choice(len(dataset), size=min(96, len(dataset)), replace=False))
    fractions = []
    for index in pool:
        with Image.open(dataset.pairs[index][1]) as source:
            labels = np.array(source)
        if labels.ndim != 2 or not np.isin(labels, list(range(num_classes)) + [255]).all():
            raise ValueError(f"Invalid mask: {dataset.pairs[index][1]}")
        fractions.append([(labels == label).mean() for label in range(num_classes)])
    fractions = np.array(fractions)
    selected = []
    if num_classes == 4:
        # Include thick cloud, thin cloud, shadow and clear-dominated examples.
        for label in [1, 2, 3, 0]:
            if len(selected) >= min(count, len(pool)):
                break
            ranked = np.argsort(-fractions[:, label], kind="stable")
            selected.append(next(int(row) for row in ranked if int(row) not in selected))
    else:
        ordered = np.argsort(fractions[:, 1], kind="stable")
        selected = list(dict.fromkeys(int(ordered[position]) for position in
                                     np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)))
    remaining = [int(row) for row in np.argsort(fractions[:, 0], kind="stable") if int(row) not in selected]
    selected += remaining
    return [int(pool[row]) for row in selected[:count]]


def panels(rgb, labels, palette):
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        colored[labels == label] = color
    colored[labels == 255] = [255, 0, 0]
    overlaid = rgb.copy()
    valid = labels != 0
    overlaid[valid] = np.rint(rgb[valid] * 0.55 + colored[valid] * 0.45).astype(np.uint8)
    return [rgb, colored, overlaid]


def save_grid(dataset, config, name, split, selected, palette, output):
    tile, gap, margin, header, row_height = 256, 12, 16, 110, 312
    canvas = Image.new("RGB", (824, header + row_height * len(selected) + 12), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), f"{name} / original {split} ({len(dataset)} images)", font=font(21), fill="#152030")
    x = margin
    for label, (class_name, color) in enumerate(zip(config["classes"], palette)):
        draw.rectangle((x, 45, x + 14, 59), fill=tuple(color), outline="#333333")
        legend = f"{label}: {class_name}"
        draw.text((x + 20, 43), legend, font=font(13), fill="#152030")
        x += 20 + int(draw.textlength(legend, font=font(13))) + 20
    draw.text((margin, 67), "Ground-truth labels; display colors only. No training augmentation.", font=font(13), fill="#253040")
    for col, title in enumerate(["RGB image", "Ground-truth mask", "Label overlay"]):
        draw.text((margin + col * (tile + gap), 87), title, font=font(15), fill="#152030")
    records = []
    for row, index in enumerate(selected):
        # Original train split is previewed without augmentation; the loader
        # otherwise performs its actual RGB conversion/normalization and resize.
        image, target = dataset[index]
        rgb, labels = denormalize(image, config), target.numpy()
        image_path, mask_path = dataset.pairs[index]
        with Image.open(image_path) as source:
            original_size = list(source.size)
        shares = {class_name: float((labels == label).mean() * 100)
                  for label, class_name in enumerate(config["classes"])}
        y = header + row * row_height
        draw.text((margin, y), f"{image_path.name} | original {original_size[0]} x {original_size[1]}", font=font(14), fill="#152030")
        caption = " | ".join(f"{key}: {value:.1f}%" for key, value in shares.items())
        draw.text((margin, y + 19), caption, font=font(12), fill="#253040")
        images = panels(rgb, labels, palette)
        for col, pixels in enumerate(images):
            panel = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.NEAREST if col == 1 else Image.Resampling.BILINEAR)
            canvas.paste(panel, (margin + col * (tile + gap), y + 42))
        # Also retain native-resolution panels for detailed inspection.
        detail = Image.new("RGB", (rgb.shape[1] * 3, rgb.shape[0]))
        for col, pixels in enumerate(images):
            detail.paste(Image.fromarray(pixels), (col * rgb.shape[1], 0))
        detail_path = output.parent / "samples" / split / f"{image_path.stem}.png"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail.save(detail_path)
        records.append(dict(image_path=str(image_path), mask_path=str(mask_path), original_size=original_size,
                            labels=np.unique(labels).tolist(), class_percent=shares, detail_figure=str(detail_path)))
    canvas.save(output)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data1/Dataset/Cloud-Adapter"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    base = json.loads((ROOT / "configs/unet/hrc_whu.json").read_text())
    for dataset_name in args.datasets:
        name, size = DATASETS[dataset_name]
        meta = class_metadata(dataset_name)
        palette = [[0, 0, 0], [255, 190, 0], [0, 200, 255], [175, 90, 225]][:len(meta["classes"])]
        config = dict(base, data_root=str(args.data_root / dataset_name), classes=list(meta["classes"]), image_size=[size, size])
        output_dir = ROOT / "experiments" / name / "data_preview/visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        report = dict(dataset=dataset_name, data_root=config["data_root"], classes=config["classes"],
                      source_palette=meta["palette"], display_palette=palette, seed=args.seed,
                      selection="Representative class coverage from a seeded pool of up to 96 masks per split",
                      augmentation=False, loader="baselines.data.CloudDataset", splits={})
        splits = [split for split in ["train", "val", "test"] if (Path(config["data_root"]) / "img_dir" / split).is_dir()]
        for split in splits:
            dataset = CloudDataset(config, split)
            indices = select_samples(dataset, min(args.samples, len(dataset)), args.seed, len(config["classes"]))
            dataset.split = "preview"  # Keep original paths, bypass random train crop/flip/jitter.
            figure = output_dir / f"{split}_samples.png"
            records = save_grid(dataset, config, name, split, indices, palette, figure)
            report["splits"][split] = dict(count=len(dataset), figure=str(figure), samples=records)
            print(f"Saved {figure}", flush=True)
        write_json(output_dir / "data_summary.json", report)


if __name__ == "__main__":
    main()
