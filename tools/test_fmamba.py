"""Evaluate a CAFBR FMamba or LS-Mamba checkpoint on all HRC-WHU test images."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.checkpoints import write_json
from baselines.data import CloudDataset
from baselines.model_factory import build_model
from baselines.cafbr_schedule import restore_cafbr_state
from tools.train_unet import evaluate
from baselines.tracking import ExperimentHistory, WandbTracker, add_wandb_arguments, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    if args.device != "cuda":
        raise ValueError("The FMamba selective-scan extension requires CUDA")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if not config["model"].startswith(("FMamba_CAFBR", "LSMamba")):
        raise ValueError("Expected a CAFBR FMamba or LS-Mamba checkpoint")
    model = build_model(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    cafbr_state = restore_cafbr_state(model, checkpoint)
    del checkpoint["model"]
    loader = DataLoader(CloudDataset(config, "test"), batch_size=config["batch_size"],
                        num_workers=args.num_workers, pin_memory=True)
    metrics = evaluate(model, loader, torch.device(args.device), config, description="Test")
    output = args.output or args.checkpoint.parent.parent / "test_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(model=config["model"], checkpoint=str(args.checkpoint.resolve()),
                  iteration=checkpoint["iteration"], split="test", executable=sys.executable,
                  gpu=torch.cuda.get_device_name(), **metrics)
    result.update(cafbr_state)
    write_json(output, result)
    history = ExperimentHistory(args.checkpoint.parent.parent, config)
    tracker = WandbTracker(args, history, resume=True)
    row = history.record("test", checkpoint["iteration"], metrics, data_split="test",
                         **cafbr_state,
                         role="standalone evaluation", checkpoint=str(args.checkpoint.resolve()),
                         checkpoint_sha256=file_digest(args.checkpoint))
    history.render()
    tracker.log("test", row)
    tracker.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
