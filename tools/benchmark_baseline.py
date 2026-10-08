"""Measure full-config GPU train/validation/checkpoint cost and extrapolate a run."""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.checkpoints import CheckpointManager, write_json
from baselines.data import CloudDataset, IterationBatchSampler
from baselines.model_factory import build_model, training_loss
from baselines.lr_schedule import lr_multiplier
from tools.train_unet import evaluate
from baselines.optimization import accumulation_steps, accumulated_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fmamba/hrc_whu_native.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1:
        raise ValueError("Warmup and measured steps must be positive")
    config = json.loads(args.config.read_text())
    config["grad_accum_steps"] = accumulation_steps(config)
    config["effective_batch_size"] = config["batch_size"] * config["grad_accum_steps"]
    device = torch.device(config["device"])
    if device.type != "cuda":
        raise ValueError("This benchmark requires CUDA")
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, config)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    train = CloudDataset(config, "train")
    train_loader = DataLoader(
        train, batch_sampler=IterationBatchSampler(len(train), config["batch_size"], 0,
                                                  (args.warmup + args.steps) * config["grad_accum_steps"], config["seed"]),
        num_workers=config["num_workers"], pin_memory=True,
    )
    iterator = iter(train_loader)
    timings = []
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.train()
        accumulated_update(model, iterator, optimizer, scaler, device, config, training_loss)
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= args.warmup:
            timings.append(elapsed)
        if (step + 1) % 10 == 0:
            print(f"Measured {step + 1}/{args.warmup + args.steps} steps", flush=True)
    validation = DataLoader(CloudDataset(config, "test"), batch_size=config["batch_size"],
                            num_workers=config["num_workers"], pin_memory=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = evaluate(model, validation, device, config)
    torch.cuda.synchronize()
    val_seconds = time.perf_counter() - started
    suffix = "_benchmark" + (f"_accum{config['grad_accum_steps']}" if config["grad_accum_steps"] > 1 else "")
    output = ROOT / "experiments" / config["dataset"] / (config["model"] + suffix)
    output.mkdir(parents=True, exist_ok=True)
    state = dict(iteration=1, config=config, model=model.state_dict(), metrics=metrics,
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
    # Same filesystem and actual production state; temporary benchmark checkpoints
    # are automatically cleaned, leaving only the timing report.
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        manager = CheckpointManager(temporary, config["keep_top_k"], config["selection_metric"])
        started = time.perf_counter()
        manager.update(state, metrics)
        best_save_seconds = time.perf_counter() - started
        state["iteration"] = 2
        lower = dict(metrics)
        lower[config["selection_metric"]] -= 1
        started = time.perf_counter()
        manager.update(state, lower)
        ranked_save_seconds = time.perf_counter() - started
        state["iteration"] = 3
        lower[config["selection_metric"]] -= 1
        manager.update(state, lower)
        state["iteration"] = 4
        lower[config["selection_metric"]] -= 1
        started = time.perf_counter()
        manager.update(state, lower)
        last_save_seconds = time.perf_counter() - started
        started = time.perf_counter()
        best = torch.load(Path(temporary) / "best.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model"])
        torch.cuda.synchronize()
        best_load_seconds = time.perf_counter() - started
        checkpoint_bytes = (Path(temporary) / "last.pth").stat().st_size
    validations = (config["max_iters"] + config["val_interval"] - 1) // config["val_interval"]
    train_seconds = statistics.mean(timings) * config["max_iters"]
    validation_seconds = val_seconds * (validations + 1)
    # Lower bound: first best, two more ranked, remaining last-only. Upper bound:
    # every validation establishes a new best (last + ranked + best copy).
    minimum_io = best_save_seconds + min(2, validations - 1) * ranked_save_seconds + max(0, validations - 3) * last_save_seconds
    maximum_io = validations * best_save_seconds
    report = dict(
        model=config["model"], config=config, executable=sys.executable,
        torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        parameters=sum(p.numel() for p in model.parameters()), warmup_steps=args.warmup,
        measured_steps=args.steps, mean_train_step_seconds=statistics.mean(timings),
        median_train_step_seconds=statistics.median(timings), min_train_step_seconds=min(timings),
        max_train_step_seconds=max(timings), full_validation_seconds=val_seconds,
        validation_samples=metrics["samples"], scheduled_validations=validations,
        checkpoint_bytes=checkpoint_bytes, new_best_save_seconds=best_save_seconds,
        ranked_save_seconds=ranked_save_seconds, last_only_save_seconds=last_save_seconds,
        best_load_seconds=best_load_seconds, estimated_train_seconds=train_seconds,
        estimated_validation_seconds=validation_seconds,
        estimated_total_seconds_range=[train_seconds + validation_seconds + minimum_io + best_load_seconds,
                                       train_seconds + validation_seconds + maximum_io + best_load_seconds],
        peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
    )
    write_json(output / "timing_estimate.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
