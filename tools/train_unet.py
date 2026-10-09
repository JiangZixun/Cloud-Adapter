"""Train cloud segmentation baselines with explicit validation and test splits."""

import argparse
import csv
import json
import os
from contextlib import nullcontext
from datetime import timedelta
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.checkpoints import CheckpointManager, write_json
from baselines.lr_schedule import lr_multiplier, warmup_steps, validate_resume_schedule
from baselines.data import CloudDataset, IterationBatchSampler, dataset_splits
from baselines.metrics import METRIC_NAMES, confusion_matrix, segmentation_metrics
from baselines.model_factory import build_model, prediction_logits, training_loss
from baselines.tracking import ExperimentHistory, WandbTracker, add_wandb_arguments, file_digest
from baselines.run_directory import prepare_fresh_run
from baselines.optimization import accumulation_steps, accumulated_update
from baselines.distributed import verify_parameter_sync
from baselines.cafbr_schedule import apply_cafbr_schedule, cafbr_start_iteration, restore_cafbr_state


def parse_args(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config or ROOT / "configs/unet/hrc_whu.json")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--val-interval", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int, help="Physical batches per optimizer update (default: configuration or 1)")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--warmup-start-factor", type=float)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="5 iterations, 8 train images; optional configured smoke validation/test limits")
    parser.add_argument("--resume", type=Path, help="Resume the same run from its last.pth")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a checkpoint without training")
    add_wandb_arguments(parser)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device, config, description="Validation", show_progress=True):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Point-sampled Mask2Former eval losses are reproducible and do not consume
    # the RNG used by subsequent training/dropout.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["seed"])
        return _evaluate(model, loader, device, config, description, show_progress)


def _evaluate(model, loader, device, config, description, show_progress):
    model.eval()
    matrix = torch.zeros((len(config["classes"]),) * 2, dtype=torch.int64, device=device)
    loss_sum, objective_sum, valid_pixels, samples, data_time = 0.0, 0.0, 0, 0, 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = previous = time.perf_counter()
    for images, labels in tqdm(loader, desc=description, leave=False, dynamic_ncols=True, disable=not show_progress):
        data_time += time.perf_counter() - previous
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        objective_sum += training_loss(model, output, labels, config).item() * len(images)
        logits = prediction_logits(output, labels.shape[-2:])
        loss_sum += nn.functional.cross_entropy(
            logits, labels, ignore_index=config["ignore_index"], reduction="sum"
        ).item()
        valid_pixels += (labels != config["ignore_index"]).sum().item()
        samples += len(images)
        matrix += confusion_matrix(logits.argmax(1), labels, len(config["classes"]), config["ignore_index"])
        previous = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics = segmentation_metrics(matrix, config["classes"])
    metrics.update(val_loss=loss_sum / max(valid_pixels, 1), ce_loss=loss_sum / max(valid_pixels, 1),
                   loss=objective_sum / max(samples, 1), samples=samples,
                   data_time=data_time / len(loader), time=(time.perf_counter() - started) / len(loader))
    return metrics


def append_metrics(directory, iteration, lr, train_loss, metrics, validation_split="test"):
    record = dict(iteration=iteration, lr=lr, train_loss=train_loss,
                  validation_split=validation_split, **metrics)
    with (directory / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = directory / "metrics.csv"
    fields = ["iteration", "lr", "train_loss", "validation_split", *METRIC_NAMES,
              "val_loss", "samples", "data_time", "time"]
    fields += [f"{name}/{metric}" for name in metrics["per_class"] for metric in metrics["per_class"][name]]
    row = {key: record[key] for key in fields if key in record}
    row.update({f"{name}/{key}": value for name, values in metrics["per_class"].items() for key, value in values.items()})
    needs_header = not csv_path.exists()
    with csv_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    write_json(directory / "validation" / f"iter_{iteration:07d}.json", record)


def main(default_config=None):
    args = parse_args(default_config)
    config = json.loads(args.config.read_text())
    if args.resume and args.evaluate:
        raise ValueError("Choose either --resume or --evaluate")
    if args.smoke_test:
        config.update(max_iters=5, val_interval=1, log_interval=1, num_workers=0,
                      train_limit=8, work_dir=f"experiments/{config['dataset']}/{config['model']}_smoke")
        for split in ("val", "test"):
            if f"smoke_{split}_limit" in config:
                config[f"{split}_limit"] = config[f"smoke_{split}_limit"]
        if config["model"] == "UNet":
            config["base_channels"] = 16
    for name in ("data_root", "work_dir", "device", "max_iters", "val_interval", "batch_size", "grad_accum_steps", "num_workers", "lr", "warmup_ratio", "warmup_start_factor"):
        value = getattr(args, name)
        if value is not None:
            config[name] = str(value) if isinstance(value, Path) else value
    config["grad_accum_steps"] = accumulation_steps(config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    primary = rank == 0
    config["world_size"] = world_size
    config["effective_batch_size"] = config["batch_size"] * config["grad_accum_steps"] * world_size
    warmup_steps(config)  # Validate the schedule before creating outputs.
    if not np.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("lr must be finite and positive")
    config["amp"] = config["amp"] or args.amp
    for name in ("max_iters", "val_interval", "log_interval", "batch_size", "keep_top_k", "base_channels"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["num_workers"] < 0 or config["selection_metric"] not in METRIC_NAMES:
        raise ValueError("Invalid worker count or selection metric")
    # Training computes only the optimization loss; metrics belong to validation/test.
    config["train_loss_only"] = True
    directory = Path(config["work_dir"])
    if not directory.is_absolute():
        directory = ROOT / directory
    device = torch.device(config["device"])
    if distributed:
        if device.type != "cuda" or config["model"] != "FMamba_CAFBR":
            raise ValueError("DDP currently supports native FMamba_CAFBR on CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Rank zero evaluates the complete split and writes large checkpoints.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    if config["model"].startswith(("FMamba_CAFBR", "LSMamba")) and device.type != "cuda":
        raise ValueError("Mamba models require CUDA for their selective-scan extension")
    if config["amp"] and device.type != "cuda":
        raise ValueError("AMP requires CUDA in this trainer")
    # Fail before creating output if the selected environment cannot execute CUDA.
    if device.type == "cuda":
        torch.ones(1, device=device).add_(1).cpu()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cudnn.benchmark = False
    model = build_model(config).to(device)
    if config["model"].startswith("FMamba_CAFBR"):
        cafbr_start_iteration(config)  # Validate the schedule before creating outputs.
    checkpoint = None
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location="cpu", weights_only=True)
        for name in ("model", "fmamba", "lsmamba", "mask2former", "loss", "classes", "base_channels", "image_size", "mean", "std", "data_root", "seed", "batch_size", "amp", "train_limit", "val_limit", "test_limit", "validation_split", "test_split"):
            if checkpoint["config"].get(name) != config.get(name):
                raise ValueError(f"Checkpoint configuration mismatch: {name}")
        if args.resume:
            if accumulation_steps(checkpoint["config"]) != config["grad_accum_steps"]:
                raise ValueError("Checkpoint configuration mismatch: grad_accum_steps; start a new run")
            validate_resume_schedule(checkpoint["config"], config)
            if checkpoint["config"].get("world_size", 1) != world_size:
                raise ValueError("Checkpoint configuration mismatch: world_size; start a new run")
        model.load_state_dict(checkpoint["model"])
        restore_cafbr_state(model, checkpoint)
        if args.resume and checkpoint["config"].get("cafbr_start_ratio", 0.0) != config.get("cafbr_start_ratio", 0.0):
            raise ValueError("Checkpoint configuration mismatch: cafbr_start_ratio; start a new run for the delayed schedule")
    splits = dataset_splits(config)
    val_dataset = CloudDataset(config, splits["validation"], config.get("val_limit"))
    val_dataset.split = "evaluation"
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    if args.evaluate:
        test_dataset = CloudDataset(config, splits["test"], config.get("test_limit"))
        test_dataset.split = "evaluation"
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                                 num_workers=config["num_workers"], pin_memory=device.type == "cuda")
        if primary:
            metrics = evaluate(model, test_loader, device, config, description="Test")
            history = ExperimentHistory(directory, config)
            tracker = WandbTracker(args, history, resume=True)
            row = history.record("test", checkpoint["iteration"], metrics,
                                 **restore_cafbr_state(model, checkpoint),
                                 checkpoint=str(args.evaluate.resolve()), checkpoint_sha256=file_digest(args.evaluate),
                                 data_split=splits["test"], role="standalone evaluation")
            history.render()
            tracker.log("test", row)
            tracker.finish()
            print(json.dumps(metrics, indent=2, allow_nan=False))
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    if not args.resume and primary:
        archive = prepare_fresh_run(directory)
        if archive is not None:
            print(f"Previous run has no saved checkpoint; preserved its files at {archive}", flush=True)
    if distributed:
        dist.barrier()
    if args.resume:
        if args.resume.resolve() != (directory / "checkpoints/last.pth").resolve():
            raise ValueError("Resume from this run's checkpoints/last.pth to preserve ranking/history")
        history_path = directory / "metrics.jsonl"
        if not history_path.exists():
            raise ValueError("Resume requires the existing metrics history")
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        if not history or history[-1]["iteration"] != checkpoint["iteration"]:
            raise ValueError("Checkpoint iteration and metrics history do not match")
    train_dataset = CloudDataset(config, "train", config.get("train_limit"))
    start_iter = checkpoint["iteration"] if checkpoint else 0
    if start_iter >= config["max_iters"]:
        raise ValueError("max_iters must be greater than the resumed iteration")
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=IterationBatchSampler(len(train_dataset), config["batch_size"], start_iter * config["grad_accum_steps"],
                                            config["max_iters"] * config["grad_accum_steps"], config["seed"], rank, world_size),
        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, config)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint["config"]["max_iters"] != config["max_iters"]:
            # Explicitly extending a completed run must recompute the current
            # LR: restoring its old final zero LR would skip the next update.
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, scheduler.lr_lambdas, last_epoch=start_iter - 1
            )
            print(f"Updated LR schedule horizon to {config['max_iters']} iterations", flush=True)
        scaler.load_state_dict(checkpoint["scaler"])
        if distributed:
            rng = checkpoint["rank_rng"][rank]
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"], device)
        else:
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if primary:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "validation").mkdir(exist_ok=True)
        write_json(directory / "config.json", config)
        write_json(directory / "environment.json", {
            "python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
            "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "model": config["model"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_samples": len(train_dataset), "validation_samples": len(val_dataset),
            "validation_split": splits["validation"], "checkpoint_selection_split": splits["validation"],
            "test_split": splits["test"],
            "train_loss_only": True,
            "physical_batch_size": config["batch_size"],
            "grad_accum_steps": config["grad_accum_steps"],
            "effective_batch_size": config["effective_batch_size"],
            "iteration_unit": "optimizer update",
            "distributed": distributed, "world_size": world_size, "backend": "nccl" if distributed else None,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        })
        write_json(directory / "data_manifest.json", {
            split: [{"image": str(image), "mask": str(mask)} for image, mask in dataset.pairs]
            for split, dataset in (("train", train_dataset), ("validation", val_dataset),
                                   ("test", CloudDataset(config, splits["test"], config.get("test_limit"))))
        })
        manager = CheckpointManager(directory / "checkpoints", config["keep_top_k"], config["selection_metric"])
        history = ExperimentHistory(directory, config)
        history.attach_files()
        tracker = WandbTracker(args, history, resume=bool(args.resume))
    if distributed:
        dist.barrier()
    training_model = None
    previous_cafbr_enabled = None
    loss_sum, loss_steps = 0.0, 0
    window_loss, window_steps = 0.0, 0
    if primary:
        print(f"{device}: world_size={world_size}, train={len(train_dataset)}, validation({splits['validation']})={len(val_dataset)}, batch={config['batch_size']} x accum={config['grad_accum_steps']} (effective={config['effective_batch_size']}), output={directory}", flush=True)
    with ((directory / "train.log").open("a") if primary else nullcontext()) as log:
        train_iterator = iter(train_loader)
        progress = tqdm(range(start_iter + 1, config["max_iters"] + 1), total=config["max_iters"], initial=start_iter,
                        desc=f"Train {config['model']}", dynamic_ncols=True, disable=not primary)
        for iteration in progress:
            model.train()
            cafbr_state = apply_cafbr_schedule(model, config, iteration)
            if distributed and (training_model is None or previous_cafbr_enabled != cafbr_state.get("cafbr_enabled")):
                # CAFBR changes requires_grad at the phase boundary. Rebuild the
                # reducer so newly enabled parameters join gradient all-reduces.
                training_model = None
                training_model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                                     find_unused_parameters=True)
                previous_cafbr_enabled = cafbr_state.get("cafbr_enabled")
                if iteration == 1 and not checkpoint:
                    torch.manual_seed(config["seed"] + rank)
            lr = optimizer.param_groups[0]["lr"]
            loss_value = accumulated_update(training_model if distributed else model, train_iterator, optimizer, scaler,
                                            device, config, training_loss)
            if distributed:
                global_loss = torch.tensor(loss_value, device=device, dtype=torch.float64)
                dist.all_reduce(global_loss)
                loss_value = global_loss.item() / world_size
            synchronization_digest = None
            if distributed and args.smoke_test:
                synchronization_digest = verify_parameter_sync(model)
            scheduler.step()
            loss_sum += loss_value
            loss_steps += 1
            window_loss += loss_value
            window_steps += 1
            postfix = dict(loss=f"{loss_value:.4f}", lr=f"{lr:.3g}")
            if cafbr_state:
                postfix["CAFBR"] = "on" if cafbr_state["cafbr_enabled"] else "off"
            progress.set_postfix(postfix, refresh=False)
            validation_due = iteration % config["val_interval"] == 0 or iteration == config["max_iters"]
            phase_boundary = cafbr_state and iteration == cafbr_state["cafbr_start_iteration"] - 1
            if primary and (iteration % config["log_interval"] == 0 or validation_due or phase_boundary):
                row = history.record("training", iteration, dict(loss=window_loss / window_steps, lr=lr),
                                     first_iteration=iteration - window_steps + 1,
                                     batches=window_steps * config["grad_accum_steps"] * world_size, optimizer_updates=window_steps,
                                     samples=window_steps * config["effective_batch_size"],
                                     micro_batches_completed=iteration * config["grad_accum_steps"] * world_size,
                                     world_size=world_size, parameter_sync_sha256=synchronization_digest,
                                     grad_accum_steps=config["grad_accum_steps"], effective_batch_size=config["effective_batch_size"],
                                     **cafbr_state,
                                     data_split="train", role="augmented training batches; evolving model weights")
                tracker.log("optimization", row)
                tracker.log("train", row)
                window_loss, window_steps = 0.0, 0
                message = f"iter={iteration} loss={loss_value:.6f} lr={lr:.8g}"
                log.write(message + "\n")
                log.flush()
            if validation_due:
                rank_rng = None
                if distributed:
                    # Rank-zero BatchNorm buffers are the checkpoint authority.
                    for buffer in model.buffers():
                        dist.broadcast(buffer, src=0)
                    rank_rng = [None] * world_size
                    dist.all_gather_object(rank_rng, {"torch": torch.get_rng_state(),
                                                       "cuda": torch.cuda.get_rng_state(device)})
                if primary:
                    metrics = evaluate(model, val_loader, device, config)
                    append_metrics(directory, iteration, lr, loss_sum / loss_steps, metrics, splits["validation"])
                    checkpoint_path = str(directory / "checkpoints" / f"iter_{iteration:07d}.pth")
                    val_row = history.record("validation", iteration, metrics, data_split=splits["validation"],
                                             **cafbr_state,
                                             role="checkpoint selection", checkpoint=checkpoint_path)
                    tracker.log("validation", val_row)
                    history.render()
                    state = dict(iteration=iteration, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 **cafbr_state,
                                 scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), config=config,
                                 metrics=metrics, torch_rng=torch.get_rng_state(),
                                 cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                                 rank_rng=rank_rng)
                    manager.update(state, metrics)
                    tracker.update_checkpoints(manager.top, iteration)
                    message = f"validation iter={iteration} loss={metrics['loss']:.6f} " + " ".join(f"{key}={metrics[key]}" for key in METRIC_NAMES)
                    tqdm.write(message)
                    log.write(message + "\n")
                    log.flush()
                if distributed:
                    dist.barrier()
                loss_sum, loss_steps = 0.0, 0
    if not primary:
        dist.barrier()
        dist.destroy_process_group()
        return
    best = torch.load(directory / "checkpoints/best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    best_cafbr_state = restore_cafbr_state(model, best)
    test_dataset = CloudDataset(config, splits["test"], config.get("test_limit"))
    test_dataset.split = "evaluation"
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    best_metrics = evaluate(model, test_loader, device, config, description="Test best checkpoint")
    write_json(directory / "best_metrics.json", {
        "iteration": best["iteration"], "checkpoint": "checkpoints/best.pth",
        "validation_split": splits["validation"], "test_split": splits["test"], **best_metrics,
        **best_cafbr_state,
    })
    test_row = history.record("test", best["iteration"], best_metrics, data_split=splits["test"],
                              **best_cafbr_state,
                              role="final best checkpoint evaluation", checkpoint="checkpoints/best.pth",
                              checkpoint_sha256=file_digest(directory / "checkpoints/best.pth"))
    history.data["checkpoint_ranking"] = manager.top
    history.save()
    history.render()
    tracker.log("test", test_row)
    tracker.finish()
    print(f"Finished. Best iteration={best['iteration']}, validation mIoU={best['metrics']['mIoU']:.4f}, test mIoU={best_metrics['mIoU']:.4f}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
