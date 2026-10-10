"""Measure real-data DDP updates, memory, evaluation and full-run duration."""

import argparse
import gc
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.cafbr_schedule import apply_cafbr_schedule, cafbr_start_iteration
from baselines.checkpoints import CheckpointManager, write_json
from baselines.data import CloudDataset, IterationBatchSampler, dataset_splits
from baselines.distributed import verify_parameter_sync
from baselines.model_factory import build_model, training_loss
from baselines.optimization import accumulated_update
from tools.train_unet import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--eval-samples', type=int, default=40)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.eval_samples) < 1:
        raise ValueError('Benchmark counts must be positive')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    config = json.loads(args.config.read_text())
    config['world_size'] = world
    config['effective_batch_size'] = config['batch_size'] * config['grad_accum_steps'] * world
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    splits = dataset_splits(config)
    train = CloudDataset(config, 'train')
    updates = 2 * (args.warmup + args.steps) * config['grad_accum_steps']
    loader = DataLoader(train, batch_sampler=IterationBatchSampler(
        len(train), config['batch_size'], 0, updates, config['seed'], rank, world),
        num_workers=config['num_workers'], pin_memory=True)
    iterator = iter(loader)
    start = cafbr_start_iteration(config)
    phases, wrapper = {}, None
    for phase, iteration in [('off', max(0, start - 1)), ('on', start)]:
        wrapper = None
        gc.collect()
        state = apply_cafbr_schedule(model, config, iteration)
        wrapper = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        torch.cuda.reset_peak_memory_stats(device)
        timings, losses = [], []
        for step in range(args.warmup + args.steps):
            model.train()
            dist.barrier()
            torch.cuda.synchronize(device)
            began = time.perf_counter()
            loss = accumulated_update(wrapper, iterator, optimizer, scaler, device, config, training_loss)
            global_loss = torch.tensor(loss, device=device, dtype=torch.float64)
            dist.all_reduce(global_loss)
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - began, device=device, dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if step >= args.warmup:
                timings.append(elapsed.item())
                losses.append(global_loss.item() / world)
        digest = verify_parameter_sync(model)
        memory = dict(rank=rank, allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                      reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                      total_gib=torch.cuda.get_device_properties(device).total_memory / 1024**3)
        memories = [None] * world
        dist.all_gather_object(memories, memory)
        phases[phase] = dict(**state, mean_update_seconds=statistics.mean(timings),
                             median_update_seconds=statistics.median(timings),
                             update_seconds=timings, measured_updates=args.steps,
                             mean_loss=statistics.mean(losses), parameter_sync_sha256=digest,
                             training_peak_memory_by_rank=memories)
        # Match the trainer's rank-zero evaluation of the unwrapped model.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
        if rank == 0:
            subset = CloudDataset(config, splits['validation'], args.eval_samples)
            subset.split = 'evaluation'
            val_loader = DataLoader(subset, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                    pin_memory=True, persistent_workers=config['num_workers'] > 0)
            next(iter(val_loader))
            torch.cuda.synchronize(device)
            began = time.perf_counter()
            metrics = evaluate(model, val_loader, device, config, show_progress=False)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - began
            count = len(CloudDataset(config, splits['validation']))
            estimate = elapsed * math.ceil(count / config['batch_size']) / math.ceil(len(subset) / config['batch_size'])
            phases[phase].update(measured_validation_samples=len(subset), validation_samples=count,
                                 validation_seconds=elapsed, full_validation_seconds=estimate)
            del val_loader
            print(f"{config['dataset']} base{config['base_channels']} CAFBR {phase}: "
                  f"{statistics.mean(timings):.3f}s/update, peak "
                  f"{max(m['allocated_gib'] for m in memories):.2f}GiB", flush=True)
        dist.barrier()
    if rank == 0:
        output = Path(config['work_dir'] + '_benchmark')
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(iteration=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), config=config, metrics=metrics, cafbr_enabled=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            manager = CheckpointManager(temporary, config['keep_top_k'], config['selection_metric'])
            began = time.perf_counter()
            manager.update(checkpoint, metrics)
            save_seconds = time.perf_counter() - began
            checkpoint_bytes = (Path(temporary) / 'last.pth').stat().st_size
            began = time.perf_counter()
            restored = torch.load(Path(temporary) / 'best.pth', map_location='cpu', weights_only=True)
            model.load_state_dict(restored['model'], strict=True)
            torch.cuda.synchronize(device)
            load_seconds = time.perf_counter() - began
        off_updates = min(config['max_iters'], max(0, start - 1))
        train_seconds = (off_updates * phases['off']['mean_update_seconds'] +
                         (config['max_iters'] - off_updates) * phases['on']['mean_update_seconds'])
        validation_iterations = list(range(config['val_interval'], config['max_iters'] + 1, config['val_interval']))
        if not validation_iterations or validation_iterations[-1] != config['max_iters']:
            validation_iterations.append(config['max_iters'])
        eval_seconds = sum(phases['on' if i >= start else 'off']['full_validation_seconds']
                           for i in validation_iterations)
        totals = [(train_seconds + eval_seconds + phases[phase]['full_validation_seconds'] +
                   len(validation_iterations) * save_seconds + load_seconds) / 3600 for phase in ('off', 'on')]
        report = dict(status='passed', config=config, python=sys.executable, torch=str(torch.__version__),
                      gpu_names=[torch.cuda.get_device_name(i) for i in range(world)],
                      parameters=sum(p.numel() for p in model.parameters()), train_samples=len(train),
                      warmup_updates_per_phase=args.warmup, phases=phases,
                      validation_iterations=validation_iterations, checkpoint_save_seconds=save_seconds,
                      checkpoint_load_seconds=load_seconds, checkpoint_bytes=checkpoint_bytes,
                      estimated_train_hours=train_seconds / 3600,
                      estimated_validation_hours=eval_seconds / 3600,
                      estimated_total_hours_range=[min(totals), max(totals)],
                      recommended_hours_range=[min(totals) * 1.1, max(totals) * 1.25],
                      methodology='Real 512x512 PNG training on two DDP ranks; max-rank update timings '
                      'after warmup, CAFBR off/on, finite losses and exact parameter-sync checks. '
                      'Rank-zero subset validation extrapolated by batch count; includes periodic validation, '
                      'one final best evaluation and worst-case new-best checkpoint saving each time. '
                      'Recommended range adds 10-25 percent for system/logging variation. '
                      'Peak training memory includes warmup, AdamW states and DDP buffers.')
        write_json(output / 'timing_estimate.json', report)
        print(f"{config['dataset']} base{config['base_channels']}: PASS; "
              f"recommended {report['recommended_hours_range']} hours", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
