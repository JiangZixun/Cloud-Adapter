"""Measure both CAFBR phases and extrapolate the full CloudSEN12 protocol."""

import argparse
import json
import math
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.cafbr_schedule import apply_cafbr_schedule, cafbr_start_iteration
from baselines.checkpoints import CheckpointManager, write_json
from baselines.data import CloudDataset, IterationBatchSampler, dataset_splits
from baselines.lr_schedule import lr_multiplier, warmup_steps
from baselines.optimization import accumulated_update, accumulation_steps
from baselines.model_factory import build_model, training_loss
from tools.train_unet import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--eval-samples', type=int, default=64)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.eval_samples) < 1:
        raise ValueError('Benchmark sample counts must be positive')
    config = json.loads(args.config.read_text())
    config["grad_accum_steps"] = accumulation_steps(config)
    config["effective_batch_size"] = config["batch_size"] * config["grad_accum_steps"]
    warmup_steps(config)
    if not config['model'].startswith('FMamba_CAFBR') or config['device'] != 'cuda':
        raise ValueError('This benchmark requires a CUDA CAFBR FMamba configuration')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model = build_model(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, config))
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
    splits = dataset_splits(config)
    datasets = {key: CloudDataset(config, value) for key, value in splits.items()}
    counts = {key: len(value) for key, value in datasets.items()}
    train = datasets['train']
    loader = DataLoader(train, batch_sampler=IterationBatchSampler(len(train), config['batch_size'], 0,
                         2 * (args.warmup + args.steps) * config["grad_accum_steps"], config['seed']),
                        num_workers=config['num_workers'], pin_memory=True)
    iterator = iter(loader)
    start = cafbr_start_iteration(config)
    phases = {}
    for name, step in [('off', max(0, start - 1)), ('on', start)]:
        context = apply_cafbr_schedule(model, config, step)
        timings = []
        for index in range(args.warmup + args.steps):
            torch.cuda.synchronize()
            began = time.perf_counter()
            model.train()
            accumulated_update(model, iterator, optimizer, scaler, device, config, training_loss)
            scheduler.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            if index >= args.warmup:
                timings.append(elapsed)
        evaluation = {}
        for role, source in datasets.items():
            if role == 'train' and config.get('train_loss_only', False):
                continue
            subset = CloudDataset(config, splits[role], args.eval_samples)
            subset.split = 'evaluation'
            evaluation_loader = DataLoader(subset, batch_size=config['batch_size'], num_workers=config['num_workers'],
                                           pin_memory=True, persistent_workers=config['num_workers'] > 0)
            # Start workers/prefetch before timing the repeated evaluation iterator.
            next(iter(evaluation_loader))
            torch.cuda.synchronize()
            began = time.perf_counter()
            metrics = evaluate(model, evaluation_loader, device, config, description=f'{name}: {role}', show_progress=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            evaluation[role] = dict(measured_samples=len(subset), full_samples=len(source), seconds=elapsed,
                                   estimated_full_seconds=elapsed * math.ceil(len(source) / config['batch_size']) /
                                   math.ceil(len(subset) / config['batch_size']))
        phases[name] = dict(**context, measured_steps=args.steps, mean_step_seconds=statistics.mean(timings),
                            median_step_seconds=statistics.median(timings), min_step_seconds=min(timings),
                            max_step_seconds=max(timings), evaluation=evaluation)
        print(f'{config["dataset"]}/{config["model"]}: CAFBR {name}, {statistics.mean(timings):.4f} s/step', flush=True)
    suffix = '_benchmark' + (f'_accum{config["grad_accum_steps"]}' if config['grad_accum_steps'] > 1 else '')
    output_directory = ROOT / 'experiments' / config['dataset'] / (config['model'] + suffix)
    output_directory.mkdir(parents=True, exist_ok=True)
    state = dict(iteration=1, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 scaler=scaler.state_dict(), config=config, metrics=metrics, cafbr_enabled=True)
    saves = []
    with tempfile.TemporaryDirectory(dir=output_directory) as temporary:
        manager = CheckpointManager(temporary, config['keep_top_k'], config['selection_metric'])
        for iteration in range(1, 5):
            state['iteration'] = iteration
            candidate = dict(metrics)
            candidate[config['selection_metric']] -= iteration - 1
            began = time.perf_counter()
            manager.update(state, candidate)
            saves.append(time.perf_counter() - began)
        checkpoint_bytes = (Path(temporary) / 'last.pth').stat().st_size
        began = time.perf_counter()
        best = torch.load(Path(temporary) / 'best.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(best['model'])
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - began
        del best
    iterations = list(range(config['val_interval'], config['max_iters'] + 1, config['val_interval']))
    if not iterations or iterations[-1] != config['max_iters']:
        iterations.append(config['max_iters'])
    off_steps = min(config['max_iters'], max(0, start - 1))
    train_seconds = off_steps * phases['off']['mean_step_seconds'] + (config['max_iters'] - off_steps) * phases['on']['mean_step_seconds']
    roles = ('validation',) if config.get('train_loss_only', False) else ('train', 'validation')
    evaluation_seconds = sum(sum(phases['on' if iteration >= start else 'off']['evaluation'][role]['estimated_full_seconds']
                                 for role in roles) for iteration in iterations)
    final_test_seconds = [phase['evaluation']['test']['estimated_full_seconds'] for phase in phases.values()]
    minimum_io = saves[0] + min(2, len(iterations) - 1) * saves[1] + max(0, len(iterations) - 3) * saves[3]
    maximum_io = len(iterations) * max(saves)
    total = [train_seconds + evaluation_seconds + min(final_test_seconds) + minimum_io + load_seconds,
             train_seconds + evaluation_seconds + max(final_test_seconds) + maximum_io + load_seconds]
    report = dict(config=config, gpu=torch.cuda.get_device_name(), executable=sys.executable, torch=str(torch.__version__),
                  warmup_steps=args.warmup, counts=counts, phases=phases, validation_iterations=iterations, evaluation_roles=roles,
                  estimated_train_seconds=train_seconds, estimated_train_validation_seconds=evaluation_seconds,
                  estimated_final_test_seconds_range=[min(final_test_seconds), max(final_test_seconds)],
                  checkpoint_save_seconds=saves, checkpoint_bytes=checkpoint_bytes, best_load_seconds=load_seconds,
                  peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  estimated_total_seconds_range=total, recommended_hours_range=[total[0] / 3600 * 1.1, total[1] / 3600 * 1.25],
                  methodology='Measured per-phase real-image steps and subset evaluation; extrapolated configured evaluation/test counts; full-train evaluation only when enabled; logging/system variability reserved separately')
    write_json(output_directory / 'timing_estimate.json', report)
    print(f'Estimated full run: {total[0]/3600:.2f}–{total[1]/3600:.2f} hours; only timing JSON retained.', flush=True)


if __name__ == '__main__':
    main()
