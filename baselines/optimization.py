"""One optimizer update from a fixed number of physical mini-batches."""

from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel


def accumulation_steps(config):
    steps = config.get("grad_accum_steps", 1)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("grad_accum_steps must be a positive integer")
    return steps


def accumulated_update(model, iterator, optimizer, scaler, device, config, loss_function):
    """Average micro-batch gradients; return the unscaled mean objective loss."""
    steps = accumulation_steps(config)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for micro_step in range(steps):
        images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        synchronization = model.no_sync() if isinstance(model, DistributedDataParallel) and micro_step < steps - 1 else nullcontext()
        with synchronization:
            with torch.autocast(device_type=device.type, enabled=config["amp"]):
                output = model(images)
                loss_model = model.module if isinstance(model, DistributedDataParallel) else model
                loss = loss_function(loss_model, output, labels, config)
            finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
            if isinstance(model, DistributedDataParallel):
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite training loss during gradient accumulation")
            total_loss += loss.detach().item()
            scaler.scale(loss / steps).backward()
        # Do not retain the previous forward graph while loading the next batch.
        del output, loss, images, labels
    scaler.step(optimizer)
    scaler.update()
    return total_loss / steps
