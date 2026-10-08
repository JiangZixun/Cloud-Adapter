"""One optimizer update from a fixed number of physical mini-batches."""

import torch


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
    for _ in range(steps):
        images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=config["amp"]):
            output = model(images)
            loss = loss_function(model, output, labels, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss during gradient accumulation")
        total_loss += loss.detach().item()
        scaler.scale(loss / steps).backward()
        # Do not retain the previous forward graph while loading the next batch.
        del output, loss, images, labels
    scaler.step(optimizer)
    scaler.update()
    return total_loss / steps
