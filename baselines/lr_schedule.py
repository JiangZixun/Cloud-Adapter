"""Linear warmup followed by cosine or legacy polynomial LR decay."""

import math


def warmup_steps(config):
    ratio = float(config.get("warmup_ratio", 0.0))
    factor = float(config.get("warmup_start_factor", 0.01))
    schedule = config.get("lr_schedule", "poly")
    if schedule not in {"poly", "cosine"}:
        raise ValueError("lr_schedule must be poly or cosine")
    minimum = float(config.get("min_lr_ratio", 0.0))
    if not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("min_lr_ratio must be finite and in [0, 1]")
    power = float(config.get("poly_power", 0.9))
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("warmup_ratio must be finite and in [0, 1)")
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("warmup_start_factor must be finite and in (0, 1]")
    if schedule == "poly" and (not math.isfinite(power) or power <= 0):
        raise ValueError("poly_power must be finite and positive")
    maximum = config["max_iters"]
    if maximum <= 0:
        raise ValueError("max_iters must be positive")
    steps = math.ceil(maximum * ratio)
    if steps >= maximum:
        raise ValueError("Warmup must leave at least one decay step")
    return steps


def lr_multiplier(step, config):
    """step counts completed optimizer updates; step=0 sets the initial LR."""
    warmup = warmup_steps(config)
    if warmup and step < warmup:
        factor = float(config.get("warmup_start_factor", 0.01))
        return factor + (1 - factor) * step / warmup
    progress = (step - warmup) / (config["max_iters"] - warmup)
    if config.get("lr_schedule", "poly") == "cosine":
        minimum = float(config.get("min_lr_ratio", 0.0))
        progress = min(1.0, max(0.0, progress))
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    return max(0, 1 - progress) ** config.get("poly_power", 0.9)


def validate_resume_schedule(saved, current):
    """Do not silently replace a checkpoint's optimizer or LR schedule."""
    for name, default in (("lr_schedule", "poly"), ("min_lr_ratio", 0.0),
                          ("warmup_ratio", 0.0), ("warmup_start_factor", 0.01),
                          ("lr", None), ("weight_decay", None)):
        if saved.get(name, default) != current.get(name, default):
            raise ValueError(f"Checkpoint configuration mismatch: {name}; start a new tuning run")
    if current.get("lr_schedule", "poly") == "poly":
        if saved.get("poly_power", 0.9) != current.get("poly_power", 0.9):
            raise ValueError("Checkpoint configuration mismatch: poly_power; start a new tuning run")
