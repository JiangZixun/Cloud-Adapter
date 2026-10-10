"""Delay CAFBR skip refinement and restore the exact checkpoint inference path."""

import math


def cafbr_start_iteration(config):
    ratio = config.get("cafbr_start_ratio", 0.0)  # Old checkpoints used CAFBR throughout.
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("cafbr_start_ratio must be between 0 and 1")
    if "max_epochs" in config:
        return int(config["max_epochs"] * ratio) * config["steps_per_epoch"] + 1
    return int(config["max_iters"] * ratio) + 1


def apply_cafbr_schedule(model, config, iteration, *, enabled=None):
    if not config["model"].startswith("FMamba_CAFBR"):
        return {}
    backbone = model.backbone if hasattr(model, "backbone") else model
    start = cafbr_start_iteration(config)
    active = iteration >= start if enabled is None else bool(enabled)
    active = active and backbone.skip_refinement_enabled
    if backbone.skip_refinement_active != active:
        backbone.set_skip_refinement_active(active)
        for parameter in backbone.skip_refiners.parameters():
            parameter.requires_grad_(active)
            if not active:
                parameter.grad = None
    return dict(cafbr_enabled=active, cafbr_start_iteration=start)


def restore_cafbr_state(model, checkpoint):
    return apply_cafbr_schedule(model, checkpoint["config"], checkpoint["iteration"],
                                enabled=checkpoint.get("cafbr_enabled"))
