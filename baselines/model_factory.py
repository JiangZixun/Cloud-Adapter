"""Shared model interface for training and timing HRC-WHU baselines."""

from torch import nn

from .unet import UNet


def build_model(config):
    name = config["model"]
    if name == "UNet":
        return UNet(num_classes=len(config["classes"]), base_channels=config["base_channels"])
    if name in {"LSMamba", "LSMamba_Mask2Former"}:
        from .lsmamba.models import native_model, mask2former_model
        return native_model(config) if name == "LSMamba" else mask2former_model(config)
    from .fmamba_cafbr.models import native_model, mask2former_model
    if name == "FMamba_CAFBR":
        return native_model(config)
    if name == "FMamba_CAFBR_Mask2Former":
        return mask2former_model(config)
    raise ValueError(f"Unknown model: {name}")


def prediction_logits(output, size):
    if isinstance(output, dict):
        from .mask2former import semantic_logits
        return semantic_logits(output, size)
    return output


def training_loss(model, output, labels, config):
    if isinstance(output, dict):
        return model.training_loss(output, labels, config["ignore_index"])
    if config["model"] == "FMamba_CAFBR" or (config["model"] == "UNet" and "loss" in config):
        from .losses import native_cafbr_loss
        return native_cafbr_loss(output, labels, config)
    if config["model"] == "LSMamba":
        from .losses import native_lsmamba_loss
        return native_lsmamba_loss(output, labels, config)
    return nn.functional.cross_entropy(output, labels, ignore_index=config["ignore_index"])
