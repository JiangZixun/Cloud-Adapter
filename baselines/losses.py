"""Pixel CE and multiclass soft Dice shared by FMamba and UNet controls."""

import torch
from torch.nn import functional as F


def native_cafbr_loss(logits, labels, config):
    settings = config.get("loss", {})
    ce_weight = float(settings.get("ce_weight", 5.0))
    dice_weight = float(settings.get("dice_weight", 5.0))
    eps = float(settings.get("dice_eps", 1e-5))
    if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight == 0 or eps <= 0:
        raise ValueError("Loss weights must be nonnegative with a positive sum; dice_eps must be positive")
    valid = labels != config["ignore_index"]
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits.float(), labels, ignore_index=config["ignore_index"])
    probabilities = logits.float().softmax(dim=1)
    targets = labels.masked_fill(~valid, 0)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probabilities = probabilities * valid
    one_hot = one_hot * valid
    # Match the source FMamba Dice: pool batch and spatial dimensions, then
    # average the class Dice coefficients (including clear sky/background).
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = 1 - ((2 * intersection + eps) / (denominator + eps)).mean()
    return ce_weight * ce + dice_weight * dice


def native_lsmamba_loss(logits, labels, config):
    """LS-Mamba CE/Dice mix; the source Dice uses squared probabilities."""
    settings = config["loss"]
    valid = labels != config["ignore_index"]
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits.float(), labels, ignore_index=config["ignore_index"])
    targets = F.one_hot(labels.masked_fill(~valid, 0), logits.shape[1]).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probabilities = logits.float().softmax(1) * valid
    targets = targets * valid
    intersection = (probabilities * targets).sum((0, 2, 3))
    denominator = (probabilities.square() + targets.square()).sum((0, 2, 3))
    eps = settings["dice_eps"]
    dice = 1 - ((2 * intersection + eps) / (denominator + eps)).mean()
    return settings["ce_weight"] * ce + settings["dice_weight"] * dice
