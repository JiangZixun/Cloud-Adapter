"""Native CAFBR FMamba and its official Transformers Mask2Former head variant."""

import torch
from torch import nn
from torch.nn import functional as F

from .fmamba import Fmamba, selective_scan_fn, selective_scan_fn_v1


def native_model(config):
    if selective_scan_fn is None and selective_scan_fn_v1 is None:
        raise RuntimeError("CAFBR FMamba requires a working selective-scan CUDA extension")
    return Fmamba(
        num_classes=len(config["classes"]), in_channels=3,
        scgm_num_groups=config["fmamba"]["scgm_num_groups"],
        skip_refinement=config["fmamba"]["skip_refinement"],
        base_channels=config.get("base_channels", 64),
        downsample_stages=config["fmamba"].get("downsample_stages", 4),
    )


def mask2former_model(config):
    from ..mask2former import mask2former_model as build_head

    def backbone(config):
        model = native_model(config)
        model.final = nn.Identity()
        return model

    if config["fmamba"].get("downsample_stages", 4) != 4:
        raise ValueError("Mask2Former requires four-downsample FMamba")
    base = config.get("base_channels", 64)
    return build_head(config, backbone, [base * 2 ** i for i in range(4)])


# Preserve callers that previously imported semantic inference from this module.
from ..mask2former import semantic_logits
