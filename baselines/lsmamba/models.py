"""LS-Mamba with its original decoder or the shared official Mask2Former head."""

from torch import nn

from .localvmunet2 import Local_VMUNet2


def native_model(config):
    return Local_VMUNet2(input_channels=3, num_classes=len(config["classes"]), **config["lsmamba"])


def mask2former_model(config):
    from ..mask2former import mask2former_model as build_head

    def backbone(config):
        model = native_model(config)
        model.vmunet.final_up = nn.Identity()
        model.vmunet.final_conv = nn.Identity()
        return model

    # Four decoded stages at strides 4/8/16/32 retain every hybrid branch.
    return build_head(config, backbone, [96, 192, 384, 768])
