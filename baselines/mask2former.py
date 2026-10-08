"""Shared official Transformers Mask2Former head for segmentation backbones."""

import torch
from torch import nn
from torch.nn import functional as F


def mask2former_model(config, backbone_builder, feature_channels):
    # Native segmentation backbones do not depend on Transformers.
    from transformers import Mask2FormerConfig
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerPixelDecoder, Mask2FormerTransformerModule, Mask2FormerLoss,
        Mask2FormerPreTrainedModel,
    )

    class BackboneMask2Former(Mask2FormerPreTrainedModel):
        def __init__(self):
            head_config = Mask2FormerConfig(
                num_labels=len(config["classes"]), feature_strides=[4, 8, 16, 32],
                common_stride=4, **config["mask2former"],
            )
            super().__init__(head_config)
            self.pixel_decoder = Mask2FormerPixelDecoder(head_config, feature_channels)
            self.transformer_module = Mask2FormerTransformerModule(head_config.feature_size, head_config)
            self.class_predictor = nn.Linear(head_config.hidden_dim, head_config.num_labels + 1)
            self.weight_dict = dict(loss_cross_entropy=head_config.class_weight,
                                    loss_mask=head_config.mask_weight, loss_dice=head_config.dice_weight)
            self.criterion = Mask2FormerLoss(head_config, self.weight_dict)
            # Initialize the official head before constructing the native backbone.
            self.post_init()
            self.backbone = backbone_builder(config)

        def forward(self, images):
            features = self.backbone(images, return_pyramid=True)
            pixel = self.pixel_decoder(features, return_dict=True)
            decoder = self.transformer_module(
                list(pixel.multi_scale_features), pixel.mask_features,
                output_hidden_states=self.config.use_auxiliary_loss,
            )
            classes = [self.class_predictor(hidden.transpose(0, 1))
                       for hidden in decoder.intermediate_hidden_states]
            masks = decoder.masks_queries_logits
            return {"class_queries_logits": classes[-1], "masks_queries_logits": masks[-1],
                    "auxiliary_predictions": [
                        dict(class_queries_logits=cls, masks_queries_logits=mask)
                        for cls, mask in zip(classes[:-1], masks[:-1])
                    ] if self.config.use_auxiliary_loss else None}

        def training_loss(self, output, labels, ignore_index):
            if (labels == ignore_index).any():
                raise ValueError("This Mask2Former target adapter expects fully annotated HRC-WHU masks")
            class_labels = [torch.unique(label) for label in labels]
            mask_labels = [(label[None] == classes[:, None, None]).float()
                           for label, classes in zip(labels, class_labels)]
            losses = self.criterion(
                masks_queries_logits=output["masks_queries_logits"],
                class_queries_logits=output["class_queries_logits"],
                mask_labels=mask_labels, class_labels=class_labels,
                auxiliary_predictions=output["auxiliary_predictions"],
            )
            # Official loss weighting includes auxiliary decoder predictions.
            return sum(value * next(weight for name, weight in self.weight_dict.items() if name in key)
                       for key, value in losses.items())

    return BackboneMask2Former()


def semantic_logits(output, size):
    if isinstance(output, torch.Tensor):
        return output
    classes = output["class_queries_logits"].float().softmax(-1)[..., :-1]
    masks = F.interpolate(output["masks_queries_logits"].float(), size=size,
                          mode="bilinear", align_corners=False).sigmoid()
    scores = torch.einsum("bqc,bqhw->bchw", classes, masks)
    # Log scores retain official semantic argmax and permit pixel CE for validation.
    return scores.clamp_min(1e-8).log()
