from .lvmamba2 import VSSM
import torch
from torch import nn


class Local_VMUNet2(nn.Module):
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 2, 2], 
                 depths_decoder=[2, 2, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    def forward(self, x, return_pyramid=False):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x, return_pyramid=return_pyramid)
        if return_pyramid:
            return logits
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
