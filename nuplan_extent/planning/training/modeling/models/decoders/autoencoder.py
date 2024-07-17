
import torch
from torch import nn
from third_party.taming.modules.diffusionmodules.model import Decoder
from typing import List


class AEDecoder(nn.Module):
    def __init__(self,
                 z_channels: int = 512,
                 resolution: int = 224,
                 in_channels: int = 5,
                 out_ch: int = 5,
                 ch: int = 32,
                 # num_down = len(ch_mult)-1
                 ch_mult: List = [1, 1, 2, 2, 4, 8],
                 num_res_blocks: int = 2,
                 attn_resolutions: List = [16],
                 dropout: float = 0.0):
        super().__init__()
        self.decoder = Decoder(
            ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=True, in_channels=in_channels,
            resolution=resolution, z_channels=z_channels, give_pre_end=False
        )

    def forward(self, input):
        x = self.decoder(input['vision_wo_pe'])
        return {
            'reconstruct_image': x
        }
