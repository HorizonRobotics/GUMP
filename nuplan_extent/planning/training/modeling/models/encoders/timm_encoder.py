from typing import List

import timm
import torch
import torch.nn as nn
import timm
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from einops import rearrange


class TimmEncoder(nn.Module):
    """
    Wrapper around raster-based CNN model that consumes ego, agent and map data in rasterized format
    and regresses ego's future trajectory.
    """

    def __init__(
        self,
        backbone_name: str = 'resnet18',
        input_channel_indexes: List[int] = [0, 1, 2, 3, 4, 5],
        pretrained: bool = True,
        output_stride: int = 32,
        with_pos_embed: bool = True,
        out_dim: int = 512,
        out_indices: List[int] = [0, 1, 2, 3, 4],
        nsize: int = 49,
        n_embed: int = 768,
        ae_enabled: bool = False,
        rl_training: bool = False,
    ):
        """
        :param backbone_name: name of the backbone used for encoding
        :param input_channel_indexes: list of input channel indexes
        :param pretrained: whether to use pretrained weights for backbone
        :param output_stride: max output stride
        :param out_indices: list of selected output layer index
        """
        super().__init__()
        self._input_dim = len(input_channel_indexes)
        self._input_channel_indexes = input_channel_indexes
        self._backbone = timm.create_model(
            backbone_name,
            in_chans=self._input_dim,
            pretrained=pretrained,
            output_stride=output_stride,
            features_only=True,
            out_indices=out_indices)
        self.adaptor = nn.Sequential(
            nn.Conv2d(
                out_dim,
                n_embed,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(n_embed),
            nn.LeakyReLU()
        )
        self.rl_training = rl_training
        self.log_data = {}

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
        :return: encoder_features: dict of predictions from network
        """

        if self.rl_training:
            raster = features["raster_processed"]
            if raster.dtype == torch.uint8:
                raster = raster.to(self.adaptor[0].weight.dtype) / 255.0
        else:
            raster = features["raster"].data.to(self.adaptor[0].weight.dtype)
        
        speed = raster[:, 5:6, 0, 0]

        ret_dict = {}

        vision_wo_pe = self._backbone.forward(raster[:, self._input_channel_indexes])[-1]  # [B, 512, 7, 7]
        vision_x = self.adaptor(vision_wo_pe)
        vision_x = rearrange(vision_x, 'b c h w -> b (h w) c')
        ret_dict.update({'vision_x': vision_x, 'speed': speed})
        return ret_dict
