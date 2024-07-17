from typing import List

import timm
import torch.nn as nn
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

feature_info = {}


class RasterEncoder(nn.Module):
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
            out_indices: List[int] = [0, 1, 2, 3, 4],
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
        global feature_info
        feature_info = self._backbone.feature_info.get_dicts()

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
        :return: encoder_features: dict of predictions from network
        """
        raster: Raster = features["raster"]
        encoder_features = self._backbone.forward(
            raster.data[:, self._input_channel_indexes])

        if "command" in features.keys():
            command = features["command"]
            return {"encoder_features": encoder_features, "command": command}
        else:
            return {"encoder_features": encoder_features}
