from typing import List

import timm
import torch
import torch.nn as nn
from einops import rearrange

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_raster_feature_builder_v2 import RASTER_NAME_TO_CHANNELS, HorizonRasterV2


class Uint8ToFloat32(nn.Module):
    def forward(self, input):
        return input.float() / 255.0 if input.dtype == torch.uint8 else input


def lean_model(model: timm.models.FeatureDictNet):
    from alf.utils.lean_function import lean_function
    assert isinstance(model, timm.models.FeatureDictNet)

    modules = torch.nn.ModuleDict(
        dict(
            pre=lean_function(
                nn.Sequential(
                    Uint8ToFloat32(),
                    model['conv1'],
                    model['bn1'],
                    model['act1'],
                    model['maxpool'],
                )),
            layer1=lean_function(model['layer1']),
            layer2=lean_function(model['layer2']),
            layer3=lean_function(model['layer3']),
            layer4=lean_function(model['layer4']),
        ))
    feature_dicts = model.feature_info.get_dicts()
    modules.feature_info = timm.models.FeatureInfo(
        feature_dicts, list(range(len(feature_dicts))))
    return type(model)(
        modules,
        out_indices=list(range(len(feature_dicts))),
        output_fmt=model.output_fmt,
        feature_concat=model.concat,
        flatten_sequential=False)

def to_device(model, device):
    model.to(device)
    for child in model.children():
        if hasattr(child, 'running_mean'):
            child.running_mean = child.running_mean.to(device)
            child.running_var = child.running_var.to(device)
        to_device(child, device)  # Recursively apply to child submodules

class RasterEncoderV2(nn.Module):
    """
    Wrapper around raster-based CNN model that consumes ego, agent and map data in rasterized format
    and regresses ego's future trajectory.

    Different from RasterEncoder, this class assumes features['raster'] is a
    HorizonRasterV2 object or a dictionary of raster layer tensors.
    """

    def __init__(
            self,
            backbone_name: str = 'resnet18',
            input_raster_names: List[str] = [
                'ego', 'past_current_agents', 'roadmap', 'baseline_paths',
                'route', 'ego_speed'
            ],
            pretrained: bool = True,
            output_stride: int = 32,
            out_indices: List[int] = [4],
            max_batch_size: int = 0,
            lean: bool = False,
    ):
        """
        :param backbone_name: name of the backbone used for encoding
        :param input_raster_names: name of raster layers. It should be one of
            the name defined in `RASTER_NAME_TO_CHANNELS`.
        :param pretrained: whether to use pretrained weights for backbone
        :param output_stride: max output stride
        :param out_indices: list of selected output layer index
        :param lean: whether to lean the model (i.e. gradient checkpointing)
            Note that this is only supported for backbone_name=resnet18.
        :param max_batch_size: maximum batch size. If > 0, the model will be
            split into multiple parts to reduce memory usage. This is only
            supported for lean=True.
        """
        super().__init__()
        self._input_raster_names = input_raster_names
        input_dim = sum(
            RASTER_NAME_TO_CHANNELS[name] for name in input_raster_names)
        self._backbone = timm.create_model(
            backbone_name,
            in_chans=input_dim,
            pretrained=pretrained,
            output_stride=output_stride,
            features_only=True,
            out_indices=out_indices)
        self._max_batch_size = max_batch_size
        if max_batch_size > 0:
            assert lean, "max_batch_size > 0 is only useful for lean=True"
        self._lean = lean
        if lean:
            assert backbone_name == 'resnet18'
            self._backbone = lean_model(self._backbone)

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing rasterized data.
            features['raster'] is a HorizonRasterV2 object or a dictionary of
            raster layer tensors.
        :return: encoder_features: dict of predictions from network
        """
        raster = features["raster"]
        if isinstance(raster, HorizonRasterV2):
            data = raster.data
        else:
            assert isinstance(raster, dict)
            data = raster

        def _atleast_4d(x):
            return x if x.ndim == 4 else x.unsqueeze(1)
        rasters = torch.cat(
            [_atleast_4d(data[name]) for name in self._input_raster_names],
            dim=1)
        if 0 < self._max_batch_size < rasters.shape[0]:
            encoder_features = []
            for i in range(0, rasters.shape[0], self._max_batch_size):
                encoder_features.append(
                    self._backbone.forward(
                        rasters[i:i + self._max_batch_size]))
            encoder_features = [
                torch.cat(features, dim=0)
                for features in zip(*encoder_features)
            ]
        else:
            if rasters.dtype == torch.uint8:
                rasters = rasters.float() / 255.0
            to_device(self._backbone, rasters.device)
            encoder_features = self._backbone.forward(rasters)
        return {"encoder_features": encoder_features}


class RasterEncoderV3(RasterEncoderV2):
    """
    Wrapper around raster-based CNN model that consumes ego, agent and map data in rasterized format
    and regresses ego's future trajectory.

    Different from RasterEncoder, this class assumes features['raster'] is a
    HorizonRasterV2 object or a dictionary of raster layer tensors.
    """

    def __init__(
            self,
            backbone_name: str = 'resnet18',
            input_raster_names: List[str] = [
                'ego', 'past_current_agents', 'roadmap', 'baseline_paths',
                'route', 'ego_speed'
            ],
            pretrained: bool = True,
            output_stride: int = 32,
            out_indices: List[int] = [4],
            max_batch_size: int = 0,
            lean: bool = False,
            out_dim: int = 512,
            n_embed: int = 512, 
    ):
        """
        :param backbone_name: name of the backbone used for encoding
        :param input_raster_names: name of raster layers. It should be one of
            the name defined in `RASTER_NAME_TO_CHANNELS`.
        :param pretrained: whether to use pretrained weights for backbone
        :param output_stride: max output stride
        :param out_indices: list of selected output layer index
        :param lean: whether to lean the model (i.e. gradient checkpointing)
            Note that this is only supported for backbone_name=resnet18.
        :param max_batch_size: maximum batch size. If > 0, the model will be
            split into multiple parts to reduce memory usage. This is only
            supported for lean=True.
        """
        super().__init__(
            backbone_name=backbone_name,
            input_raster_names=input_raster_names,
            pretrained=pretrained,
            output_stride=output_stride,
            out_indices=out_indices,
            max_batch_size=max_batch_size,
            lean=lean,
        )
        self._adaptor = nn.Sequential(
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

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Forward pass for training
        :param features: input features
        :return: encoder_features: dict of predictions from network
        """
        image_features = super().forward(features)["encoder_features"][-1]
        to_device(self._adaptor, image_features.device)
        image_features = self._adaptor(image_features)
        image_features = rearrange(image_features, 'b c h w -> b (h w) c')
        return image_features
