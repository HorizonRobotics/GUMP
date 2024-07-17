from __future__ import annotations

from dataclasses import dataclass
from nuplan.planning.training.preprocessing.features.abstract_model_feature import \
    FeatureDataType
from nuplan.planning.training.preprocessing.features.raster import Raster
import torch
from torch.utils.data.dataloader import default_collate


@dataclass
class HorizonRaster(Raster):

    @property
    def ego_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the ego layer
        located at channel 0.
        """
        return self._get_data_channel(range(0, 1))

    @property
    def agents_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the agents layer
        located at channel 1.
        """
        return self._get_data_channel(range(1, 2))

    @property
    def roadmap_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the map layer
        located at channel 2.
        """
        return self._get_data_channel(2)

    @property
    def baseline_paths_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the baseline paths layer
        located at channel 3.
        """
        return self._get_data_channel(3)

    @property
    def navigation_block_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the navigation block layer
        located at channel 4.
        """
        return self._get_data_channel(4)

    @property
    def speed_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the speed layer
        located at channel 5.
        """
        return self._get_data_channel(5)

    @property
    def drivable_area_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the drivable area layer
        located at channel 6.
        """
        return self._get_data_channel(6)

    @property
    def speed_limit_layer(self) -> FeatureDataType:
        """
        Get the 2D grid representing the speed limit layer
        located at channel 7.
        """
        return self._get_data_channel(7)
