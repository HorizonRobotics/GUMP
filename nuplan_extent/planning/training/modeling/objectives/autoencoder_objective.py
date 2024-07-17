from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from torch.nn import functional as F

from einops import rearrange


class AutoEncoderObjective(AbstractObjective):
    """
    A class that represents the speed heatmap objective for trajectory prediction models in autonomous driving.
    Enforces the predicted heatmap to be close to the optimal speed heatmap.
    Can improve speed limit compliance, and ego progress along expert routes.
    """

    def __init__(self,
                 scenario_type_loss_weighting: Dict[str, float],
                 weight: float = 1.0,
                 input_channel_indexes: List[int] = [2, 3, 4, 6, 7]):
        """
        """
        self._name = 'autoencoder_objective'
        self._weight = weight
        self._input_channel_indexes = input_channel_indexes
        self._criterion = torch.nn.L1Loss()

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: FeaturesType, targets: TargetsType,
                scenarios: ScenarioListType) -> torch.Tensor:
        # ego_raster,
        # agents_raster 1,
        # roadmap_raster 2 ,
        # baseline_paths_raster3 ,
        # route_raster4 ,
        # ego_speed_raster,
        # static_agents_raster 6,
        target = predictions["raster"].data[:, self._input_channel_indexes]
        reconstructed_image = predictions['reconstruct_image'].data
        loss = self._criterion(reconstructed_image, target)
        return loss * self._weight
