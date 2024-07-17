from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from torchmetrics import Metric
from nuplan.planning.training.modeling.types import TargetsType
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils
from copy import deepcopy
from einops import rearrange


class CollidedAggregatedMetric(Metric):
    """
    Metric representing the IoU between occupancy prediction adn occupancy target.
    """

    def __init__(self, 
                 name: str = 'collided_aggregated_metrics',
                 is_ego: bool = False) -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        super(CollidedAggregatedMetric, self).__init__()
        self._name = name
        self._is_ego = is_ego
        if self._is_ego:
            self._name += '_ego'
        
        self.add_state("collision_rate", default=torch.tensor(0.0, device=self.device), dist_reduce_fx="sum")
        self.add_state("batch", default=torch.tensor(0.0, device=self.device), dist_reduce_fx="sum")


    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return [""]

    def update(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        predicted_tokenized_arrays = predictions['predicted_tokenized_arrays']
        for i in range(predicted_tokenized_arrays.shape[0]):
            collision_rate = gutils.calculate_collision_rate_batch(
                predicted_tokenized_arrays[i], 
                ego_only=self._is_ego)
            self.batch += 1
            self.collision_rate += collision_rate

    def compute(self) -> dict:
        """
        Computes the metric.

        :return: metric scalar tensor
        """
        self.collision_rate = self.collision_rate.to(self.device)
        self.batch = self.batch.to(self.device)
        return {
            'collision_rate': self.collision_rate / self.batch
        }
        
    def log(self, logger, data: dict):
        for k, v in data.items():
            logger(f'collision_rate/{self._name}', v)
