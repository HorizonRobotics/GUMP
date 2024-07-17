from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from torchmetrics import Metric
from nuplan.planning.training.modeling.types import TargetsType
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils
from copy import deepcopy


class SminADEAggregatedMetric(Metric):
    """
    Metric representing the IoU between occupancy prediction adn occupancy target.
    """

    def __init__(self, 
                 name: str = 'SminADE_aggregated_metrics',
                 is_avg_corner_dist: bool = False,
                 is_ego: bool = False) -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        super(SminADEAggregatedMetric, self).__init__()
        self._name = name
        self._is_avg_corner_dist = is_avg_corner_dist
        self._is_ego = is_ego
        if self._is_avg_corner_dist:
            self._name += '_avg_corner_dist'
        if self._is_ego:
            self._name += '_ego'
        
        self.add_state("SminADE", default=torch.tensor(0.0, device=self.device), dist_reduce_fx="sum")
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
        tokenized_arrays = predictions['tokenized_arrays']
        predicted_tokenized_arrays = predictions['predicted_tokenized_arrays']
        smin_ade = gutils.calculate_smin_ade_batch(
            predicted_tokenized_arrays, 
            tokenized_arrays[:, 0], 
            is_avg_corner_dist=self._is_avg_corner_dist,
            ego_only=self._is_ego)
        self.batch += len(tokenized_arrays)
        self.SminADE += smin_ade
        
    def compute(self) -> dict:
        """
        Computes the metric.

        :return: metric scalar tensor
        """
        self.SminADE = self.SminADE.to(self.device)
        self.batch = self.batch.to(self.device)
        return {
            'SminADE': self.SminADE / self.batch
        }
        
    def log(self, logger, data: dict):
        for k, v in data.items():
            logger(f'SminADE/{self._name}', v)
