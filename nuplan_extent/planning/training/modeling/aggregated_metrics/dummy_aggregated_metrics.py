from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from torchmetrics import Metric
from nuplan.planning.training.modeling.types import TargetsType

from nuplan_extent.planning.training.callbacks.utils.visualization_utils import draw_bev_bboxes
from nuplan_extent.planning.training.modeling.models.utils import shift_down
from copy import deepcopy

SPEED_NORMALIZATION_FACTOR = 16

class DummyAggregatedMetric(Metric):
    """
    Metric representing the IoU between occupancy prediction adn occupancy target.
    """

    def __init__(self, 
                 name: str = 'dummy_aggregated_metrics') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        super(DummyAggregatedMetric, self).__init__()
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def update(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        pass
        
    def compute(self) -> dict:
        """
        Computes the metric.

        :return: metric scalar tensor
        """
        return {}
        
    def log(self, logger, data: dict):
        pass
