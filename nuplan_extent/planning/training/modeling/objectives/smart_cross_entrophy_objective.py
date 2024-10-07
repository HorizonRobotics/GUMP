from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from torch.nn import functional as F
import torch.nn as nn
from third_party.functions.print_grad import PrintGrad


class SMARTCrossEntrophyObjective(AbstractObjective):
    """
    A class that represents the speed heatmap objective for trajectory prediction models in autonomous driving.
    Enforces the predicted heatmap to be close to the optimal speed heatmap.
    Can improve speed limit compliance, and ego progress along expert routes.
    """

    def __init__(self,
                 scenario_type_loss_weighting: Dict[str, float],
                 weight: float = 1.0):
        """
        """
        self._name = f'cross_entrophy_objective'
        self._weight = weight
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.map_cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

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
        
        next_token_prob = predictions['next_token_prob']
        next_token_idx_gt = predictions['next_token_idx_gt']
        next_token_eval_mask = predictions['next_token_eval_mask']
        cls_loss = self.cls_loss(next_token_prob[next_token_eval_mask], next_token_idx_gt[next_token_eval_mask])
        loss = cls_loss
        return loss * self._weight
