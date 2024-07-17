from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from torch.nn import functional as F


class MaskedCrossEntrophyObjectiveV1_1(AbstractObjective):
    """
    A class that represents the speed heatmap objective for trajectory prediction models in autonomous driving.
    Enforces the predicted heatmap to be close to the optimal speed heatmap.
    Can improve speed limit compliance, and ego progress along expert routes.
    """

    def __init__(self,
                 scenario_type_loss_weighting: Dict[str, float],
                 weight_agent: float = 1.0,
                 weight_ctrl: float = 0.1,
                 weight: float = 1.0):
        """
        """
        self._name = f'cross_entrophy_objective'
        self._weight = weight
        self._weight_agent = weight_agent
        self._weight_ctrl = weight_ctrl

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
        if 'pred_agent_logits' not in predictions:
            return torch.tensor(0.0)
        pred_agent_state_logits = predictions['pred_agent_logits'].reshape(-1, predictions['pred_agent_logits'].shape[-1])
        target_agent_state_tokens = predictions['target_tokenized_state']
        loss_agent = F.cross_entropy(pred_agent_state_logits,
                                    target_agent_state_tokens.reshape(-1),
                                    ignore_index=-1,
                                    reduction='none') 
        pred_control_logits = predictions['pred_control_logits']
        target_ctrl_tokens = predictions['target_ctrl_tokens']
        loss_ctrl = F.cross_entropy(pred_control_logits,
                                    target_ctrl_tokens.reshape(-1),
                                    ignore_index=-1,
                                    reduction='none')
        loss = torch.cat([self._weight_agent * loss_agent, self._weight_ctrl * loss_ctrl]).mean()
        return loss * self._weight
