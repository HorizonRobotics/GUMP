from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from torch.nn import functional as F
from third_party.functions.print_grad import PrintGrad


class KineticsCrossEntropyObjective(AbstractObjective):
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
        valid_mask = torch.tensor(predictions['valid_mask'], dtype=torch.bool).to(predictions['pred_agent_logits'].device)
        pred_agent_state_logits = predictions['pred_agent_logits'][valid_mask]
        target_agent_state_tokens = predictions['target_tokenized_state'][valid_mask].reshape(-1)
        
        loss_agent = F.cross_entropy(pred_agent_state_logits,
                                    target_agent_state_tokens,
                                    ignore_index=-1,
                                    reduction='none') 
        loss = loss_agent.mean()
        
        return loss * self._weight
