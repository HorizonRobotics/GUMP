from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from torch.nn import functional as F


class MaskedCrossEntrophyObjective(AbstractObjective):
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
        self._name = f'masked_cross_entrophy_objective'
        self._weight = weight
        self.log_data = {}
        # self.step = 0

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
        output_sequence = predictions['transitional_output']
        pred_agent_state_logits, target_agent_state_tokens = output_sequence.get_agent_state_prediction_target(return_ego_only=False)
        pred_control_logits, target_control_tokens = output_sequence.get_control_prediction_target()
        device = pred_agent_state_logits.device
        dtype = pred_agent_state_logits.dtype
        
        if pred_agent_state_logits.shape[0] == target_agent_state_tokens.reshape(-1).shape[0]:
            loss = []
            agent_state_loss = F.cross_entropy(pred_agent_state_logits,
                                        target_agent_state_tokens.reshape(-1),
                                        ignore_index=-1,
                                        reduction='none') 
            loss.append(agent_state_loss)  

            control_loss = F.cross_entropy(pred_control_logits,
                                        target_control_tokens.reshape(-1),
                                        ignore_index=-1,
                                        reduction='none')  
            loss.append(control_loss)      
        else:
            loss = [torch.tensor([0.0], device=device, dtype=dtype), 
                    torch.tensor([0.0], device=device, dtype=dtype)]
            
        loss = torch.cat(loss, dim=0).mean()
        return loss * self._weight
