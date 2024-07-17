from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


class AgentsADEObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(
        self,
        scenario_type_loss_weighting: Dict[str, float],
        name: str = 'agent_ade_objective',
        weight: float = 1.0,
        level: int = 0,
        heading_weight=2.0,
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._level = level
        self._name = name + '_' + str(level)
        self._weight = weight
        self._heading_weight = heading_weight
        self._fn_xy = torch.nn.modules.loss.L1Loss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

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
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        output_sequence = predictions['transitional_output']
        lvl_pred_past_trajectory, target_past_trajectory = output_sequence.get_past_trajectory_prediction_target()
        pred_past_trajectory = lvl_pred_past_trajectory[self._level]
        target_past_trajectory_sin = torch.sin(target_past_trajectory[..., -1:])
        target_past_trajectory_cos = torch.cos(target_past_trajectory[..., -1:])
        target_past_trajectory = torch.cat([target_past_trajectory[...,:2], target_past_trajectory_sin, target_past_trajectory_cos], dim=-1)
        mask = ~torch.isnan(target_past_trajectory)
        
        batch_size, sequence_length, _ = pred_past_trajectory.shape
        
        # from third_party.functions.print_grad import PrintGrad
        # pred_past_trajectory = PrintGrad.apply(pred_past_trajectory)
        # import pdb; pdb.set_trace()
        # loss_xy = self._fn_xy(pred_past_trajectory[..., :2], target_past_trajectory[..., :2]) * mask[:, :, :2]
        # loss_xy = torch.pow(loss_xy.sum(dim=-1) + 1e-5, 0.5).sum(dim=1) / sequence_length  # Nb, num_mode
        # loss_xy = loss_xy.mean()
        
        loss_xy = self._fn_xy(pred_past_trajectory[...,:2], target_past_trajectory[..., :2].detach()).sum(dim=-1)[mask[..., :2].any(dim=-1)]
        loss_xy = loss_xy.mean()
        # loss_xy = torch.pow(loss_xy + 1e-5, 0.5).mean()
        
        loss_heading = self._fn_heading(pred_past_trajectory[...,2:], target_past_trajectory[..., 2:].detach())[mask[...,2:]].mean()
        loss = loss_xy + loss_heading * self._heading_weight
        return self._weight * loss 
