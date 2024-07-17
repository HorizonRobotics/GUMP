from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories
from nuplan_extent.planning.training.modeling.models.utils import angle_to_sin_cos


class AgentsMTPObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(
        self,
        scenario_type_loss_weighting: Dict[str, float],
        name: str = 'agent_mtp_objective',
        weight: float = 1.0,
        level: int = 0,
        past_pred_step: int = 4,
        future_pred_step: int = 17,
        alpha=1.0,
        beta=1.0,
        heading_weight=2.0,
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._level = level
        self.alpha = alpha
        self.beta = beta
        self._name = name + '_' + str(level)
        self._weight = weight
        self._past_pred_step = past_pred_step
        self._future_pred_step = future_pred_step
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting
        self._heading_weight = heading_weight


    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        output_sequence = predictions['transitional_output']
        lvl_pred_log_prob, lvl_pred_future_trajectory, target_future_trajectory = output_sequence.get_future_trajectory_prediction_target()
        pred_log_prob, pred_future_trajectory = lvl_pred_log_prob[self._level], lvl_pred_future_trajectory[self._level]
        target_future_trajectory_sin = torch.sin(target_future_trajectory[..., -1:])
        target_future_trajectory_cos = torch.cos(target_future_trajectory[..., -1:])
        targets_trajectory = torch.cat([target_future_trajectory[...,:2], target_future_trajectory_sin, target_future_trajectory_cos], dim=-1)
        mask = ~torch.isnan(targets_trajectory)
        # Obtain mode with minimum ADE with respect to ground truth:
        # Nb, num_mode, seq_len, 2
        batch_size, num_modes, sequence_length, _ = pred_future_trajectory.shape
        targets_trajectory = torch.nan_to_num(targets_trajectory, nan=0.0)
        # from third_party.functions.print_grad import PrintGrad
        # pred_future_trajectory = PrintGrad.apply(pred_future_trajectory)
        ade_errs = self._fn_xy(pred_future_trajectory[..., :2], targets_trajectory[:, None, :, :2]) * mask[:, None, :, :2]
        ade_errs = torch.pow(ade_errs.sum(dim=3) + 1e-5, 0.5).sum(dim=2) / sequence_length  # Nb, num_mode
        
        ahe_errs = self._fn_heading(pred_future_trajectory[..., 2:], targets_trajectory[:, None, :, 2:]) * mask[:, None, :, 2:]
        ahe_errs = ahe_errs.sum(dim=3).sum(dim=2) / sequence_length  # N, num_mode

        # Compute the regularized loss
        l_reg = ade_errs + ahe_errs * self._heading_weight
        inds = l_reg.argmin(dim=1)  # N
        l_reg = l_reg.gather(1, inds[:, None]).mean()
        # Compute classification loss
        l_class = - torch.mean(torch.squeeze(pred_log_prob.gather(1, inds.unsqueeze(1))))

        loss = self.beta * l_reg + self.alpha * l_class

        return self._weight * loss 
