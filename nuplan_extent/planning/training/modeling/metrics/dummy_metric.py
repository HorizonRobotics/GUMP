from typing import List

import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import (
    AbstractTrainingMetric,
)
from nuplan.planning.training.modeling.types import TargetsType
class DummyMetric(AbstractTrainingMetric):
    def __init__(self) -> None:
        pass

    def name(self) -> str:
        return "Dummy"

    def get_list_of_required_target_types(self) -> List[str]:
        return []

    def compute(self, predictions, targets) -> torch.Tensor:
        return torch.tensor(0.0)


class DummyDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'test') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        return torch.tensor(0.0)
        # predicted_trajectory: Trajectory = predictions["multimode_trajectory"]
        # targets_trajectory: MultiModeTrajectory = targets["trajectory"]
        # predicted_trajectory = Trajectory(predicted_trajectory.data[:, 0, :, :].squeeze(1))

        # return torch.norm(predicted_trajectory.xy - targets_trajectory.xy, dim=-1).mean()
