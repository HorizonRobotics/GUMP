from __future__ import annotations
import numpy as np
from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan_extent.planning.training.preprocessing.features.tensor import Tensor
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan_extent.planning.training.preprocessing.features.dict_tensor_feature import DictTensorFeature
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan_extent.planning.training.preprocessing.target_builders.wod_ego_trajectory_target_builder import id_to_int_array


class WODInteractiveTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "trajectory"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return DictTensorFeature  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> DictTensorFeature:
        """Inherited, see superclass."""
        current_absolute_state = scenario.initial_ego_state

        scenario_id = id_to_int_array(scenario.scenario_name)

        ret = {}
        ret["scenario_id"] = scenario_id
        ret["ego_id"] = np.array([int(scenario.agent_id)])
        ret["local_to_global_transform"] = current_absolute_state.rear_axle.as_matrix()

        return DictTensorFeature(ret)
