from __future__ import annotations

from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan_extent.planning.training.preprocessing.features.tensor import Tensor
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan_extent.planning.training.preprocessing.features.waymo_tensor import WaymoTensor
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

import numpy as np


def id_to_int_array(hex_id):
    # Validate the input to ensure it's a 16-character hexadecimal string
    if len(hex_id) < 16:
        hex_id = hex_id.zfill(16)
    elif len(hex_id) > 16:
        print("Invalid ID length: ", hex_id)
        exit(-1)

    # Break the ID into two halves
    first_half = hex_id[:8]
    second_half = hex_id[8:]

    # Convert each half to an integer
    first_int = int(first_half, 16)
    second_int = int(second_half, 16)

    # Store the two integers in an array
    int_array = [first_int, second_int]

    return np.array(int_array, dtype=np.int64)


def int_array_to_id(int_array):
    # Validate the input to ensure it's an array of two integers
    if len(int_array) != 2:
        return "Invalid array length"

    # Convert each integer to a hexadecimal string
    first_hex = format(int_array[0], '08x')
    second_hex = format(int_array[1], '08x')

    # Concatenate the two hexadecimal strings
    hex_id = first_hex + second_hex
    while hex_id[0] == '0':
        hex_id = hex_id[1:]

    return hex_id


class WODEgoTrajectoryTargetBuilder(AbstractTargetBuilder):
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
        return WaymoTensor  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> WaymoTensor:
        """Inherited, see superclass."""
        current_absolute_state = scenario.initial_ego_state
        trajectory_absolute_states = scenario.get_ego_all_trajectory()

        # Get all future poses relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        states = np.zeros((len(trajectory_relative_poses), 8), dtype=np.float32)
        for i in range(len(trajectory_relative_poses)):
            states[i][0] = trajectory_relative_poses[i][0]
            states[i][1] = trajectory_relative_poses[i][1]
            states[i][2] = trajectory_absolute_states[i].car_footprint.oriented_box.length
            states[i][3] = trajectory_absolute_states[i].car_footprint.oriented_box.width
            states[i][4] = trajectory_relative_poses[i][2]
            states[i][5] = trajectory_absolute_states[i].dynamic_car_state.rear_axle_velocity_2d.x
            states[i][6] = trajectory_absolute_states[i].dynamic_car_state.rear_axle_velocity_2d.y
            states[i][7] = scenario.ego_type

        scenario_id = id_to_int_array(scenario.scenario_name)

        return WaymoTensor(data=states, scenario_id=scenario_id, agent_id=int(scenario.agent_id), local_to_global_transform=current_absolute_state.rear_axle.as_matrix())
