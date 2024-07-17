from __future__ import annotations
from dataclasses import dataclass
import torch
import math
import numpy as np
import numpy.typing as npt
from typing import Type
from torch.utils.data.dataloader import default_collate

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from nuplan_extent.planning.scenario_builder.prepared_scenario import PreparedScenario, NpEgoState
from nuplan_extent.planning.training.preprocessing.features.raster_builders import PreparedScenarioFeatureBuilder


@dataclass
class EgoTrajectoryV2(AbstractModelFeature):
    """Raster features

    Different from Trajectory, this class stores the data as a dictionary with
    the raster name as the key and the Trajectory data as the value.
    """
    data: Dict[str, FeatureDataType]

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        return EgoTrajectoryV2(data=data)

    @classmethod
    def collate(cls, batch: List[AbstractModelFeature]) -> AbstractModelFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        serialized = [sample.serialize() for sample in batch]
        return cls.deserialize(serialized)

    def to_feature_tensor(self) -> AbstractModelFeature:
        return self

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        return self

    def unpack(self) -> List[AbstractModelFeature]:
        batch_size = list(self.data.values())[0].shape[0]
        return [
            EgoTrajectoryV2(
                data={name: self.data[name][i]
                      for name in self.data}) for i in range(batch_size)
        ]

class EgoTrajectoryTargetBuilderV2(AbstractTargetBuilder,
                                   PreparedScenarioFeatureBuilder):
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
        return EgoTrajectoryV2  # type: ignore

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        interval = scenario.database_interval

        future_window = int(self._time_horizon / interval)
        assert future_window % self._num_future_poses == 0, (
            f"future window {future_window} must be divisible by "
            f"num_future_poses {self._num_future_poses}")
        step_interval = future_window // self._num_future_poses

        # 1. get all the future iteration and current iteration for iterations
        all_iterations = set()
        for iteration in iterations:
            all_iterations.update(
                range(iteration + step_interval, iteration + future_window + 1,
                      step_interval))
        all_iterations = sorted(all_iterations)

        last_iteration = scenario.get_number_of_iterations() - 1

        if all_iterations[-1] > last_iteration:
            num_samples = all_iterations[-1] - last_iteration
            if num_samples == 1 and isinstance(scenario, NuPlanScenario):
                # Due to an artifact of scenario_utils.sample_indices_with_time_horizon
                # that when num_samples=1, it actually gets a sample at one
                # iteration before the given iteration.
                # See nuplan issue https://github.com/motional/nuplan-devkit/issues/348
                # TODO: remove this hack when the issue is fixed.
                time_horizon = 0.5 * interval
            else:
                time_horizon = num_samples * interval

            future_ego_states = list(
                scenario.get_ego_future_trajectory(
                    iteration=last_iteration,
                    time_horizon=time_horizon,
                    num_samples=num_samples))

        # 2. prepare ego states for all iterations
        for iteration in all_iterations:
            if prepared_scenario.feature_exists_at_iteration(
                    "ego_state", iteration):
                continue
            if iteration > last_iteration:
                ego_state = future_ego_states[iteration - last_iteration - 1]
            else:
                ego_state = scenario.get_ego_state_at_iteration(iteration)
            prepared_scenario.add_ego_state_at_iteration(ego_state, iteration)

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        step_interval = int(self._time_horizon / scenario.database_interval /
                            self._num_future_poses)
        ego_states = [
            scenario.get_ego_state_at_iteration(iter) for iter in range(
                iteration + step_interval, iteration +
                step_interval * self._num_future_poses + 1, step_interval)
        ]
        positions = np.array([[ego.x, ego.y] for ego in ego_states])
        current_position = np.array([ego_state.x, ego_state.y])
        current_heading = ego_state.heading
        c = math.cos(current_heading)
        s = math.sin(current_heading)
        global_to_local_rotation_mat = np.array([[c, -s], [s, c]])
        relative_positions = np.matmul(positions - current_position,
                                       global_to_local_rotation_mat)

        headings = np.array([ego.heading for ego in ego_states])
        relative_headings = headings - current_heading
        relative_headings[relative_headings > math.pi] -= 2 * math.pi
        relative_headings[relative_headings < -math.pi] += 2 * math.pi
        relative_poses = np.hstack((relative_positions,
                                    relative_headings.reshape(-1, 1)))
        return Trajectory(data=relative_poses.astype(np.float32))

    def get_targets(self, scenario: AbstractScenario) -> Trajectory:
        """Inherited, see superclass."""
        iterations = range(0, 1)
        prepared_scenario = PreparedScenario()
        prepared_scenario.prepare_scenario(scenario, iterations)
        ego_state = prepared_scenario.get_ego_state_at_iteration(0)
        return self.get_features_from_prepared_scenario(
            prepared_scenario, 0, ego_state)
