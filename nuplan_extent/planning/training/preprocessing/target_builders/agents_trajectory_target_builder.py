from __future__ import annotations

from typing import Type

import torch
from nuplan.common.actor_state.tracked_objects import (AGENT_TYPES,
                                                       STATIC_OBJECT_TYPES)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import \
    TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import \
    AbstractModelFeature
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.agents_trajectories import \
    AgentsTrajectories
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import \
    AbstractTargetBuilder
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative, filter_agents_tensor,
    pack_agents_tensor, pad_agent_states, sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor)
from nuplan_extent.planning.training.preprocessing.utils.agents_preprocessing import \
    sampled_tracked_objects_based_on_object_type


class AgentTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon

        self._agents_states_dim = Agents.agents_states_dim()

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "agents_trajectory_target"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return AgentsTrajectories  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> AgentsTrajectories:
        """Inherited, see superclass."""
        ego_state = scenario.initial_ego_state
        ego_states_tensor = sampled_past_ego_states_to_tensor([ego_state])
        ego_states_tensor = ego_states_tensor[-1, :].squeeze()

        future_time_stamps = list(
            scenario.get_future_timestamps(
                iteration=0,
                num_samples=self._num_future_poses,
                time_horizon=self._time_horizon))
        future_time_stamps_tensor = sampled_past_timestamps_to_tensor(
            future_time_stamps)

        # Retrieve present/future agent boxes
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        future_tracked_objects_lst = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=self._time_horizon,
                num_samples=self._num_future_poses)
        ]
        present_future_observations = [present_tracked_objects
                                       ] + future_tracked_objects_lst

        # Retrieve future tracked objects based on object type
        object_types = list(AGENT_TYPES) + list(STATIC_OBJECT_TYPES)
        present_future_tracked_objects_tensor_list = sampled_tracked_objects_based_on_object_type(
            present_future_observations, object_types)

        # keep tracked objects that only appears in the current frame
        present_future_agents = filter_agents_tensor(
            present_future_tracked_objects_tensor_list, reverse=False)

        if len(present_future_agents
               ) == 0 or present_future_agents[-1].shape[0] == 0:
            # Return zero array when there are no agents in the scene
            future_agents_tensor: torch.Tensor = torch.zeros(
                (len(present_future_agents) - 1, 0,
                 self._agents_states_dim)).float()
        else:
            padded_present_future_agent_states = pad_agent_states(
                present_future_agents, reverse=False)
            padded_future_agent_states = padded_present_future_agent_states[1:]

            local_coords_future_agent_states = convert_absolute_quantities_to_relative(
                padded_future_agent_states, ego_states_tensor)

            # Calculate yaw rate
            future_yaw_rate_horizon = compute_yaw_rate_from_state_tensors(
                padded_future_agent_states, future_time_stamps_tensor)

            future_agents_tensor = pack_agents_tensor(
                local_coords_future_agent_states, future_yaw_rate_horizon)

        agent_features = future_agents_tensor.detach().numpy()

        return AgentsTrajectories(data=[agent_features])
