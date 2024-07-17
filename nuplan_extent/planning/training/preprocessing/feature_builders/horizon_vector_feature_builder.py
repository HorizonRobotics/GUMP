from typing import Dict, List, Tuple, Type, cast

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_generic_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
    AgentInternalIndex,
)
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import (
    pad_agent_with_max,
    pack_agents_tensor_withindex,
    build_generic_ego_features_from_tensor,
)


class GenericAgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self,
                agent_features: List[str],
                past_trajectory_sampling: TrajectorySampling,
                future_trajectory_sampling: TrajectorySampling,
                num_max_agents: int = 256,
                agent_tokenized_params: Dict = {}) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self._agent_features = agent_features
        self._num_past_poses = past_trajectory_sampling.num_poses
        self._past_time_horizon = past_trajectory_sampling.time_horizon
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._future_time_horizon = future_trajectory_sampling.time_horizon
        self._num_max_agents = num_max_agents

        self._agents_states_dim = GenericAgents.agents_states_dim()
        # Sanitize feature building parameters
        if 'EGO' in self._agent_features:
            raise AssertionError("EGO not valid agents feature type!")
        for feature_name in self._agent_features:
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(
                    f"Object representation for layer: {feature_name} is unavailable!")

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "generic_agents"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return GenericAgents  # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(
            self, scenario: AbstractScenario) -> GenericAgents:
        """Inherited, see superclass."""
        # Retrieve present/past ego states and agent boxes
        with torch.no_grad():
            anchor_ego_state = scenario.initial_ego_state

            past_ego_states = scenario.get_ego_past_trajectory(
                iteration=0, num_samples=self._num_past_poses, time_horizon=self._past_time_horizon
            )
            future_ego_states = scenario.get_ego_future_trajectory(
                iteration=0, num_samples=self._num_future_poses, time_horizon=self._future_time_horizon
            )
            past_ego_states = list(past_ego_states)
            sampled_all_ego_states = past_ego_states + \
                [anchor_ego_state] + list(future_ego_states)
            ego_state_index = len(past_ego_states)

            time_stamps = list(
                scenario.get_past_timestamps(
                    iteration=0, num_samples=self._num_past_poses, time_horizon=self._past_time_horizon
                )
            ) + [scenario.start_time] + list(
                scenario.get_future_timestamps(
                    iteration=0, num_samples=self._num_future_poses, time_horizon=self._future_time_horizon
                )
            )
            # Retrieve present/future agent boxes
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self._past_time_horizon, num_samples=self._num_past_poses
                )
            ]

            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self._future_time_horizon, num_samples=self._num_future_poses
                )
            ]

            # Extract and pad features
            sampled_all_observations = past_tracked_objects + \
                [present_tracked_objects] + future_tracked_objects

            assert len(sampled_all_ego_states) == len(sampled_all_observations), (
                "Expected the trajectory length of ego and agent to be equal. "
                f"Got ego: {len(sampled_all_ego_states)} and agent: {len(sampled_all_observations)}"
            )

            assert len(sampled_all_observations) > 2, (
                "Trajectory of length of " f"{len(sampled_all_observations)} needs to be at least 3"
            )

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                sampled_all_ego_states, time_stamps, sampled_all_observations
            )

            tensors, list_tensors, list_list_tensors = self.scriptable_forward(
                tensors, list_tensors, list_list_tensors, ego_state_index)

            output: GenericAgents = self._unpack_feature_from_tensor_dict(
                tensors, list_tensors, list_list_tensors)
            return output

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            history = current_input.history
            assert isinstance(
                history.observations[0], DetectionsTracks
            ), f"Expected observation of type DetectionTracks, got {type(history.observations[0])}"
            present_ego_state, present_observation = history.current_state

            past_observations = history.observations[:-1]
            past_ego_states = history.ego_states[:-1]

            assert history.sample_interval, "SimulationHistoryBuffer sample interval is None"

            indices = sample_indices_with_time_horizon(
                self._num_past_poses, self._past_time_horizon, history.sample_interval
            )

            try:
                sampled_past_observations = [
                    cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)
                ]
                sampled_past_ego_states = [
                    past_ego_states[-idx] for idx in reversed(indices)]
            except IndexError:
                raise RuntimeError(
                    f"SimulationHistoryBuffer duration: {history.duration} is "
                    f"too short for requested past_time_horizon: {self._past_time_horizon}. "
                    f"Please increase the simulation_buffer_duration in default_simulation.yaml"
                )

            sampled_past_observations = sampled_past_observations + [
                cast(DetectionsTracks, present_observation).tracked_objects
            ]
            ego_state_index = len(sampled_past_ego_states)
            sampled_past_ego_states = sampled_past_ego_states + \
                [present_ego_state]
            time_stamps = [
                state.time_point for state in sampled_past_ego_states]

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                sampled_past_ego_states, time_stamps, sampled_past_observations
            )
            tensors, list_tensors, list_list_tensors = self.scriptable_forward(
                tensors, list_tensors, list_list_tensors, ego_state_index)

            output: GenericAgents = self._unpack_feature_from_tensor_dict(
                tensors, list_tensors, list_list_tensors)

            return output

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        all_ego_states: List[EgoState],
        all_time_stamps: List[TimePoint],
        all_tracked_objects: List[TrackedObjects],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param all_ego_states: The all states of the ego vehicle.
        :param all_time_stamps: The all time stamps of the input data.
        :param all_tracked_objects: The all tracked objects.
        :return: The packed tensors.
        """
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        all_ego_states_tensor = sampled_past_ego_states_to_tensor(
            all_ego_states)
        all_time_stamps_tensor = sampled_past_timestamps_to_tensor(
            all_time_stamps)

        # for feature_name in self._agent_features:
        #     all_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(
        #         all_tracked_objects, TrackedObjectType[feature_name]
        #     )
        #     list_tensor_data[f"all_tracked_objects.{feature_name}"] = all_tracked_objects_tensor_list
        for feature_name in self._agent_features:
            all_tracked_objects_tensor_list, track_token_to_ids = sampled_tracked_objects_to_tensor_list(
                all_tracked_objects, TrackedObjectType[feature_name]
            )
            list_tensor_data[f"all_tracked_objects.{feature_name}"] = all_tracked_objects_tensor_list
            list_tensor_data[f"all_tracked_objects.{feature_name}.token_mapping"] = track_token_to_ids
            # print(f"feature_name: {feature_name}, track_token_to_ids: {track_token_to_ids}")

        return (
            {
                "all_ego_states": all_ego_states_tensor,
                "all_time_stamps": all_time_stamps_tensor,
            },
            list_tensor_data,
            {},
        )

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> GenericAgents:
        """
        Unpacks the data returned from the scriptable core into an GenericAgents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed GenericAgents object.
        """
        ego_features = [
            list_tensor_data["generic_agents.ego"][0].detach().numpy()]
        agent_features = {}
        for key in list_tensor_data:
            if key.startswith("generic_agents.agents."):
                feature_name = key[len("generic_agents.agents."):]
                if isinstance(list_tensor_data[key][0], torch.Tensor):
                    agent_features[feature_name] = [
                        list_tensor_data[key][0].detach().numpy()]
                else:
                    agent_features[feature_name] = [
                        list_tensor_data[key][0]]

        return GenericAgents(ego=ego_features, agents=agent_features)

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
        ego_state_index: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[torch.Tensor]] = {}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}

        ego_history: torch.Tensor = tensor_data["all_ego_states"]
        time_stamps: torch.Tensor = tensor_data["all_time_stamps"]
        anchor_ego_state = ego_history[ego_state_index, :].squeeze()

        # ego features
        ego_tensor = build_generic_ego_features_from_tensor(
            ego_history, ego_state_index)
        output_list_dict["generic_agents.ego"] = [ego_tensor]

        # agent features
        for feature_name in self._agent_features:

            if f"all_tracked_objects.{feature_name}" in list_tensor_data:
                agents: List[torch.Tensor] = list_tensor_data[
                    f"all_tracked_objects.{feature_name}"]
                agent_history = agents
                # agent_history = filter_agents_tensor(agents, reverse=True)
                track_token_to_ids = list_tensor_data[f"all_tracked_objects.{feature_name}.token_mapping"]

                if agent_history[-1].shape[0] == 0:
                    # Return zero array when there are no agents in the scene
                    agents_tensor: torch.Tensor = torch.zeros(
                        (len(agent_history), 0, self._agents_states_dim)).float()
                else:
                    agent_history = pad_agent_with_max(
                        agent_history, self._num_max_agents)

                    local_coords_agent_states = convert_absolute_quantities_to_relative(
                        agent_history, anchor_ego_state, position_only=True
                    )

                    # Calculate yaw rate
                    agents_tensor = pack_agents_tensor_withindex(
                        local_coords_agent_states)

                output_list_dict[f"generic_agents.agents.{feature_name}"] = [
                    agents_tensor]
                output_list_dict[f"generic_agents.agents.{feature_name}.token_mapping"] = [
                    track_token_to_ids]

        return output_dict, output_list_dict, output_list_list_dict

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {
            "all_ego_states": {
                "iteration": "0",
                "num_samples": str(self._num_all_poses),
                "time_horizon": str(self._all_time_horizon),
            },
            "all_time_stamps": {
                "iteration": "0",
                "num_samples": str(self._num_all_poses),
                "time_horizon": str(self._all_time_horizon),
            },
            "all_tracked_objects": {
                "iteration": "0",
                "time_horizon": str(self._all_time_horizon),
                "num_samples": str(self._num_all_poses),
                "agent_features": ",".join(self._agent_features),
            },
        }
