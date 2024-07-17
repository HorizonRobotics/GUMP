from typing import Dict, List, Tuple, Type, cast

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import GenericAgentsFeatureBuilder
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    convert_absolute_quantities_to_relative, filter_agents_tensor)
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import (
    pad_agent_with_max,
    pack_agents_tensor_withindex,
    build_generic_ego_features_from_tensor,
)
from nuplan_extent.planning.training.preprocessing.features.waymo_generic_agents import (
    WaymoEgoInternalIndex, WaymoAgentInternalIndex, WaymoGenericAgents)


class WaymoGenericAgentsFeatureBuilder(GenericAgentsFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self,
                 agent_features: List[str],
                 past_trajectory_sampling: TrajectorySampling,
                 future_trajectory_sampling: TrajectorySampling,
                 num_max_agents: int = 320,
                 agent_tokenized_params: Dict = {}) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__(
            agent_features,
            past_trajectory_sampling,
        )
        self._agent_features = agent_features
        self._num_past_poses = past_trajectory_sampling.num_poses
        self._past_time_horizon = past_trajectory_sampling.time_horizon
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._future_time_horizon = future_trajectory_sampling.time_horizon
        self._num_max_agents = num_max_agents

        self._agents_states_dim = WaymoGenericAgents.agents_states_dim()
        # Sanitize feature building parameters
        if 'EGO' in self._agent_features:
            raise AssertionError("EGO not valid agents feature type!")
        for feature_name in self._agent_features:
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(
                    f"Object representation for layer: {feature_name} is unavailable!"
                )

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "generic_agents"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return WaymoGenericAgents  # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(self,
                                   scenario: AbstractScenario,
                                   iteration: int = 10) -> WaymoGenericAgents:
        """Inherited, see superclass."""
        self.ego_type = scenario.ego_type
        with_future = True
        # Retrieve present/past ego states and agent boxes
        with torch.no_grad():
            anchor_ego_state = scenario.initial_ego_state

            past_ego_states = scenario.get_ego_past_trajectory(
                iteration=iteration,
                num_samples=self._num_past_poses,
                time_horizon=self._past_time_horizon)
            if with_future:
                future_ego_states = scenario.get_ego_future_trajectory(
                    iteration=iteration,
                    num_samples=self._num_future_poses,
                    time_horizon=self._future_time_horizon)
            past_ego_states = list(past_ego_states)

            if with_future:
                sampled_all_ego_states = past_ego_states + [
                    anchor_ego_state
                ] + list(future_ego_states)
            else:
                sampled_all_ego_states = past_ego_states + [
                    anchor_ego_state
                ]
            ego_state_index = len(past_ego_states)

            # Retrieve present/future agent boxes
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self._past_time_horizon,
                    num_samples=self._num_past_poses)
            ]
            if with_future:
                future_tracked_objects = [
                    tracked_objects.tracked_objects
                    for tracked_objects in scenario.get_future_tracked_objects(
                        iteration=0,
                        time_horizon=self._future_time_horizon,
                        num_samples=self._num_future_poses)
                ]

            # Extract and pad features
            if with_future:
                sampled_all_observations = past_tracked_objects + [
                    present_tracked_objects
                ] + future_tracked_objects
            else:
                sampled_all_observations = past_tracked_objects + [
                    present_tracked_objects
                ]

            assert len(sampled_all_ego_states) == len(
                sampled_all_observations
            ), ("Expected the trajectory length of ego and agent to be equal. "
                f"Got ego: {len(sampled_all_ego_states)} and agent: {len(sampled_all_observations)}"
                )

            assert len(sampled_all_observations) > 2, (
                "Trajectory of length of "
                f"{len(sampled_all_observations)} needs to be at least 3")

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                sampled_all_ego_states, sampled_all_observations)

            tensors, list_tensors, list_list_tensors = self.scriptable_forward(
                tensors, list_tensors, list_list_tensors, ego_state_index)

            output: WaymoGenericAgents = self._unpack_feature_from_tensor_dict(
                tensors, list_tensors, list_list_tensors)

            return output

    def extract_ego_agent_tensor(
            self, past_ego_states: List[EgoState]) -> torch.Tensor:
        """
        Extracts the relevant data from the agents present in a past detection into a tensor.
        Only objects of specified type will be transformed. Others will be ignored.
        The output is a tensor as described in AgentInternalIndex
        :param tracked_objects: The tracked objects to turn into a tensor.
        :track_token_ids: A dictionary used to assign track tokens to integer IDs.
        :object_type: TrackedObjectType to filter agents by.
        :return: The generated tensor and the updated track_token_ids dict.
        """

        output = torch.zeros(
            (len(past_ego_states), WaymoEgoInternalIndex.dim()),
            dtype=torch.float32)
        for i in range(0, len(past_ego_states), 1):
            output[i, WaymoEgoInternalIndex.x()] = past_ego_states[i].rear_axle.x
            output[i, WaymoEgoInternalIndex.y()] = past_ego_states[i].rear_axle.y
            output[i, WaymoEgoInternalIndex.heading()] = past_ego_states[i].rear_axle.heading
            output[i, WaymoEgoInternalIndex.vx()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
            output[i, WaymoEgoInternalIndex.vy()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
            output[i, WaymoEgoInternalIndex.ax()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
            output[i, WaymoEgoInternalIndex.ay()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y
            output[i, WaymoEgoInternalIndex.width()] = past_ego_states[i].car_footprint.oriented_box.width
            output[i, WaymoEgoInternalIndex.length()] = past_ego_states[i].car_footprint.oriented_box.length
            output[i, WaymoEgoInternalIndex.z()] = 0.0
            output[i, WaymoEgoInternalIndex.type()] = self.ego_type        

        return output

    def extract_agent_tensor(
        self,
        past_tracked_objects: List[TrackedObjects],
        object_type: TrackedObjectType = TrackedObjectType.VEHICLE
    ) -> List[torch.Tensor]:
        """
        Tensorizes the agents features from the provided past detections.
        For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
        :param past_tracked_objects: The tracked objects to tensorize.
        :param object_type: TrackedObjectType to filter agents by.
        :return: The tensorized objects.
        """
        output: List[torch.Tensor] = []
        track_token_ids: Dict[str, int] = {}
        ids_track_token: Dict[int, str] = {}
        for i in range(len(past_tracked_objects)):
            agents = past_tracked_objects[i].get_tracked_objects_of_type(object_type)
            tensorized = torch.zeros((len(agents), WaymoAgentInternalIndex.dim()), dtype=torch.float32)
            max_agent_id = len(track_token_ids)

            for idx, agent in enumerate(agents):
                if agent.track_token not in track_token_ids:
                    track_token_ids[agent.track_token] = max_agent_id
                    ids_track_token[max_agent_id] = agent.track_token
                    max_agent_id += 1
                track_token_int = track_token_ids[agent.track_token]

                tensorized[idx, WaymoAgentInternalIndex.track_token()] = float(track_token_int)
                tensorized[idx, WaymoAgentInternalIndex.vx()] = agent.velocity.x
                tensorized[idx, WaymoAgentInternalIndex.vy()] = agent.velocity.y
                tensorized[idx, WaymoAgentInternalIndex.heading()] = agent.center.heading
                tensorized[idx, WaymoAgentInternalIndex.width()] = agent.box.width
                tensorized[idx, WaymoAgentInternalIndex.length()] = agent.box.length
                tensorized[idx, WaymoAgentInternalIndex.x()] = agent.center.x
                tensorized[idx, WaymoAgentInternalIndex.y()] = agent.center.y
                tensorized[idx, WaymoAgentInternalIndex.z()] = 0.0
                
            output.append(tensorized)
        return output, ids_track_token
    
    @torch.jit.unused
    def get_features_from_simulation(
            self, current_input: PlannerInput,
            initialization: PlannerInitialization) -> WaymoGenericAgents:
        """Inherited, see superclass."""
        pass

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        all_ego_states: List[EgoState],
        all_tracked_objects: List[TrackedObjects],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[
            str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param all_ego_states: The all states of the ego vehicle.
        :param all_time_stamps: The all time stamps of the input data.
        :param all_tracked_objects: The all tracked objects.
        :return: The packed tensors.
        """
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        all_ego_states_tensor = self.extract_ego_agent_tensor(all_ego_states)

        for feature_name in self._agent_features:
            all_tracked_objects_tensor_list, ids_track_token = self.extract_agent_tensor(
                all_tracked_objects, TrackedObjectType[feature_name])
            list_tensor_data[
                f"all_tracked_objects.{feature_name}"] = all_tracked_objects_tensor_list
            list_tensor_data[f"all_tracked_objects.{feature_name}.token_mapping"] = ids_track_token

        return (
            {
                "all_ego_states": all_ego_states_tensor,
            },
            list_tensor_data,
            {},
        )

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
        ego_state_index: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[
            str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[torch.Tensor]] = {}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}

        ego_history: torch.Tensor = tensor_data["all_ego_states"]
        anchor_ego_state = ego_history[ego_state_index, :].squeeze()

        # ego features
        ego_tensor = build_generic_ego_features_from_tensor(
            ego_history, ego_state_index)
        # Add ego width, length and type back to tensor
        ego_tensor[:, WaymoEgoInternalIndex.width():] = ego_history[
            ego_state_index, WaymoEgoInternalIndex.width():]
        ego_tensor[:, WaymoEgoInternalIndex.z()] = ego_tensor[:, WaymoEgoInternalIndex.z()] - anchor_ego_state[WaymoEgoInternalIndex.z()]
        output_list_dict["generic_agents.ego"] = [ego_tensor]
        # agent features
        for feature_name in self._agent_features:

            if f"all_tracked_objects.{feature_name}" in list_tensor_data:
                agents: List[torch.Tensor] = list_tensor_data[
                    f"all_tracked_objects.{feature_name}"]
                se2_agents = [a[:,:WaymoAgentInternalIndex.z()] for a in agents]
                agent_history = filter_agents_tensor(se2_agents, reverse=True)
                agent_token_mapping = list_tensor_data[f"all_tracked_objects.{feature_name}.token_mapping"]

                if agent_history[-1].shape[0] == 0:
                    # Return zero array when there are no agents in the scene
                    agents_tensor: torch.Tensor = torch.zeros(
                        (len(agent_history), 0,
                            self._agents_states_dim)).float()
                else:
                    agents_tensor: torch.Tensor = torch.zeros(
                        (len(agent_history), self._num_max_agents,
                            self._agents_states_dim)).float()
                    for i in range(len(agents)):
                        transfer_shape = min(agents[i].shape[0], self._num_max_agents)
                        agents_tensor[i, :transfer_shape, WaymoAgentInternalIndex.z()] =  agents[i][:transfer_shape, WaymoAgentInternalIndex.z()] - anchor_ego_state[WaymoEgoInternalIndex.z()]
                                        
                    agent_history = pad_agent_with_max(agent_history,
                                                       self._num_max_agents)

                    local_coords_agent_states = convert_absolute_quantities_to_relative(
                        agent_history, anchor_ego_state[:WaymoEgoInternalIndex.width()], position_only=True) # pass the validation

                    agents_tensor[..., :WaymoAgentInternalIndex.z()] = pack_agents_tensor_withindex(
                        local_coords_agent_states)

                raw_id_tensor = torch.zeros((agents_tensor.shape[1])).int()
                for i in range(agents_tensor.shape[1]):
                    # if not nan
                    if not torch.isnan(agents_tensor[0, i, WaymoAgentInternalIndex.track_token()]):
                        raw_id_tensor[i] = int(agent_token_mapping[
                            int(agents_tensor[0, i, WaymoAgentInternalIndex.track_token()])])

                output_list_dict[f"generic_agents.agents.{feature_name}"] = [
                    agents_tensor
                ]
                output_list_dict[f"generic_agents.agents.{feature_name}.raw_ids"] = [
                    raw_id_tensor
                ]
         
          
        return output_dict, output_list_dict, output_list_list_dict

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> WaymoGenericAgents:
        """
        Unpacks the data returned from the scriptable core into an WaymoGenericAgents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed WaymoGenericAgents object.
        """
        ego_features = [list_tensor_data["generic_agents.ego"][0].detach().numpy()]
        agent_features = {}
        for key in list_tensor_data:
            if key.startswith("generic_agents.agents."):
                feature_name = key[len("generic_agents.agents.") :]
                if isinstance(list_tensor_data[key][0], torch.Tensor):
                    agent_features[feature_name] = [
                        list_tensor_data[key][0].detach().numpy()]
                else:
                    agent_features[feature_name] = [
                        list_tensor_data[key][0]]

        return WaymoGenericAgents(ego=ego_features, agents=agent_features)
