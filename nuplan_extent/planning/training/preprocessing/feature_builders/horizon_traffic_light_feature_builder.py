from typing import Dict, List, Tuple, Type, cast, Optional, Generator

import torch
import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder, AbstractModelFeature)
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
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType, TrafficLightStatusData
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import (
    pad_agent_with_max,
    pack_agents_tensor_withindex,
    build_generic_ego_features_from_tensor,
)
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_traffic_light_dict_from_generator,
)
from nuplan_extent.planning.training.preprocessing.features.vector_utils import (
    get_vectorized_traffic_light_data,
)
from nuplan_extent.planning.training.preprocessing.features.traffic_light_data import TrafficLightData


class TrafficLightFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, past_trajectory_sampling: TrajectorySampling,
                 future_trajectory_sampling: TrajectorySampling, max_num_traffic_lights: int = 29) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self._num_past_poses = past_trajectory_sampling.num_poses
        self._past_time_horizon = past_trajectory_sampling.time_horizon
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._future_time_horizon = future_trajectory_sampling.time_horizon
        self._max_num_traffic_lights = max_num_traffic_lights
        
    def get_future_traffic_light_state(
        self, time_horizon: float, num_samples: Optional[int], scenario: AbstractScenario
    ) -> List[Dict[str, TrafficLightStatusType]]:       
        """
        extract future traffic light state
        """    
        database_interval = scenario.database_interval
        database_row_interval = scenario._database_row_interval
        num_samples = num_samples if num_samples else int(time_horizon / database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, database_row_interval)
        traffic_light_by_iter_dict_future = []
        for iteration in indices:
            traffic_light_generator = scenario.get_traffic_light_status_at_iteration(iteration=iteration)
            traffic_light_by_iter_dict = get_traffic_light_dict_from_generator(traffic_light_generator)
            traffic_light_by_iter_dict_future.append(traffic_light_by_iter_dict)
        return traffic_light_by_iter_dict_future

    def get_current_traffic_light_state(
        self, scenario: AbstractScenario
    ) -> List[Dict[str, TrafficLightStatusType]]:  
        """
        extract currrent traffic light state
        """         
        traffic_light_by_iter_dict_current = []
        traffic_light_generator = scenario.get_traffic_light_status_at_iteration(iteration=0)
        traffic_light_by_iter_dict = get_traffic_light_dict_from_generator(traffic_light_generator)
        traffic_light_by_iter_dict_current.append(traffic_light_by_iter_dict)
        return traffic_light_by_iter_dict_current
        
    def get_past_traffic_light_state(
        self, time_horizon: float, num_samples: Optional[int], scenario: AbstractScenario
    ) -> List[Dict[str, TrafficLightStatusType]]:
        database_interval = scenario.database_interval
        database_row_interval = scenario._database_row_interval
        num_samples = num_samples if num_samples else int(time_horizon / database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon, database_row_interval)
        traffic_light_by_iter_dict_past = []
        for iteration in indices:
            traffic_light_generator = scenario.get_traffic_light_status_at_iteration(iteration=-iteration)
            traffic_light_by_iter_dict = get_traffic_light_dict_from_generator(traffic_light_generator)
            traffic_light_by_iter_dict_past.append(traffic_light_by_iter_dict)
        return traffic_light_by_iter_dict_past
        
    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "traffic_lights"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return TrafficLightData  # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> GenericAgents:
        """
        Retrieves traffic light features from the provided scenario.

        This method extracts the state of traffic lights at different times (past, present, future) 
        and processes this data into a format suitable for further use in the model.

        Args:
            scenario (AbstractScenario): The scenario object from which to extract traffic light data.

        Returns:
            GenericAgents: An object containing processed traffic light data.
        """
        # Disable gradient calculations for efficiency
        with torch.no_grad():
            # Retrieve current ego state
            ego_state = scenario.get_ego_state_at_iteration(iteration=0)
            # Access map API from the scenario
            map_api = scenario.map_api
            # Get past, current, and future traffic light states
            traffic_light_by_iter_dict_past = self.get_past_traffic_light_state(
                time_horizon=self._past_time_horizon,
                num_samples=self._num_past_poses,
                scenario=scenario,
            )
            traffic_light_by_iter_dict_current = self.get_current_traffic_light_state(
                scenario=scenario,
            )
            traffic_light_by_iter_dict_future = self.get_future_traffic_light_state(
                time_horizon=self._future_time_horizon,
                num_samples=self._num_future_poses,
                scenario=scenario,
            )
            # Combine past, current, and future traffic light states
            traffic_light_by_iter_list = traffic_light_by_iter_dict_past + traffic_light_by_iter_dict_current + traffic_light_by_iter_dict_future
            # Compute traffic light feature vectors
            traffic_light_vector_list = self._compute_feature(ego_state, map_api, traffic_light_by_iter_list)
            # Create traffic light data object
            traffic_light_data = TrafficLightData(data=traffic_light_vector_list)
            # # Ensure the data size does not exceed the maximum allowed traffic lights
            # assert traffic_light_data.data.shape[1] <= self._max_num_traffic_lights
        return traffic_light_data
            
    def _compute_feature(self, ego_state: EgoState, map_api: AbstractMap, traffic_light_by_iter_list: List[Dict[str, TrafficLightStatusType]]) -> List:
        """
        Computes feature vectors for traffic lights.

        Args:
            ego_state (EgoState): The current state of the ego vehicle.
            map_api (AbstractMap): The map API providing contextual map information.
            traffic_light_by_iter_list (List[Dict[str, TrafficLightStatusType]]): A list of traffic light states.

        Returns:
            List: A list of vectorized traffic light feature data.
        """
        traffic_light_vector_list = []
        # Iterate over traffic light states and vectorize data
        for traffic_light_by_iter in traffic_light_by_iter_list:
            traffic_light_vector_data = get_vectorized_traffic_light_data(
                ego_state, map_api, traffic_light_by_iter,
            )
            traffic_light_vector_list.append(traffic_light_vector_data)
        # Remap lane IDs in traffic light vectors
        traffic_light_vector_list = self._remapping_lane_id(traffic_light_vector_list)
        return traffic_light_vector_list


    def _remapping_lane_id(self, traffic_light_vector_list: List) -> List:
        """
        Remaps lane IDs in traffic light vector data for consistent representation across time steps.

        This method adjusts lane IDs in the traffic light data to ensure that they are consistent
        with the current state of the scenario, especially considering lane changes or updates. It
        also handles the traffic light status, setting it to UNKNOWN if a lane ID from the current
        traffic light vector is not present in other vectors.

        Args:
            traffic_light_vector_list (List): A list of traffic light vectors, each containing
                                            original lane IDs and other traffic light data.

        Returns:
            List: Updated list of traffic light vectors with remapped lane IDs.
        """
        remapped_traffic_light_vector_list = []
        current_traffic_light_vector = traffic_light_vector_list[self._num_past_poses]
        
        current_lane_id_list = [tl_vector['lane_id'] for tl_vector in current_traffic_light_vector]
        current_lane_id_mapping = {tl_vector['lane_id']: (index+1) for index, tl_vector in enumerate(current_traffic_light_vector)}
        for i in range(len(traffic_light_vector_list)):
            remapped_traffic_light_vector_list.append([])
            lane_id_list = [tl_vector['lane_id'] for tl_vector in traffic_light_vector_list[i]]
            for current_lane_id, current_lane_index in current_lane_id_mapping.items():
                if current_lane_id in lane_id_list:
                    index = lane_id_list.index(current_lane_id)
                    traffic_light_status = traffic_light_vector_list[i][index]['traffic_light_status']
                    lane_coords = traffic_light_vector_list[i][index]['lane_coords']
                    index = current_lane_id_list.index(current_lane_id)
                    tl_index = current_traffic_light_vector[index]['tl_index']
                else:
                    index = current_lane_id_list.index(current_lane_id)
                    traffic_light_status = TrafficLightStatusType.UNKNOWN
                    lane_coords = current_traffic_light_vector[index]['lane_coords']
                    tl_index = current_traffic_light_vector[index]['tl_index']
                    
                # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
                remapped_tl_vector = {
                    'lane_id': current_lane_index,
                    'tl_index': tl_index,
                    'lane_coords': lane_coords[:2],
                    'traffic_light_status': traffic_light_status,
                }
                remapped_traffic_light_vector_list[-1].append(remapped_tl_vector)
        return remapped_traffic_light_vector_list

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> GenericAgents:
        """
        Retrieves traffic light features from the provided scenario.

        This method extracts the state of traffic lights at different times (past, present, future) 
        and processes this data into a format suitable for further use in the model.

        Args:
            scenario (AbstractScenario): The scenario object from which to extract traffic light data.

        Returns:
            GenericAgents: An object containing processed traffic light data.
        """
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

            ego_state_index = len(sampled_past_ego_states)
            sampled_past_ego_states = sampled_past_ego_states + \
                [present_ego_state]
            time_stamps = [state.time_point for state in sampled_past_ego_states]
            
            map_api = initialization.map_api
            scenario = initialization.scenario
            traffic_light_by_iter = current_input.traffic_light_data
            traffic_light_by_iter_dict_current = [get_traffic_light_dict_from_generator(traffic_light_by_iter)]
            traffic_light_by_iter_dict_past = self.get_past_traffic_light_state(
                time_horizon=self._past_time_horizon,
                num_samples=self._num_past_poses,
                scenario=scenario,
            )
            # current_input.history
            # Combine past, current, and future traffic light states
            traffic_light_by_iter_list = traffic_light_by_iter_dict_past + traffic_light_by_iter_dict_current 
            # Compute traffic light feature vectors
            traffic_light_vector_list = self._compute_feature(present_ego_state, map_api, traffic_light_by_iter_list)
            # Create traffic light data object
            traffic_light_data = TrafficLightData(data=traffic_light_vector_list)
        return traffic_light_data
        
