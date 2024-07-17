from __future__ import annotations

import os
import pickle
from typing import Generator, List, Optional, Set

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import (
    StateSE2, StateVector2D, TimePoint)
from nuplan.common.actor_state.tracked_objects import (TrackedObject,
                                                       TrackedObjects)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import (VehicleParameters)
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, Transform, TrafficLightStatusType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.utils.helpers import get_unique_incremental_track_id
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import \
    sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks, Sensors)
from nuplan.planning.simulation.trajectory.trajectory_sampling import \
    TrajectorySampling
from nuplan_extent.common.maps.wod_map.map_factory import get_maps_api
from waymo_open_dataset.protos.scenario_pb2 import ObjectState

wod_local2agent_type = {
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "BICYCLE",
    4: "GENERIC_OBJECT",
}

wod_traffic_light_type = {
    0: TrafficLightStatusType.UNKNOWN,
    1: TrafficLightStatusType.RED,
    2: TrafficLightStatusType.YELLOW,
    3: TrafficLightStatusType.GREEN,
    4: TrafficLightStatusType.RED,
    5: TrafficLightStatusType.YELLOW,
    6: TrafficLightStatusType.GREEN,
    7: TrafficLightStatusType.RED,
    8: TrafficLightStatusType.YELLOW,
}

class WodScenario(AbstractScenario):
    """_summary_

    Args:
        BaseScenario (_type_): _description_
    """

    def __init__(
            self,
            data_root: str,
            split: str,
            scenario_id: str,
            agent_idx: str,
    ) -> None:
        self._data_root = data_root
        self._split = split
        self._scenario_id = scenario_id
        self._agent_idx = int(agent_idx)
        self._scenario_type = "waymo_open_dataset"
        self._pickle_path = os.path.join(self._data_root, self._split,
                                         self._scenario_id + ".pkl")
        self.ego_type = 1
        self.current_index = 10
        self.scenario_length = 91
        self.agent_id = None
        self.interactive_agent_id = None
        self.interactive_agent_idx = None
        self.interactive_agent_type = None

        self._database_row_interval = 0.1

    @property
    def split(self) -> str:
        """Inherited, see superclass."""
        return self._split

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        return get_maps_api(
            os.path.join(self._data_root, self._split), self._scenario_id)

    @property
    def initial_ego_state(self) -> EgoState:
        """
        Return the initial ego state
        :return: EgoState of ego.
        """
        return self.get_ego_state_at_iteration(self.current_index)

    @property
    def initial_tracked_objects(self) -> DetectionsTracks:
        """
        Get initial tracked objects
        :return: DetectionsTracks.
        """
        return self.get_tracked_objects_at_iteration(self.current_index)

    def load_agent_tracks(self) -> List[TrackedObject]:
        with open(self._pickle_path, 'rb') as f:
            data = pickle.load(f)
            all_agent_tracks = data.tracks
            if "interactive" in self._split:
                assert self._agent_idx == data.tracks_to_predict[0].track_index
        self.agent_id = all_agent_tracks[self._agent_idx].id
        self.ego_type = all_agent_tracks[self._agent_idx].object_type
        if "interactive" in self._split:
            self.interactive_agent_idx = int(data.tracks_to_predict[1].track_index)
            self.interactive_agent_id = all_agent_tracks[self.interactive_agent_idx].id
            self.interactive_agent_type = all_agent_tracks[self.interactive_agent_idx].object_type

        return all_agent_tracks

    def transform_state_to_EgoState(self, state: ObjectState,
                                    iteration: int) -> EgoState:
        """Transforms a state from the dataset to an EgoState"""
        if state.valid:
            x = state.center_x
            y = state.center_y
            z = state.center_z
            heading = state.heading
            width = state.width
            length = state.length
            height = state.height
            velocity_x = state.velocity_x
            velocity_y = state.velocity_y
        else:
            x = np.nan
            y = np.nan
            z = np.nan
            heading = np.nan
            width = np.nan
            length = np.nan
            height = np.nan
            velocity_x = np.nan
            velocity_y = np.nan
        return EgoState.build_from_rear_axle(
            StateSE2(x, y, heading),
            tire_steering_angle=0.0,
            vehicle_parameters=VehicleParameters(
                vehicle_name="pacifica",  # same as Nuplan pacifica
                vehicle_type="gen1",  # same as Nuplan pacifica
                width=width,
                front_length=length * 0.75,
                rear_length=length * 0.25,
                wheel_base=3.089,  # same as Nuplan pacifica
                cog_position_from_rear_axle=1.67,  # same as Nuplan pacifica
                height=height,
            ),
            time_point=TimePoint(max(1, iteration * 1e5)),
            rear_axle_velocity_2d=StateVector2D(velocity_x, velocity_y),
            rear_axle_acceleration_2d=StateVector2D(
                x=0, y=0),  # acceleration is not available in the dataset
        )

    def transform_state_to_TrackedObject(self, state: ObjectState,
                                         object_type: int, iteration: int,
                                         object_id: int) -> Agent:
        """Transforms a state from the dataset to a TrackedObject"""
        tracked_object_type = TrackedObjectType[
            wod_local2agent_type[object_type]]
        if state.valid:
            x = state.center_x
            y = state.center_y
            z = state.center_z
            heading = state.heading
            width = state.width
            length = state.length
            height = state.height
            velocity_x = state.velocity_x
            velocity_y = state.velocity_y
        else:
            x = np.nan
            y = np.nan
            z = np.nan
            heading = np.nan
            width = np.nan
            length = np.nan
            height = np.nan
            velocity_x = np.nan
            velocity_y = np.nan        

        pose = StateSE2(x, y, heading)
        oriented_box = OrientedBox(pose,
                                   width=width,
                                   length=length,
                                   height=height)
        
        token = "{}_{}".format(self._scenario_id, str(object_id))

        return Agent(
            tracked_object_type=tracked_object_type,
            oriented_box=oriented_box,
            velocity=StateVector2D(velocity_x, velocity_y),
            predictions=[],  # to be filled in later
            angular_velocity=np.nan,
            metadata=SceneObjectMetadata(
                token=token,
                track_token=str(object_id),
                track_id=get_unique_incremental_track_id(str(token)),
                timestamp_us=max(1, iteration * 1e5),
                category_name=wod_local2agent_type[object_type],
            ),
        )

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""
        all_agent_tracks = self.load_agent_tracks()
        state = all_agent_tracks[self._agent_idx].states[iteration]
        return self.transform_state_to_EgoState(state, iteration)

    def get_ego_past_trajectory(self,
                                iteration: int = 10,
                                time_horizon: float = 1.0,
                                num_samples: Optional[int] = None
                                ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        all_agent_tracks = self.load_agent_tracks()
        num_samples = num_samples if num_samples else int(
            time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon,
                                                   self._database_row_interval)
        past_indices = [self.current_index - idx for idx in indices[::-1]]
        for idx in past_indices:
            yield self.transform_state_to_EgoState(
                all_agent_tracks[self._agent_idx].states[idx], idx)

    def get_ego_future_trajectory(self,
                                  iteration: int,
                                  time_horizon: float,
                                  num_samples: Optional[int] = None
                                  ) -> Generator[EgoState, None, None]:
        """
        Find ego future trajectory
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the future ego trajectory with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """
        all_agent_tracks = self.load_agent_tracks()
        num_samples = num_samples if num_samples else int(
            time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon,
                                                   self._database_row_interval)
        future_indices = [self.current_index + idx for idx in indices]
        return [
            self.transform_state_to_EgoState(
                all_agent_tracks[self._agent_idx].states[idx], idx)
            for idx in future_indices
        ]

    def get_ego_all_trajectory(self) -> List[EgoState]:
        all_agent_tracks = self.load_agent_tracks()
        return [self.transform_state_to_EgoState(all_agent_tracks[self._agent_idx].states[idx], idx) for idx in range(self.scenario_length)]

    def get_agent_all_trajectory(self, agent_idx: int) -> List[Agent]:
        all_agent_tracks = self.load_agent_tracks()
        agent_type = all_agent_tracks[agent_idx].object_type
        agent_id = all_agent_tracks[agent_idx].id
        return [self.transform_state_to_TrackedObject(all_agent_tracks[agent_idx].states[idx], agent_type, idx, agent_id) for idx in range(self.scenario_length)]

    def get_tracked_objects_at_iteration(
            self,
            iteration: int,
            future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        all_agent_tracks = self.load_agent_tracks()
        tracked_objects: List[TrackedObject] = []
        for track_idx, track in enumerate(all_agent_tracks):
            if track_idx == self._agent_idx:
                continue
            tracked_objects.append(
                self.transform_state_to_TrackedObject(track.states[iteration],
                                                      track.object_type,
                                                      iteration, track.id))
        return DetectionsTracks(
            TrackedObjects(tracked_objects=tracked_objects))

    def get_past_tracked_objects(
            self,
            iteration: int = 10,
            time_horizon: float = 1.0,
            num_samples: Optional[int] = None,
            future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        all_agent_tracks = self.load_agent_tracks()
        num_samples = num_samples if num_samples else int(
            time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon,
                                                   self._database_row_interval)
        past_indices = [self.current_index - idx for idx in indices[::-1]]
        for idx in past_indices:
            tracked_objects: List[TrackedObject] = []
            for track_idx, track in enumerate(all_agent_tracks):
                if track_idx == self._agent_idx:
                    continue
                tracked_objects.append(
                    self.transform_state_to_TrackedObject(
                        track.states[idx], track.object_type, idx, track.id))
            yield DetectionsTracks(
                TrackedObjects(tracked_objects=tracked_objects))

    def get_future_tracked_objects(
            self,
            iteration: int,
            time_horizon: float,
            num_samples: Optional[int] = None
    ) -> Generator[DetectionsTracks, None, None]:
        """
        Find future detections
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the past detections
        """
        all_agent_tracks = self.load_agent_tracks()
        num_samples = num_samples if num_samples else int(
            time_horizon / self.database_interval)
        indices = sample_indices_with_time_horizon(num_samples, time_horizon,
                                                   self._database_row_interval)
        future_indices = [self.current_index + idx for idx in indices]
        for idx in future_indices:
            tracked_objects: List[TrackedObject] = []
            for track_idx, track in enumerate(all_agent_tracks):
                if track_idx == self._agent_idx:
                    continue
                tracked_objects.append(
                    self.transform_state_to_TrackedObject(
                        track.states[idx], track.object_type, idx, track.id))
            yield DetectionsTracks(
                TrackedObjects(tracked_objects=tracked_objects))

    @property
    def token(self) -> str:
        return self._scenario_id + "_" + str(self._agent_idx)

    @property
    def log_name(self) -> str:
        return self._scenario_id

    @property
    def scenario_name(self) -> str:
        return self._scenario_id

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        raise NotImplementedError(
            "Vehicle params for NuScenes scenario has not been implemented yet."
        )

    @property
    def scenario_type(self) -> str:
        return self._scenario_type

    @property
    def database_interval(self) -> float:
        return self._database_row_interval

    def get_number_of_iterations(self) -> int:
        """
        Get how many frames does this scenario contain
        :return: [int] representing number of scenarios.
        """
        return self.scenario_length

    def get_time_point(self, iteration: int) -> TimePoint:
        """
        Get time point of the iteration
        :param iteration: iteration in scenario 0 <= iteration < number_of_iterations
        :return: global time point.
        """

    def get_lidar_to_ego_transform(self) -> Transform:
        """
        Return the transformation matrix between lidar and ego
        :return: [4x4] rotation and translation matrix.
        """

    def get_mission_goal(self) -> StateSE2:
        """
        Goal far into future (in generally more than 100m far beyond scenario length).
        :return: StateSE2 for the final state.
        """

    def get_route_roadblock_ids(self) -> List[str]:
        """
        Get list of roadblock ids comprising goal route.
        :return: List of roadblock id strings.
        """

    def get_expert_goal_state(self) -> StateSE2:
        """
        Get the final state which the expert driver achieved at the end of the scenario
        :return: StateSE2 for the final state.
        """

    def get_sensors_at_iteration(self, iteration: int) -> Sensors:
        """
        Return sensor from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: Sensors.
        """

    def get_traffic_light_status_at_iteration(
            self,
            iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """
        Get traffic light status at an iteration.
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return traffic light status at the iteration.
        """
        with open(self._pickle_path, 'rb') as f:
            data = pickle.load(f)
        select_dynamic_map_states = data.dynamic_map_states[iteration]
        for lane_state in select_dynamic_map_states.lane_states:
            yield TrafficLightStatusData(
                status=wod_traffic_light_type[lane_state.state],
                lane_connector_id=lane_state.lane,
                timestamp=max(1, iteration * 1e5),
            )
            
    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ):
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        pass

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ):
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        pass
               

    def get_future_timestamps(self,
                              iteration: int,
                              time_horizon: float,
                              num_samples: Optional[int] = None
                              ) -> Generator[TimePoint, None, None]:
        """
        Find timesteps in future
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the future timestamps with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """

    def get_past_timestamps(self,
                            iteration: int,
                            time_horizon: float,
                            num_samples: Optional[int] = None
                            ) -> Generator[TimePoint, None, None]:
        """
        Find timesteps in past
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the past
        :param time_horizon: the desired horizon to the past
        :return: the future timestamps with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """

    def get_past_sensors(self,
                         iteration: int,
                         time_horizon: float,
                         num_samples: Optional[int] = None
                         ) -> Generator[Sensors, None, None]:
        """
        Find past sensors
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the past sensors with the best matching entries to the desired time_horizon/num_samples
        timestamp (best matching to the database)
        """

    def get_tracked_objects_within_time_window_at_iteration(
            self,
            iteration: int,
            past_time_horizon: float,
            future_time_horizon: float,
            filter_track_tokens: Optional[Set[str]] = None,
            future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """
        Gets all tracked objects present within a time window that stretches from past_time_horizon before the iteration to future_time_horizon afterwards.
        Also optionally filters the included results on the provided track_tokens.
        Results will be sorted by object type, then by timestamp, then by track token.
        :param iteration: The iteration of the scenario to query.
        :param past_time_horizon [s]: The amount of time to look into the past from the iteration timestamp.
        :param future_time_horizon [s]: The amount of time to look into the future from the iteration timestamp.
        :param filter_track_tokens: If provided, then the results will be filtered to only contain objects with
            track_tokens included in the provided set. If None, then all results are returned.
        :param future_trajectory_sampling: sampling parameters of agent future ground truth predictions if desired.
        :return: The retrieved detection tracks.
        """
        pass
