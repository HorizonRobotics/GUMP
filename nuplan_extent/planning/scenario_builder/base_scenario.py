from typing import Generator, List, Optional, Set

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, Transform
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks, Sensors)


class BaseScenario(AbstractScenario):
    """Base Scenario that is used in training and simulation.
    """

    def get_input_at_iteration(self, iteration: int):
        """Generate input at a certain iteration as the initial frame.

        :param iteration: index of the initial frame in the scene.

        Return list of Cameras dataclass, future ego motions and intention
        for every context frame.
        """
        cameras = []
        future_egomotion = []
        command = []
        if iteration + self._past_steps + self._predict_steps <= self._queue_length:
            for i in range(self._past_steps):
                index = iteration + i
                cameras.append(self.cameras[index])
                future_egomotion.append(self.future_egomotion[index])
                command.append(self.command[index])

        return cameras, future_egomotion, command

    def get_ego_trajectory_at_iteration(self, iteration: int):
        """Generate future trajectory at a certain iteration as the initial frame.

        :param iteration: index of the initial frame in the scene.

        Return future trajectory for the current frame.
        """
        if iteration + self._past_steps + self._predict_steps <= self._queue_length:
            index = iteration + self._past_steps - 1
            return self.gt_sdc_fut_traj[index]
        else:
            return None

    @property
    def total_steps(self) -> int:
        return self._past_steps + self._predict_steps

    @property
    def trim_scenario_end(self) -> bool:
        """
        In sequential mode, trim scenario max len at the end, because we don't have
        additional frames at the end of the scenario for future.
        Only applies to NuScenes and Carla.
        """
        return True

    @property
    def token(self) -> str:
        return self.sample_token[0]

    @property
    def log_name(self) -> str:
        return self.scene_token

    @property
    def scenario_name(self) -> str:
        return self.sample_token[0]

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        raise NotImplementedError(
            "Vehicle params for NuScenes scenario has not been implemented yet."
        )

    @property
    def scenario_type(self) -> str:
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        return None

    @property
    def database_interval(self) -> float:
        return 0.05

    def get_number_of_iterations(self) -> int:
        """
        Get how many frames does this scenario contain
        :return: [int] representing number of scenarios.
        """
        return self._queue_length

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

    def get_tracked_objects_at_iteration(self,
                                         iteration: int) -> DetectionsTracks:
        """
        Return tracked objects from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: DetectionsTracks.
        """

    @property
    def initial_tracked_objects(self) -> DetectionsTracks:
        """
        Get initial tracked objects
        :return: DetectionsTracks.
        """
        return self.get_tracked_objects_at_iteration(0)

    def get_sensors_at_iteration(self, iteration: int) -> Sensors:
        """
        Return sensor from iteration
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: Sensors.
        """

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """
        Return ego (expert) state in a dataset
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return: EgoState of ego.
        """

    def get_traffic_light_status_at_iteration(
            self,
            iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """
        Get traffic light status at an iteration.
        :param iteration: within scenario 0 <= iteration < number_of_iterations
        :return traffic light status at the iteration.
        """

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

    def get_ego_past_trajectory(self,
                                iteration: int,
                                time_horizon: float,
                                num_samples: Optional[int] = None
                                ) -> Generator[EgoState, None, None]:
        """
        Find ego past trajectory
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the past ego trajectory with the best matching entries to the desired time_horizon/num_samples
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

    def get_past_tracked_objects(self,
                                 iteration: int,
                                 time_horizon: float,
                                 num_samples: Optional[int] = None
                                 ) -> Generator[DetectionsTracks, None, None]:
        """
        Find past detections
        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations
        :param num_samples: number of entries in the future
        :param time_horizon: the desired horizon to the future
        :return: the past detections
        """

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

    def get_tracked_objects_within_time_window_at_iteration(
            self,
            iteration: int,
            past_time_horizon: float,
            future_time_horizon: float,
            filter_track_tokens: Optional[Set[str]] = None,
    ) -> DetectionsTracks:
        """
        Gets all tracked objects present within a time window that stretches from past_time_horizon before the iteration to future_time_horizon afterwards.
        Also optionally filters the included results on the provided track_tokens.
        Results will be sorted by object type, then by timestamp, then by track token.
        :param iteration: The iteration of the scenario to query.
        :param past_time_horizon: The amount of time to look into the past from the iteration timestamp.
        :param future_time_horizon: The amount of time to look into the future from the iteration timestamp.
        :param filter_track_tokens: If provided, then the results will be filtered to only contain objects with track_tokens included in the provided set. If None, then all results are returned.

        :return: The retrieved detection tracks.
        """
