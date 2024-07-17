from abc import ABC, abstractmethod
import copy
import math
import numpy as np
import torch
from typing import Any, Dict, List, Tuple

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature


class MissingFeatureAtCenterException(Exception):
    pass


class MissingFeatureAtIterationException(Exception):
    pass


class MissingFeatureException(Exception):
    pass


NONEXISTENT_FEATURE = object()

class NpAgentState(np.ndarray):
    """A numpy array wrapper for other agent state."""
    dim = 8
    track_token_index = 0
    vx_index = 1
    vy_index = 2
    heading_index = 3
    width_index = 4
    length_index = 5
    x_index = 6
    y_index = 7

    @property
    def data(self):
        return self
    
    @property
    def track_token(self):
        return self[..., 0]
    
    @property
    def vx(self):
        return self[..., 1]
    
    @property
    def vy(self):
        return self[..., 2]
    
    @property
    def heading(self):
        return self[..., 3]
    
    @property
    def width(self):
        return self[..., 4]
    
    @property
    def length(self):
        return self[..., 5]
    
    @property
    def x(self):
        return self[..., 6]
    
    @property
    def y(self):
        return self[..., 7]


    def bounding_box(self):
        """Get the bounding box of the ego vehicle.

        :return The bounding box of the ego vehicle, represented by 5 numbers:
            center_x, center_y, heading, half_length, half_width
        """
        x = self.x
        y = self.y
        heading = self.heading
        half_length = self.length / 2
        half_width = self.width / 2
        return np.stack([x, y, heading, half_length, half_width], axis=-1)
    
    def transform(self, trans, heading):
        xy = np.stack([self.x, self.y], axis=-1) @ trans[:2, :2].T + trans[:2, 2]
        vxy = np.stack([self.vx, self.vy], axis=-1) @ trans[:2, :2].T
        self[..., self.x_index] = xy[..., 0]
        self[..., self.y_index] = xy[..., 1]
        self[..., self.vx_index] = vxy[..., 0]
        self[..., self.vy_index] = vxy[..., 1]
        self[..., self.heading_index] = self.heading - heading
    
    def inverse_transform(self, trans, heading):
        xy = (np.stack([self.x, self.y], axis=-1) - trans[:2, 2]) @ np.linalg.inv(trans[:2, :2].T)
        vxy = np.stack([self.vx, self.vy], axis=-1) @ np.linalg.inv(trans[:2, :2].T)
        self[..., self.x_index] = xy[..., 0]
        self[..., self.y_index] = xy[..., 1]
        self[..., self.vx_index] = vxy[..., 0]
        self[..., self.vy_index] = vxy[..., 1]
        self[..., self.heading_index] = self.heading + heading
    

class NpEgoState(np.ndarray):
    """A numpy array wrapper for ego state."""

    dim = 17
    time_index = 0
    x_index = 1
    y_index = 2
    heading_index = 3
    vel_lon_index = 4
    vel_lat_index = 5
    acc_lon_index = 6
    acc_lat_index = 7
    tire_steering_angle_index = 8
    yaw_rate_index = 9
    yaw_acc_index = 10
    wheel_base_index = 15

    @property
    def data(self):
        return self

    @property
    def time(self):
        # in seconds
        return self[..., 0]

    @property
    def x(self):
        """Real axle x coordinate."""
        return self[..., 1]

    @property
    def y(self):
        """Real axle y coordinate."""
        return self[..., 2]

    @property
    def heading(self):
        return self[..., 3]

    @property
    def vel_lon(self):
        return self[..., 4]

    @property
    def vel_lat(self):
        return self[..., 5]

    @property
    def acc_lon(self):
        return self[..., 6]

    @property
    def acc_lat(self):
        return self[..., 7]

    @property
    def tire_steering_angle(self):
        return self[..., 8]

    @property
    def yaw_rate(self):
        return self[..., 9]

    @property
    def yaw_acc(self):
        return self[..., 10]

    @property
    def width(self):
        return self[..., 11]

    @property
    def front_length(self):
        return self[..., 12]

    @property
    def rear_length(self):
        return self[..., 13]

    @property
    def cog_position_from_rear_axle(self):
        return self[..., 14]

    @property
    def wheel_base(self):
        return self[..., 15]

    @property
    def height(self):
        return self[..., 16]

    @property
    def rear_axle_pose(self):
        return self[..., self.x_index:self.x_index + 3]

    @property
    def rear_axle_velocity_2d(self):
        return self[..., self.vel_lon_index:self.vel_lon_index + 2]

    @property
    def rear_axle_acceleration_2d(self):
        return self[..., self.acc_lon_index:self.acc_lon_index + 2]

    def bounding_box(self):
        """Get the bounding box of the ego vehicle.

        :return The bounding box of the ego vehicle, represented by 5 numbers:
            center_x, center_y, heading, half_length, half_width
        """
        heading = self.heading
        c = np.cos(heading)
        s = np.sin(heading)
        half_length = (self.rear_length + self.front_length) / 2
        half_width = self.width / 2
        d = half_length - self.rear_length
        x = self.x + d * c
        y = self.y + d * s
        return np.stack([x, y, heading, half_length, half_width], axis=-1)

    def to_nuplan_ego_state(self) -> EgoState:
        assert self.ndim == 1, "Only support converting 1 ego state at a time"
        vehicle_parameters = VehicleParameters(
            width=self.width,
            front_length=self.front_length,
            rear_length=self.rear_length,
            cog_position_from_rear_axle=self.cog_position_from_rear_axle,
            wheel_base=self.wheel_base,
            height=self.height if self.height == 0.0 else None,
            vehicle_name="pacifica",  # same as Nuplan pacifica
            vehicle_type="gen1",  # same as Nuplan pacifica
        )
        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(*self.rear_axle_pose),
            rear_axle_velocity_2d=StateVector2D(*self.rear_axle_velocity_2d),
            rear_axle_acceleration_2d=StateVector2D(*self.rear_axle_acceleration_2d),
            tire_steering_angle=self.tire_steering_angle,
            time_point=TimePoint(1e6 * float(self.time)),
            vehicle_parameters=vehicle_parameters,
        )

    def transform(self, trans, heading):
        xy = np.stack([self.x, self.y], axis=-1) @ trans[:2, :2].T + trans[:2, 2]
        self[..., self.x_index] = xy[..., 0]
        self[..., self.y_index] = xy[..., 1]
        self[..., self.heading_index] = self.heading - heading
    
    def inverse_transform(self, trans, heading):
        xy = (np.stack([self.x, self.y], axis=-1) - trans[:2, 2]) @ np.linalg.inv(trans[:2, :2].T)
        self[..., self.x_index] = xy[..., 0]
        self[..., self.y_index] = xy[..., 1]
        self[..., self.heading_index] = self.heading + heading
    
    def with_range(self, map_range, delta):
        mask = (self.x >= (map_range[0] + delta)) & (self.x <= (map_range[1] - delta)) & (self.y >= (map_range[2] + delta)) & (self.y <= (map_range[3] - delta))
        return mask.view(np.ndarray)


def _radian(d):
    """Convert the given radian to [-pi, pi] range."""
    pi = math.pi
    if d > pi:
        return d - 2 * pi
    elif d < -pi:
        return d + 2 * pi
    else:
        return d


def _get_ego_state(scenario: AbstractScenario, iter: int):
    """Get the ego state at the given iteration.

    :param scenario: The scenario to get the ego state from.
    :param iter: The iteration to get the ego state for.
    """
    if iter < 0:
        gen = scenario.get_ego_past_trajectory(
            0, -iter * scenario.database_interval, 1)
    elif iter >= scenario.get_number_of_iterations():
        n = scenario.get_number_of_iterations() - 1
        gen = scenario.get_ego_future_trajectory(
            n, (iter - n) * scenario.database_interval, 1)
    else:
        return scenario.get_ego_state_at_iteration(iter)
    ego_states = list(gen)
    if len(ego_states) == 0:
        return None
    else:
        return ego_states[0]


class PreparedScenario(AbstractModelFeature):
    """Prepared scenario.

    A prepared scenario stores precomputed features to facilitate faster feature
    extraction during training.

    The main mechanism is "add_feature/get_feature" series of methods.
    The "add_feature" methods are used to add features to the prepared scenario.
    The "get_feature" methods are used to get features from the prepared scenario.

    During the stage of preparing the scenario, if a feature can be found in
    the prepared scenario, there is no need to calculate it again.
    """

    def __init__(self):
        self._data = {}
        self._backup = {}
        self._mission_goal = None

    def serialize(self) -> Dict[str, Any]:
        assert self._scenario is not None, "Scenario is not prepared yet"
        self._populate_yaw_rate_and_acc()
        return self._data

    def _populate_yaw_rate_and_acc(self):
        """Populate yaw_rate and yaw_acc for ego state."""
        ego_states = self.get_feature('ego_state')
        iters = sorted(ego_states.keys())
        ego_state = ego_states[iters[0]]
        prev_state: EgoState = _get_ego_state(self._scenario, iters[0] - 1)
        prev_prev_state: EgoState = _get_ego_state(self._scenario,
                                                   iters[0] - 2)
        if prev_state is not None:
            prev_yaw = prev_state.rear_axle.heading
            prev_time = prev_state.time_seconds
        else:
            prev_yaw = ego_state[NpEgoState.heading_index]
            prev_time = ego_state[
                NpEgoState.time_index] - self._scenario.database_interval
        if prev_prev_state is not None:
            prev_yaw_rate = _radian(
                prev_yaw - prev_prev_state.rear_axle.heading) / (
                    prev_time - prev_prev_state.time_seconds)
        else:
            prev_yaw_rate = 0.0
        for i in iters:
            ego_state = ego_states[i]
            yaw = ego_state[NpEgoState.heading_index]
            t = ego_state[NpEgoState.time_index]
            dt = t - prev_time
            yaw_rate = _radian(yaw - prev_yaw) / dt
            ego_state[NpEgoState.yaw_rate_index] = yaw_rate
            ego_state[NpEgoState.
                      yaw_acc_index] = (yaw_rate - prev_yaw_rate) / dt
            prev_time = t
            prev_yaw = yaw
            prev_yaw_rate = yaw_rate

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        scenario = PreparedScenario()
        scenario._data = data
        scenario._mission_goal = data['mission_goal']
        scenario._offset = data['offset']
        scenario._iteration_start = data['iteration_start']
        scenario._iteration_end = data['iteration_end']
        scenario._iteration_step = data['iteration_step']
        # the tire_steering_angle from NuPlanScenario is always zero (see
        # get_ego_state_for_lidarpc_token_from_db()). So we populate it here
        scenario._populate_tire_steering_angle()
        return scenario

    def _populate_tire_steering_angle(self):
        for ego_state in self._data['ego_state'].values():
            yaw_rate = ego_state[NpEgoState.yaw_rate_index]
            v = ego_state[NpEgoState.vel_lon_index]
            # the velocity can be inaccurate when it is very small
            # in this case, we set the curvature to 0
            curvature = yaw_rate / v if abs(v) > 0.1 else 0
            tire_steering_angle = np.arctan(
                curvature * ego_state[NpEgoState.wheel_base_index])
            ego_state[NpEgoState.
                      tire_steering_angle_index] = tire_steering_angle

    def to_feature_tensor(self) -> AbstractModelFeature:
        raise NotImplementedError()

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        raise NotImplementedError()

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError()

    def prepare_scenario(self, scenario: AbstractScenario,
                         iterations: range) -> None:
        """Prepare the basic information of the scenario for the given iterations.

        :param scenario: The scenario to prepare.
        :param iterations: The iterations to prepare the scenario for.
        """
        self._scenario = scenario
        self._mission_goal = scenario.get_mission_goal()
        vp = scenario.get_ego_state_at_iteration(
            0).car_footprint.vehicle_parameters
        self._vehical_parameters = [
            vp.width, vp.front_length, vp.rear_length,
            vp.cog_position_from_rear_axle, vp.wheel_base,
            vp.height if vp.height is not None else 0.0
        ]
        timestamps = []
        for i in iterations:
            ego_state = scenario.get_ego_state_at_iteration(i)
            if i == iterations.start:
                self._offset = ego_state.rear_axle.point.array
                self._data['offset'] = self._offset
            self.add_ego_state_at_iteration(ego_state, i)
            timestamps.append(scenario.get_time_point(i).time_us)

        self._data['log_name'] = scenario.log_name
        self._data['token'] = scenario.token
        self._data['iteration_start'] = iterations.start
        self._data['iteration_end'] = iterations.stop
        self._data['iteration_step'] = iterations.step
        self._data['timestamps'] = np.array(timestamps)
        self._data['database_interval'] = scenario.database_interval
        mission_goal = scenario.get_mission_goal()
        if mission_goal is not None:
            self._data['mission_goal'] = StateSE2(
                mission_goal.x - float(self._offset[0]),
                mission_goal.y - float(self._offset[1]), mission_goal.heading)
        else:
            # For example, P3CScenario does not have mission goal
            self._data['mission_goal'] = None
        self._iteration_start = iterations.start
        self._iteration_end = iterations.stop
        self._iteration_step = iterations.step

    def add_ego_state_at_iteration(self, ego_state: EgoState, iteration: int):
        """Add ego state at the given iteration to the prepared scenario.

        :param ego_state: The scenario to prepare.
        :param iteration: The iteration to prepare the scenario for.
        """
        ego_state_vec = list(ego_state)
        ego_state_vec[0] *= 1e-6  # convert to seconds
        yaw_state = [0, 0.]  # placeholder for yaw_rate and yaw_acc
        ego_state = np.array(ego_state_vec + yaw_state +
                             self._vehical_parameters)
        ego_state[NpEgoState.x_index:NpEgoState.y_index + 1] -= self._offset
        self.add_feature_at_iteration("ego_state", ego_state, iteration)

    def get_offset(self) -> np.ndarray:
        """"The origin of the prepared scenario in the original map coordinates.

        All the cooordinates in prepared scenario are relative to this origin.
        So when adding features to the prepared scenario, the coordinates be the
        original map coordinates minus this offset.
        """
        return self._offset

    def get_feature(self, feature_name: str) -> Any:
        """Get feature with the given name.

        :param feature_name: The name of the feature. None if the feature does
            not exist.
        """
        feature = self._data.get(feature_name, NONEXISTENT_FEATURE)
        if feature is NONEXISTENT_FEATURE:
            raise MissingFeatureException(
                f"Missing feature {feature_name} for "
                f"{self._data['log_name']}:{self._data['token']}")
        return feature

    def feature_exists(self, feature_name: str) -> bool:
        """Whether there is feature with the given name.

        :param feature_name: The name of the feature.
        :return True if there is feature with the given name, False otherwise.
        """
        return feature_name in self._data

    def add_feature(self, feature_name: str, feature_value: Any):
        """Add feature with the given name.

        :param feature_name: The name of the feature.
        :param feature_value: The value of the feature.
        """
        self._data[feature_name] = feature_value

    def get_feature_at_center(self, feature_name: str,
                              center: Tuple[int, int]) -> Any:
        """Get feature at the given discretized center.

        :param feature_name: The name of the feature.
        :param center: The discritized center of the feature.
        """
        if feature_name not in self._data:
            raise MissingFeatureException(
                f"Missing feature {feature_name} for "
                f"{self._data['log_name']}:{self._data['token']}")
        feature = self._data[feature_name]
        f = feature.get(center, NONEXISTENT_FEATURE)
        if f is NONEXISTENT_FEATURE:
            raise MissingFeatureAtCenterException(
                f"Missing feature {feature_name} at center {center} for "
                f"{self._data['log_name']}:{self._data['token']}")
        return f

    def feature_exists_at_center(self, feature_name: str,
                                 center: Tuple[int, int]) -> bool:
        """Whether there is feature at the given discretized center.

        :param feature_name: The name of the feature.
        :param center: The discritized center of the feature.
        :return True if there is feature at the given discretized center, False otherwise.
        """
        if feature_name not in self._data:
            return False
        feature = self._data[feature_name]
        return center in feature

    def add_feature_at_center(self, feature_name: str, feature_value,
                              center: Tuple[int, int]):
        """Add feature at the given discretized center.

        :param feature_name: The name of the feature.
        :param feature_value: The value of the feature.
        :param center: The discritized center of the feature.
        """
        if feature_name not in self._data:
            self._data[feature_name] = {}
            feature = self._data[feature_name]
        else:
            feature = self._data[feature_name]
        feature[center] = feature_value

    def get_feature_at_iteration(self, feature_name: str,
                                 iteration: int) -> Any:
        """Get feature at the given iteration.

        :param feature_name: The name of the feature.
        :param iteration: The iteration to get the feature for, it should be
            the iteration in the original scenario.
        """
        if feature_name not in self._data:
            raise MissingFeatureException(
                f"Missing feature {feature_name} for "
                f"{self._data['log_name']}:{self._data['token']}")
        feature = self._data[feature_name]
        f = feature.get(iteration, NONEXISTENT_FEATURE)
        if f is NONEXISTENT_FEATURE:
            raise MissingFeatureAtIterationException(
                f"Missing feature {feature_name} at iteration {iteration} for "
                f"{self._data['log_name']}:{self._data['token']}")
        return f

    def feature_exists_at_iteration(self, feature_name: str,
                                    iteration: int) -> bool:
        """Whether there is feature at the given iteration.

        :param feature_name: The name of the feature.
        :param iteration: The iteration to get the feature for, it should be
            the iteration in the original scenario.
        :return True if there is feature at the given iteration, False otherwise.
        """
        if feature_name not in self._data:
            return False
        feature = self._data[feature_name]
        return iteration in feature

    def add_feature_at_iteration(self, feature_name: str, feature_value,
                                 iteration: int):
        """Add feature at the given iteration.

        :param feature_name: The name of the feature.
        :param feature_value: The value of the feature.
        :param iteration: The iteration to get the feature for.
        """
        if feature_name not in self._data:
            self._data[feature_name] = {}
            feature = self._data[feature_name]
        else:
            feature = self._data[feature_name]
        feature[iteration] = feature_value

    @property
    def log_name(self) -> str:
        """The log name of the original scenario."""
        return self._data['log_name']

    @property
    def token(self) -> str:
        """The token of the original scenario."""
        return self._data['token']

    @property
    def database_interval(self) -> float:
        """The database interval of the original scenario."""
        return self._data['database_interval']

    @property
    def iterations(self) -> range:
        """All the prepared iterations."""
        return range(self._iteration_start, self._iteration_end,
                     self._iteration_step)

    def get_mission_goal(self) -> StateSE2:
        return self._data['mission_goal']

    def _iter_to_index(self, iteration: int) -> int:
        return (iteration - self._iteration_start) // self._iteration_step

    def get_ego_state_at_iteration(self, iteration: int) -> NpEgoState:
        """Get the ego state at the given iteration.

        Note that the state is in coordinates of the prepared scenario (i.e.,
        the cooridinates of the original scenario minus the offset).

        :param iteration: The iteration to get the time point for, it should be
            the iteration in the original scenario.
        :return The ego state at the given iteration.
        """
        return self.get_feature_at_iteration('ego_state',
                                             iteration).view(NpEgoState)

    def get_number_of_iterations(self) -> int:
        """The number of iterations in the original scenario."""
        return self._iteration_end

    def get_time_point(self, iteration: int) -> TimePoint:
        """Get the time point of the given iteration.

        :param iteration: The iteration to get the time point for, it should be
            the iteration in the original scenario.
        """
        i = self._iter_to_index(iteration)
        return TimePoint(time_us=int(self._data['timestamps'][i]))

    def backup_feature(self, feature_name: str):
        """Backup the feature with the given name.

        Make a copy of the feature and store it in the backup. Note that later
        backup will overwrite the previous backup.

        :param feature_name: The name of the feature.
        """
        self._backup[feature_name] = copy.deepcopy(self._data[feature_name])

    def get_backup_feature(self, feature_name: str):
        """Get the backup feature with the given name.

        :param feature_name: The name of the feature.
        """
        return self._backup[feature_name]

    def restore_feature(self, feature_name: str):
        """Restore the feature with the given name.

        The feature with name `feature_name` will be restored from its backup from
        the last call of `backup_feature`.

        :param feature_name: The name of the feature.
        """
        self._data[feature_name] = copy.deepcopy(self._backup[feature_name])


class ScenarioPreparer(ABC):
    """Base class for scenario preparer.

    A scenario preparer is used to prepare PreparedScenario for future use.
    """

    @abstractmethod
    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        """Prepare the scenario for the given iterations.

        :param scenario: The scenario to prepare.
        :param prepared_scenario: The prepared scenario to store the prepared
            features.
        :param iterations: The iterations to prepare the scenario for.
        """
        pass


class PreparedScenarioFeatureBuilder(ScenarioPreparer):
    """Feature builder for horizon features."""

    @abstractmethod
    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        """Prepare the scenario for the given iterations.

        :param scenario: The scenario to prepare.
        :param prepared_scenario: The prepared scenario to store the prepared
            features.
        :param iterations: The iterations to prepare the scenario for.
        """
        pass

    @abstractmethod
    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> AbstractModelFeature:
        """Get features from the prepared scenario.

        :param scenario: The prepared scenario.
        :param iteration: The iteration to get the features for.
        :param ego_state: The ego state at the given iteration.
        """
        pass
