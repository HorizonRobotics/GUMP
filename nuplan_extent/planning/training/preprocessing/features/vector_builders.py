from abc import ABC, abstractmethod
import cv2
import math
import numpy.typing as npt
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.coords import CoordinateSequence
from typing import Dict, List, NamedTuple, Optional, Tuple
import alf
from copy import deepcopy
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex, EgoInternalIndex)
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, PolygonMapObject, PolylineMapObject
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import get_route_polygon_from_roadblock_ids
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan_extent.planning.scenario_builder.prepared_scenario import (
    NpEgoState, NpAgentState, PreparedScenario, PreparedScenarioFeatureBuilder)
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    generate_virtual_center,
    _get_proximal_map_objects,
)
from nuplan_extent.planning.training.preprocessing.utils.route_utils import (
    route_roadblock_correction, )
from nuplan_extent.common.geometry.oriented_box import box_to_corners

from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from typing import Set
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import prune_route_by_connectivity

class ShapeList(NamedTuple):
    """Represents a list of shapes.

    All the coordinates are put into one single array for efficent storage
    and computation. The sizes array is used to index into the coordinates
    """

    # concatenated  coordinates of the points of all the shapes, shape [N, 2]
    coords: npt.NDArray[np.float32]

    # number of points for each shape, shape [M], sizes.sum() == coords.shape[0]
    sizes: npt.NDArray[np.int32]

    @property
    def num(self):
        """The number of shapes in this list."""
        return len(self.sizes)


class PreparedMapObject(NamedTuple):
    coords: npt.NDArray[np.float32]
    speed_limit: Optional[float]


# This matrix transforms the ego coordinates to raster coordinates.
# The x-axis of the raster coordinate system corresponds to the y-axis of
# the ego coordinate (i.e. the left direction of the ego vehicle). The y-axis of
# the raster coordinate corresponds to the negative x-axis of the ego coordinate
# (i.e the backward direction of the ego vehicle).
_map_align_transform = np.diag([1., -1., 1.]) @ R.from_euler(
    'z', 90, degrees=True).as_matrix()


def get_global_to_local_transform(pose: StateSE2) -> npt.NDArray[np.float32]:
    """The the transformation matrix from global to local coordinates.

    :param pose: the pose of the local coordinate system in the global coordinate system.
    :return: inverse of the 3x3 2D transformation matrix
    """
    c = math.cos(pose.heading)
    s = math.sin(pose.heading)
    x = pose.x
    y = pose.y
    return np.array([
        [c, s, -c * x - s * y],
        [-s, c, s * x - c * y],
        [0, 0, 1],
    ])


def get_local_to_global_transform(pose: StateSE2) -> npt.NDArray[np.float32]:
    """The the transformation matrix from global to local coordinates.

    :param pose: the pose of the local coordinate system in the global coordinate system.
    :return: inverse of the 3x3 2D transformation matrix
    """
    c = math.cos(pose.heading)
    s = math.sin(pose.heading)
    x = pose.x
    y = pose.y
    return np.array([
        [c, -s, x],
        [s, c, y],
        [0, 0, 1],
    ])


def transform_to_pixel_coords(untransformed_coords: npt.NDArray[np.float32],
                              center: StateSE2, radius: float, image_size: int,
                              bit_shift: int):
    """Transform global coordinates to pixel coordinates.

    Note that the resulted coordinates are the actual pixel locations multiplied
    by 2**bit_shift. bit_shift is for cv2.fillPoly/cv2.polylines to work with
    float coordinates.

    :param untransformed_coords: global coordinates
    :param center: the pose of the local coordinate system in the global
        coordinate system.
    :param radius: radius of the raster in meters
    :param image_size: size of the raster image in pixels
    :param bit_shift: bit shift for converting float to int
    """

    if len(untransformed_coords) == 0:
        return []
    resolution = (2 * radius) / image_size
    scale = 2**bit_shift / resolution
    global_transform = get_global_to_local_transform(center)
    # By default the map is right-oriented, this makes it top-oriented.
    transform = _map_align_transform @ global_transform
    mat = (transform[:2, :2].T * scale).astype(np.float32)
    vec = transform[:2, 2] + radius
    # Previously, the raster is generated with a different _map_align_transform
    # without multiplying diag(1,-1, 1). And the result is flipped vertically.
    # (See raster_utils.get_roadmap_raster for example).
    # The current way achieves the flip directly by changing _map_align_transform.
    # However, this results in a one pixel vertical shift. The following line
    # is to compensate for this shift so that the generated raster is compatible
    # with the previously trainied model.
    vec[1] -= resolution
    vec = (vec * scale).astype(np.float32)
    object_coords = (untransformed_coords @ mat + vec).astype(np.int64)
    return object_coords


def draw_polygon_image(polygons: ShapeList, colors: List[float],
                       image_size: int, center: StateSE2,
                       radius: float) -> npt.NDArray[np.float32]:
    """Draw polygons on the raster.

    :param polygons: polygons to be drawn. The coordinates are in the global
        frame.
    :param colors: colors of the polygons
    :param image_size: size of the raster image in pixels
    :param center: the pose of the local coordinate system in the global
        coordinate system.
    :param radius: radius of the raster in meters
    """
    raster: npt.NDArray[np.float32] = np.zeros((image_size, image_size),
                                               dtype=np.float32)
    if polygons.coords.size == 0:
        return raster

    bit_shift = 10
    coords = transform_to_pixel_coords(polygons.coords, center, radius,
                                       image_size, bit_shift)
    start = 0
    for color, size in zip(colors, polygons.sizes):
        cv2.fillPoly(
            raster,
            coords[start:start + size][None],
            color=color,
            shift=bit_shift,
            lineType=cv2.LINE_AA)
        start += size
    return raster

class VectorBuilderBase(PreparedScenarioFeatureBuilder):
    """The base class for raster builders.

    :param image_size: size of the raster image in pixels
    :param radius: radius of the raster in meters
    :param ego_longitudinal_offset: the center of the raster is `longitudinal_offset * radius`
        meters in front of the rear axle of ego vehicle. 0 means the center is
        at the rear axle.
    """

    def __init__(self, radius: float,
                 longitudinal_offset: float):
        super().__init__()
        self._radius = radius
        self._longitudinal_offset = longitudinal_offset
        self._cache_enabled = True
        # see set_cache_parameter for explanations about the following two parameters
        self._cache_grid_step = self._radius / 2
        self._cache_radius = self._radius * 2.5

    def set_cache_parameter(self, cache_grid_step: float,
                            cache_radius: float) -> None:
        """Setup cache parameters.

        The map features are cached based on the discretized center of the raster.
        If two raster has same discretized center, they will share the same cache.
        The discretized center is calculated as `(round(x / cache_grid_step), round(y / cache_grid_step))`.
        In order for the same cached feature to cover different raster at different
        centers, the cache radius should be larger than the radius of the raster.
        The default cache_grid_step is radius / 2, and the default cache_radius is
        radius * 2.5.

        :param cache_grid_step: grid step for the cache
        :param cache_radius: radius for the cache
        """
        self._cache_grid_step = cache_grid_step
        self._cache_radius = cache_radius

    def calc_raster_center(self, ego_state: NpEgoState) -> StateSE2:
        """Calculate the center of the raster.

        The center of the raster is `longitudinal_offset * radius` meters in front
        of the rear axle of ego_state.

        :param ego_state: ego state
        """
        return generate_virtual_center(
            StateSE2(*ego_state.rear_axle_pose.tolist()), self._radius,
            self._longitudinal_offset)

    def calc_cache_key(self, center: Point2D) -> Tuple[int, int]:
        """Calculate cache key.

        The cache key is the discretized center of the raster and it is used
        for retrieving the map objects from the cache.

        :param center: center of the raster in the coordinates of prepared scenario
        :return: cache key for retrieving the map objects in the cache.
        """
        step = self._cache_grid_step
        grid_x = int(math.floor(center.x / step + 0.5))
        grid_y = int(math.floor(center.y / step + 0.5))
        return grid_x, grid_y

    def calc_cache_key_and_center(self, center: Point2D, offset: npt.NDArray[2]
                                  ) -> Tuple[Tuple[int, int], Point2D]:
        """Calculate cache key and map center for getting map objects from the map.

        :param center: center of the raster in the coordinates of the prepared scenario
        :param offset: offset for shifting the coordinates for the prepared scenario
        :return:
            - cache key for storing/retrieving the map objects in the cache.
            - the center for retrieving map objects (in the original map coordinate)
                The caller should use the returned center and self._cache_radius to
                get map objects from the map.
        """
        step = self._cache_grid_step
        grid_x = int(math.floor(center.x / step + 0.5))
        grid_y = int(math.floor(center.y / step + 0.5))
        # Avoid numpy types since Point2D members are float
        x0 = grid_x * step + float(offset[0])
        y0 = grid_y * step + float(offset[1])
        return (grid_x, grid_y), Point2D(x0, y0)

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        pass


@alf.configurable(whitelist=["relative_overlay"])
class PastCurrentAgentsVectorBuilder(VectorBuilderBase):
    """Raster representing the past and current agents.

    Cached Features:
        tracked_object_boxes: Dict[int, np.ndarray[n, 5]): boxes of the tracked
            object for each iteration. Key is iteration. Each box is reparesented
            by 5 numbers: center_x, center_y, heading, half_length, half_width.
        ego_state: Dict[int, NpEgoState]: ego states for iterations before 0,
            up to `past_time_horizon`.
    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the raster
    :param past_time_horizon: [s] time horizon of past agents
    :param past_num_steps: number of past steps to sample
    :param relative_overlay: If False, each agent is represented a series of boxes
        corresponding to their past and current pose. If True, each agent is represented
        as a series of boxes, where each box represents the pose of the agent
        relative to the pose of the ego vehicle at the corresponding iteration.
    """

    def __init__(self,
                 radius: float,
                 longitudinal_offset: float,
                 past_time_horizon: float,
                 past_num_steps: int,
                 future_time_horizon: float,
                 future_num_steps: int,
                 agent_features: List[str] = ["VEHICLE", "PEDESTRIAN", "CYCLIST"],
                 num_max_agents: List[int] = [256,128,32],
                 relative_overlay: bool = False):
        super().__init__(radius, longitudinal_offset)
        self._past_time_horizon = past_time_horizon
        self._past_num_steps = past_num_steps
        self._future_time_horizon = future_time_horizon
        self._future_num_steps = future_num_steps
        self._agent_features = agent_features
        self._num_max_agents = num_max_agents
        self._relative_overlay = relative_overlay

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        offset = prepared_scenario.get_offset()
        interval = scenario.database_interval

        if self._past_time_horizon > 0:
            history_window = int(self._past_time_horizon / interval)
            assert history_window % self._past_num_steps == 0, (
                f"history window {history_window} must be divisible by "
                f"past_num_steps {self._past_num_steps}")
            history_step = history_window // self._past_num_steps
        else:
            history_window = 0
            history_step = 1

        if self._future_time_horizon > 0:
            future_window = int(self._future_time_horizon / interval)
            assert future_window % self._future_num_steps == 0, (
                f"future window {future_window} must be divisible by "
                f"future_num_steps {self._future_num_steps}")
            future_step = future_window // self._future_num_steps
            assert history_step == future_step, (
                f"history_step {history_step} must be equal to future_step ") 
        else:
            future_window = 0
            future_step = 1

        # 1. get all the history/current/future iteration for iterations
        all_iterations = set()
        for iteration in iterations:
            all_iterations.update(
                range(iteration - history_window, iteration + 1 + future_window, history_step))
        all_iterations = sorted(all_iterations)

        # 2. find the earliest unprepared iteration
        earliest_unprepared_iteration = None
        for iteration in all_iterations:
            if not prepared_scenario.feature_exists_at_iteration(
                    "vectorized_tracked_object_boxes", iteration):
                earliest_unprepared_iteration = iteration
                break

        # 3. Get past tracked objects
        if earliest_unprepared_iteration is not None and earliest_unprepared_iteration < 0:
            num_samples = -earliest_unprepared_iteration
            if num_samples == 1 and isinstance(scenario, NuPlanScenario):
                # 0.5 * iterval is due to an artifact of NuPlanScenario.get_past_tracked_object()
                # that when num_samples=1, it actually gets a sample at one
                # iteration before the given iteration. This is caused by
                # scenario_utils.sample_indices_with_time_horizon()
                # See nuplan issue https://github.com/motional/nuplan-devkit/issues/348
                # TODO: remove this hack when the issue is fixed.
                time_horizon = 0.5 * interval
            else:
                time_horizon = num_samples * interval
            past_detections = list(
                scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=time_horizon,
                    num_samples=num_samples))
            assert len(past_detections) == num_samples
            past_ego_states = list(
                scenario.get_ego_past_trajectory(
                    iteration=0,
                    time_horizon=time_horizon,
                    num_samples=num_samples))
        # 4. prepare polygons for all iterations
        track_token_id_mapping = {feature_name:{} for feature_name in self._agent_features}
        for iteration in all_iterations:
            if prepared_scenario.feature_exists_at_iteration(
                    "vectorized_tracked_object_boxes", iteration):
                continue
            if iteration < 0:
                detections = past_detections[iteration]
                past_ego_state = past_ego_states[iteration]
                prepared_scenario.add_ego_state_at_iteration(
                    past_ego_state, iteration)
            else:
                detections = scenario.get_tracked_objects_at_iteration(
                    iteration)
                future_ego_state = scenario.get_ego_state_at_iteration(
                    iteration)
                prepared_scenario.add_ego_state_at_iteration(
                    future_ego_state, iteration)
            vector_data, track_token_id_mapping = self._get_vector_data(
                detections, 
                offset, 
                track_token_id_mapping
            )
            prepared_scenario.add_feature_at_iteration('vectorized_tracked_object_boxes',
                                                       vector_data, iteration)

    def _get_vector_data(
        self, 
        detections: DetectionsTracks,
        offset,
        track_token_id_mapping
    ) -> npt.NDArray[np.float32]:
        x0 = float(offset[0])
        y0 = float(offset[1])
        vector_data = {}
        for i,feature_name in enumerate(self._agent_features):
            object_type = TrackedObjectType[feature_name]
            agents = detections.tracked_objects.get_tracked_objects_of_type(object_type)
            output = np.zeros((self._num_max_agents[i], AgentInternalIndex.dim()), dtype=np.float32) * np.nan
            max_agent_id = len(track_token_id_mapping[feature_name])

            for idx, agent in enumerate(agents):
                # print(idx, self._num_max_agents)
                if idx >= self._num_max_agents[i]:
                    continue
                if agent.track_token not in track_token_id_mapping[feature_name]:
                    track_token_id_mapping[feature_name][agent.track_token] = max_agent_id
                    max_agent_id += 1
                track_token_int = track_token_id_mapping[feature_name][agent.track_token]

                output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
                output[idx, AgentInternalIndex.vx()] = agent.velocity.x
                output[idx, AgentInternalIndex.vy()] = agent.velocity.y
                output[idx, AgentInternalIndex.heading()] = agent.center.heading
                output[idx, AgentInternalIndex.width()] = agent.box.width
                output[idx, AgentInternalIndex.length()] = agent.box.length
                output[idx, AgentInternalIndex.x()] = agent.center.x - x0
                output[idx, AgentInternalIndex.y()] = agent.center.y - y0
            vector_data[feature_name] = output
        return vector_data, track_token_id_mapping

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:

        step_interval = int(self._past_time_horizon /
                            scenario.database_interval / self._past_num_steps)
        center = self.calc_raster_center(ego_state)
        history_vector_data = {}
        for i in range(self._past_num_steps + 1 + self._future_num_steps):
            iter = iteration - (self._past_num_steps - i) * step_interval
            vector_data = scenario.get_feature_at_iteration('vectorized_tracked_object_boxes',
                                                      iter)
            vector_data = deepcopy(vector_data)
            for class_name, class_vector_data in vector_data.items():
                class_vector_data = class_vector_data.view(NpAgentState)
                valid_num = np.sum(~np.isnan(class_vector_data).any(axis=-1))
                if valid_num > 0:
                    class_vector_data = self._transform_global_to_local(class_vector_data, center)
                vector_data[class_name] = class_vector_data
                history_vector_data[class_name] = history_vector_data.get(class_name, []) + [class_vector_data]
        for class_name, class_vector_data in history_vector_data.items():
            history_vector_data[class_name] = np.stack(class_vector_data, axis=0)
        return history_vector_data

    def _transform_global_to_local(self, class_vector_data, center):
        trans = get_global_to_local_transform(center)
        class_vector_data.transform(trans, center.heading)
        return class_vector_data


@alf.configurable(whitelist=["relative_overlay"])
class PastCurrentEgoVectorBuilder(VectorBuilderBase):
    """
    """

    def __init__(self,
                 radius: float,
                 longitudinal_offset: float,
                 past_time_horizon: float,
                 past_num_steps: int,
                 future_time_horizon: float,
                 future_num_steps: int,
                 relative_overlay: bool = False):
        super().__init__(radius, longitudinal_offset)
        self._past_time_horizon = past_time_horizon
        self._past_num_steps = past_num_steps
        self._future_time_horizon = future_time_horizon
        self._future_num_steps = future_num_steps
        self._relative_overlay = relative_overlay

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        pass

    def _get_ego_array(
        self, 
        ego_state: NpEgoState,
    ) -> npt.NDArray[np.float32]:
        ego_state = ego_state.to_nuplan_ego_state()
        output = np.zeros((EgoInternalIndex.dim()), dtype=np.float32) * np.nan
        output[EgoInternalIndex.x()] = ego_state.rear_axle.x
        output[EgoInternalIndex.y()] = ego_state.rear_axle.y
        output[EgoInternalIndex.heading()] = ego_state.rear_axle.heading
        output[EgoInternalIndex.vx()] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        output[EgoInternalIndex.vy()] = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        output[EgoInternalIndex.ax()] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        output[EgoInternalIndex.ay()] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
        return output

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            current_ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        step_interval = int(self._past_time_horizon /
                            scenario.database_interval / self._past_num_steps)
        current_center = self.calc_raster_center(current_ego_state)
        ego_array_history = []
        for i in range(self._past_num_steps + 1 + self._future_num_steps):
            iter = iteration - (self._past_num_steps - i) * step_interval
            ego_state = scenario.get_ego_state_at_iteration(iter)
            ego_state = deepcopy(ego_state)
            ego_state = self._transform_global_to_local(ego_state, current_center)
            ego_array = self._get_ego_array(ego_state)
            ego_array_history.append(ego_array)
        return np.stack(ego_array_history, axis=0)
    
    def _transform_global_to_local(self, ego_array, center):
        trans = get_global_to_local_transform(center)
        ego_array.transform(trans, center.heading)
        return ego_array