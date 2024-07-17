from __future__ import annotations

from typing import Callable, List, Tuple, Optional, Dict, Generator, Union
import copy
import math
from array import array
from collections import defaultdict, deque

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

import shapely.geometry as geom
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.maps.nuplan_map.polygon_map_object import NuPlanPolygonMapObject
from nuplan.common.maps.nuplan_map.polyline_map_object import NuPlanPolylineMapObject
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.common.maps.nuplan_map.roadblock import NuPlanRoadBlock
from nuplan.planning.simulation.observation.observation_type import \
    DetectionsTracks
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import \
    get_route_polygon_from_roadblock_ids
from nuplan.planning.training.preprocessing.features.raster_utils import (
    _cartesian_to_projective_coords, _draw_linestring_image,
    _draw_polygon_image, _linestring_to_coords, _polygon_to_coords)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.common.maps.abstract_map_objects import PolylineMapObject

BASELINE_TL_COLOR = {
    TrafficLightStatusType.RED: (1, 0, 0),
    TrafficLightStatusType.YELLOW: (1, 1, 0),
    TrafficLightStatusType.GREEN: (0, 1, 0),
    # Also the deafult color for baseline path
    TrafficLightStatusType.UNKNOWN: (0, 0, 1),
}


def generate_virtual_center(center: StateSE2, radius: float,
                            longitudinal_offset: float) -> StateSE2:
    """Generate virtual ego center

    Args:
        ego_state (StateSE2): input SE2 center [center or real_axle].
        radius (float): [m] the radius of the square raster map.
        longitudinal_offset (float): longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center

    Returns:
        StateSE2: SE2 state of virtual ego center
    """
    if longitudinal_offset == 0:
        return center
    virtual_center = copy.deepcopy(center)
    center_offset = [
        2 * radius * longitudinal_offset * np.cos(virtual_center.heading),
        2 * radius * longitudinal_offset * np.sin(virtual_center.heading)
    ]
    virtual_center.x += center_offset[0]
    virtual_center.y += center_offset[1]
    return virtual_center


def calc_raster_center(ego_state: EgoState, radius: float,
                       longitudinal_offset: float) -> StateSE2:
    """Calculate the center of the raster.

    The center of the raster is `longitudinal_offset * radius` meters in front
    of the rear axle of ego_state.

    :param ego_state: ego state
    """
    return generate_virtual_center(ego_state.rear_axle, radius,
                                   longitudinal_offset)


def get_route_raster(
        ego_state: EgoState,
        route_roadblock_ids: List[str],
        map_api: AbstractMap,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        feature_color: Optional[float] = 1.0,
        with_coords: bool = False,
        longitudinal_offset: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Construct the route layer of the raster by converting roadblock ids to raster map.
    :param ego_state: SE2 state of ego.
    :param route_roadblock_ids: roadblock ids along with the route
    :param map_api: map api.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param feature_color: feature value fill in road block polygon
    :return route_raster: the constructed map raster layer.
    """
    # Assume the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (
        y_range[1] -
        y_range[0]), f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'

    radius = (x_range[1] - x_range[0]) / 2
    route_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)
    global_transform = np.linalg.inv(virtual_center.as_matrix())

    # By default the map is right-oriented, this makes it top-oriented.
    map_align_transform = R.from_euler(
        'z', 90, degrees=True).as_matrix().astype(np.float32)
    transform = map_align_transform @ global_transform
    block_polylines = get_route_polygon_from_roadblock_ids(
        map_api, virtual_center.point, radius,
        route_roadblock_ids).to_vector()
    coords = [(transform @ _cartesian_to_projective_coords(
        np.array(polylines)).T).T[:, :2] for polylines in block_polylines]
    route_raster = _draw_polygon_image(route_raster, coords, radius,
                                       resolution, feature_color)

    # Flip the agents_raster along the horizontal axis.
    route_raster = np.flip(route_raster, axis=0)
    route_raster = np.ascontiguousarray(route_raster, dtype=np.float32)
    if with_coords:
        return route_raster, coords
    return route_raster


def get_past_current_agents_raster(
        agents_raster: npt.NDArray[np.float32],
        ego_state: EgoState,
        detections: DetectionsTracks,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        polygon_bit_shift: int = 9,
        color_value: Optional[float] = 1.0,
        longitudinal_offset: Optional[float] = 0.0,
):
    """
    Construct the history agents layer of the raster by transforming all detected boxes around the agent
    and creating polygons of them in a raster grid.
    :param ego_state: current state of ego.
    :param detections: list of 3D bounding box of detected agents.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed agents raster layer.
    """

    width, height = raster_shape
    radius = (x_range[1] - x_range[0]) / 2
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)

    ego_to_global = virtual_center.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)

    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()

    for tracked_object in detections.tracked_objects:
        # Transform the box relative to agent.
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix(
        )
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        # Filter out boxes outside the raster.
        valid_x = -radius < raster_object_pose.y < radius
        valid_y = -radius < raster_object_pose.x < radius
        if not (valid_x and valid_y):
            continue

        # Get the 2D coordinate of the detected agents.
        raster_oriented_box = OrientedBox(
            raster_object_pose, tracked_object.box.length,
            tracked_object.box.width, tracked_object.box.height)
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray(
            [corner.x for corner in box_bottom_corners])  # type: ignore
        y_corners = np.asarray(
            [corner.y for corner in box_bottom_corners])  # type: ignore

        # Discretize
        y_corners = (y_corners + radius) / (2 * radius) * height
        x_corners = (x_corners + radius) / (2 * radius) * width

        box_2d_coords = np.stack([x_corners, y_corners],
                                 axis=1)  # type: ignore
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

        # Draw the box as a filled polygon on the raster layer.
        box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(
            agents_raster,
            box_2d_coords,
            color=color_value,
            shift=polygon_bit_shift,
            lineType=cv2.LINE_AA)

    return agents_raster


def get_static_agents_raster(
        agents_raster: npt.NDArray[np.float32],
        ego_state: EgoState,
        detections: DetectionsTracks,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        polygon_bit_shift: int = 9,
        color_value: Optional[float] = 1.0,
        longitudinal_offset: Optional[float] = 0.0,
):
    """
    Construct the history agents layer of the raster by transforming all detected boxes around the agent
    and creating polygons of them in a raster grid.
    :param ego_state: SE2 state of ego.
    :param detections: list of 3D bounding box of detected agents.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed agents raster layer.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    # ego_to_global = ego_state.rear_axle.as_matrix()
    radius = (x_range[1] - x_range[0]) / 2
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)

    ego_to_global = virtual_center.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)

    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()

    for tracked_object in detections.tracked_objects:
        if tracked_object.tracked_object_type not in [TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.BARRIER, TrackedObjectType.CZONE_SIGN, TrackedObjectType.GENERIC_OBJECT]:
            continue
        # print(tracked_object.tracked_object_type)
        # Transform the box relative to agent.
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix(
        )
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        # Filter out boxes outside the raster.
        valid_x = x_range[0] < raster_object_pose.x < x_range[1]
        valid_y = y_range[0] < raster_object_pose.y < y_range[1]
        if not (valid_x and valid_y):
            continue

        # Get the 2D coordinate of the detected agents.
        raster_oriented_box = OrientedBox(
            raster_object_pose, tracked_object.box.length,
            tracked_object.box.width, tracked_object.box.height)
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray(
            [corner.x for corner in box_bottom_corners])  # type: ignore
        y_corners = np.asarray(
            [corner.y for corner in box_bottom_corners])  # type: ignore

        # Discretize
        y_corners = (y_corners - ymin) / (ymax - ymin) * height  # type: ignore
        x_corners = (x_corners - xmin) / (xmax - xmin) * width  # type: ignore

        box_2d_coords = np.stack([x_corners, y_corners],
                                 axis=1)  # type: ignore
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

        # Draw the box as a filled polygon on the raster layer.
        box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(
            agents_raster,
            box_2d_coords,
            color=color_value,
            shift=polygon_bit_shift,
            lineType=cv2.LINE_AA)

    return agents_raster


def get_speed_raster(
        past_trajectory: npt.NDArray[np.float32],
        raster_shape: Tuple[float, float],
        feature_time_interval: float,
        max_speed_normalizer: float,
) -> npt.NDArray[np.float32]:
    """
    Construct speed raster with normlized speed filling the full grid.
    :param past_trajectory: past trajectory of ego vehicle.
    :param raster_shape: shape of the target raster.
    :param max_speed_normalizer: use max speed to normalize current speed
    :return: constructed speed raster layer.
    """
    ego_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)
    ego_raster[:, :] = np.hypot(past_trajectory[-1, 0], past_trajectory[-1, 1])

    ego_raster = np.ascontiguousarray(ego_raster, dtype=np.float32)
    ego_raster /= feature_time_interval
    ego_raster /= max_speed_normalizer
    return ego_raster


def get_target_raster(
        future_trajectory: npt.NDArray[np.float32],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[float, float],
) -> npt.NDArray[np.float32]:
    """
    Construct future target of ego vehicle on raster by drawing a pixel in the grid.
    :param future_trajectory: future trajectory of ego vehicle.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :return: constructed speed raster layer.
    """
    ego_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    # Discretize
    target_y = (future_trajectory[-1, 0] - ymin) / \
        (ymax - ymin) * height  # type: ignore
    target_x = (future_trajectory[-1, 1] - xmin) / \
        (xmax - xmin) * width  # type: ignore

    target_x = int(np.clip(target_x, 0, height - 1))
    target_y = int(np.clip(target_y, 0, width - 1))

    cv2.circle(
        ego_raster, (target_x, target_y), radius=0, color=1.0, thickness=-1)
    ego_raster = np.flip(ego_raster, axis=0)
    ego_raster = np.flip(ego_raster, axis=1)
    ego_raster = np.ascontiguousarray(ego_raster, dtype=np.float32)
    return ego_raster


def get_augmented_ego_raster(
        raster_shape: Tuple[int, int],
        ego_longitudinal_offset: float,
        ego_width_pixels: float,
        ego_front_length_pixels: float,
        ego_rear_length_pixels: float,
        offset_yaw: float,
        offset_xy: Tuple[float, float],
        polygon_bit_shift: int = 9,
) -> npt.NDArray[np.float32]:
    """
    Construct the ego layer of the raster by drawing a polygon of the ego's extent in the middle of the grid.
    :param raster_shape: shape of the target raster.
    :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
    :param ego_width_pixels: width of the ego vehicle in pixels.
    :param ego_front_length_pixels: distance between the rear axle and the front bumper in pixels.
    :param ego_rear_length_pixels: distance between the rear axle and the rear bumper in pixels.
    :param yaw: offset heading of ego vehicle on the raster map.
    :param offset: [m] offset distance of ego vehicle on the raster map.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed ego raster layer.
    """
    ego_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    # Construct a rectangle representing the ego vehicle in the center of the
    # raster.
    map_x_center = int(raster_shape[1] * 0.5)
    map_y_center = int(raster_shape[0] * (0.5 + ego_longitudinal_offset))
    ego_top_left = [-ego_width_pixels // 2, -ego_front_length_pixels, 1]
    ego_bottom_left = [ego_width_pixels // 2, -ego_front_length_pixels, 1]
    ego_bottom_right = [ego_width_pixels // 2, +ego_rear_length_pixels, 1]
    ego_top_right = [-ego_width_pixels // 2, +ego_rear_length_pixels, 1]
    corners = np.array(
        [ego_top_left, ego_bottom_left, ego_bottom_right, ego_top_right])
    # TR matrix
    tr_matrix = np.array([[
        np.cos(offset_yaw),
        np.sin(offset_yaw), offset_xy[0] + map_x_center
    ], [-np.sin(offset_yaw),
        np.cos(offset_yaw), offset_xy[1] + map_y_center], [0, 0, 1]])
    corners = np.dot(tr_matrix, corners.T).T

    box_2d_coords = np.expand_dims(corners[:, :2], axis=0)

    # Draw the box as a filled polygon on the raster layer.
    box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
    cv2.fillPoly(
        ego_raster,
        box_2d_coords,
        color=1.0,
        shift=polygon_bit_shift,
        lineType=cv2.LINE_AA)
    ego_raster = np.asarray(ego_raster)
    ego_raster = np.ascontiguousarray(ego_raster, dtype=np.float32)
    return ego_raster


def get_fut_agents_raster(
        ego_state: EgoState,
        tracked_objects: List[DetectionsTracks],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        polygon_bit_shift: int = 9,
):
    """
    Construct the future agents layer of the raster by transforming all detected boxes around the agent
    and creating polygons of them in a raster grid.

    :param ego_state: SE2 state of ego.
    :param tracked_objects: list of detected tracks.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :return: constructed agents raster layer.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    ego_to_global = ego_state.rear_axle.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)

    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()
    agents_raster = np.zeros(raster_shape, dtype=np.float32)

    for tracked_object in tracked_objects:
        # Transform the box relative to agent.
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix(
        )
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        # Filter out boxes outside the raster.
        valid_x = x_range[0] < raster_object_pose.x < x_range[1]
        valid_y = y_range[0] < raster_object_pose.y < y_range[1]
        if not (valid_x and valid_y):
            continue

        # Get the 2D coordinate of the detected agents.
        raster_oriented_box = OrientedBox(
            raster_object_pose, tracked_object.box.length,
            tracked_object.box.width, tracked_object.box.height)
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray(
            [corner.x for corner in box_bottom_corners])  # type: ignore
        y_corners = np.asarray(
            [corner.y for corner in box_bottom_corners])  # type: ignore

        # Discretize
        y_corners = (y_corners - ymin) / (ymax - ymin) * height  # type: ignore
        x_corners = (x_corners - xmin) / (xmax - xmin) * width  # type: ignore

        box_2d_coords = np.stack([x_corners, y_corners],
                                 axis=1)  # type: ignore
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

        # Draw the box as a filled polygon on the raster layer.
        value = tracked_object.tracked_object_type.value + 1
        box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(
            agents_raster,
            box_2d_coords,
            color=value,
            shift=polygon_bit_shift,
            lineType=cv2.LINE_AA)

    # Flip the agents_raster along the horizontal axis.
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)
    agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

    return agents_raster


def filter_tracked_objects(tracked_objects_lst: List[TrackedObjects],
                           reverse: bool) -> List[TrackedObjects]:
    selected_tracked_agents_lst = []
    if reverse:
        anchor_tracked_objects = tracked_objects_lst[-1]
        candidate_tracked_objects_lst = tracked_objects_lst[:-1]
    else:
        anchor_tracked_objects = tracked_objects_lst[0]
        candidate_tracked_objects_lst = tracked_objects_lst[1:]

    valid_track_token_set = set()
    for tracked_agent in anchor_tracked_objects.get_agents():
        valid_track_token_set.add(tracked_agent.track_token)

    for i, tracked_objects in enumerate(candidate_tracked_objects_lst):
        tracked_agents_inside_anchor = []
        for tracked_object in tracked_objects:
            if tracked_object.track_token in valid_track_token_set:
                tracked_agents_inside_anchor.append(tracked_object)
        selected_tracked_agents_lst.append(tracked_agents_inside_anchor)

    return selected_tracked_agents_lst


def get_drivable_area_raster(
        focus_agent: Union[AgentState, EgoState],
        map_api: AbstractMap,
        map_features: Dict[str, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        longitudinal_offset: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Constructs a raster map of the drivable area around the focus agent.

    Args:
        focus_agent (AgentState, EgoState): The state of the agent.
        map_api: An instance of a map API to query map data.
        map_features: A dictionary of map features to be drawn and their color for encoding.
        x_range: A tuple of the minimum and maximum range from the edges of the grid in the x direction, in meters.
        y_range: A tuple of the minimum and maximum range from the edges of the grid in the y direction, in meters.
        raster_shape: A tuple of the shape of the target raster (assumed to be square).
        resolution: The pixel size in meters.
        longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center

    Returns:
        A NumPy array of the constructed drivable area raster layer.

    Raises:
        AssertionError: If the x and y ranges do not form a square shape.

    """
    # Ensure the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (
        y_range[1] - y_range[0]
    ), f"Raster shape is assumed to be square but got width: {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}"

    # Compute the radius of the raster.
    radius = (x_range[1] - x_range[0]) / 2

    # Initialize the raster map to all zeros.
    drivable_area_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    # Get the coordinates of the drivable area around the focus agent.
    if isinstance(map_api, NuPlanMap):
        # Current NuPlanMap does not support nuplan native function _get_layer_coords.
        coords, _, _ = _get_extended_layer_coords(
            focus_agent, map_api, SemanticMapLayer['DRIVABLE_AREA'], 'polygon',
            radius, longitudinal_offset)
    else:
        # If map_api does not support _get_layer_coords function, just add additional if_else conditions.
        coords, _, = _get_layer_coords(focus_agent, map_api,
                                       SemanticMapLayer['DRIVABLE_AREA'],
                                       'polygon', radius, longitudinal_offset)

    # Encode the drivable area polygon in the raster map.
    feature_color = 1.0
    drivable_area_raster = _draw_polygon_image(
        drivable_area_raster, coords, radius, resolution, feature_color)

    # Flip the raster map along the horizontal axis to match image conventions.
    drivable_area_raster = np.flip(drivable_area_raster, axis=0)

    # Ensure the raster map is contiguous and has the correct data type.
    drivable_area_raster = np.ascontiguousarray(
        drivable_area_raster, dtype=np.float32)

    return drivable_area_raster


def _polyline_to_coords(
        geometry: List[PolylineMapObject]) -> List[Tuple[array[float]]]:
    """Get 2d coordinates of the vertices of a polyline.

    :param geometry: the polyline.
    :return: 2d coordinates of the vertices of the polygon.
    """
    return [element.linestring.coords.xy for element in geometry]


def get_interior_edges(
        map_api: AbstractMap,
        route_id: str,
) -> List[NuPlanLane]:
    """
    Function returns lanes(no lane connectors) when getting an input id of road block or road block connector
    :param map_api: An instance of a map API to query map data.
    :param route_id: A route roadblock id.
    :return: List of filtered roadblock interior_edges
    """
    roadblock = map_api._get_roadblock(route_id)
    if roadblock:
        return roadblock.interior_edges
    else:
        roadblock_connector = map_api._get_roadblock_connector(route_id)
        if roadblock_connector:
            return roadblock_connector.interior_edges
        else:
            raise ValueError("Should be road_block or road_block connector")


def get_map_object_raster(
        focus_agent: EgoState,
        map_api: AbstractMap,
        map_feature_name: str,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        route_roadblock_ids: List[str],
        draw_ego_route_separately: bool = False,
) -> npt.NDArray[np.float32]:
    """
    Constructs a raster map of the map_objects area around the focus agent.

    Args:
        focus_agent: The state of the agent representing the ego vehicle.
        map_api: An instance of a map API to query map data.
        map_feature_names: List of str
        x_range: A tuple of the minimum and maximum range from the edges of the grid in the x direction, in meters.
        y_range: A tuple of the minimum and maximum range from the edges of the grid in the y direction, in meters.
        raster_shape: A tuple of the shape of the target raster (assumed to be square).
        resolution: The pixel size in meters.
        route_roadblock_ids: route block id of the current map.
        draw_ego_route_separately: whether to extract self-route objects by route block ids and save in a different channel.

    Returns:
        A NumPy array of the constructed map_objects area raster layer.

    Raises:
        AssertionError: check the shape of data, and validation of input

    """
    # Ensure the raster has a square shape.
    # because nuplan map_api _get_extended_layer_coords requires square input
    assert (x_range[1] - x_range[0]) == (
        y_range[1] - y_range[0]
    ), f"Raster shape is assumed to be square but got width: {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}"

    # Compute the radius of the raster.
    radius = (x_range[1] - x_range[0]) / 2

    # Initialize the raster map to all zeros.
    if draw_ego_route_separately:
        map_objects: npt.NDArray[np.float32] = np.zeros((*raster_shape, 2),
                                                        dtype=np.float32)
    else:
        map_objects: npt.NDArray[np.float32] = np.zeros(
            raster_shape, dtype=np.float32)

    self_map_objects: npt.NDArray[np.float32] = np.zeros((*raster_shape, 1),
                                                         dtype=np.float32)
    other_map_objects: npt.NDArray[np.float32] = np.zeros((*raster_shape, 1),
                                                          dtype=np.float32)

    # get proximal coords. Raster objects are represented as coords, and we can draw linestrings or polygons with coords.
    # _get_extended_layer_coords in hoplan here implemented more layers than
    # nuplan version.
    coords, ids, _ = _get_extended_layer_coords(
        focus_agent, map_api, SemanticMapLayer[map_feature_name], 'polygon',
        radius)  # here must polygon no matter what

    # here piece of code is for boundaries only
    # if draw_ego_route_separately is true, we will select some of the coords from them. we will at first get lanes,
    # then check whether the lane's related block id is in the route_block_ids, and retained if yes.
    # at last, we filtered the result in previous get_coords to ensure no
    # outlier values.
    if map_feature_name == 'BOUNDARIES':
        lane_ids_by_routeblock_ids = [
            interior_edge.id for route_id in route_roadblock_ids
            for interior_edge in get_interior_edges(map_api, route_id)
        ]
        lane_ids_by_mapapi_and_route_ids = []
        self_coords = []
        self_ids = []
        other_coords = []
        other_ids = []
        for map_layer_name in ['LANE', 'LANE_CONNECTOR']:
            mp_coords, mp_ids, _ = _get_extended_layer_coords(
                focus_agent, map_api, SemanticMapLayer[map_layer_name],
                'polygon', radius)
            for index in range(len(mp_ids)):
                lane_id = mp_ids[index]
                if map_layer_name == 'LANE':
                    cnt_lane = map_api._get_lane(lane_id)
                elif map_layer_name == 'LANE_CONNECTOR':
                    cnt_lane = map_api._get_lane_connector(lane_id)
                else:
                    raise TypeError("Can't do for none lane type")
                if cnt_lane.id in lane_ids_by_routeblock_ids:
                    lane_ids_by_mapapi_and_route_ids.append(
                        cnt_lane.left_boundary.id)
                    lane_ids_by_mapapi_and_route_ids.append(
                        cnt_lane.right_boundary.id)
        # coords and ids are 1-1 map, so then must have same length
        assert len(coords) == len(ids)
        for index in range(len(ids)):
            if ids[index] in lane_ids_by_mapapi_and_route_ids:
                self_coords.append(coords[index])
                self_ids.append(ids[index])
            else:
                other_coords.append(coords[index])
                other_ids.append(ids[index])

    # for different features, we use different drawing methods. Some of them are drawed by plygon,
    # others are drawed by linestring.
    if map_feature_name in ["ROADBLOCK"]:
        # Encode the drivable area polygon in the raster map.
        feature_color = 1.0
        map_objects = _draw_polygon_image(map_objects, coords, radius,
                                          resolution, feature_color)

    elif map_feature_name in ["CARPARK_AREA", "WALKWAYS"]:
        ids = np.asarray(ids)
        lane_colors: npt.NDArray[np.uint8] = np.full(
            len(ids), 1, dtype=np.uint8)
        map_objects = _draw_linestring_image(
            image=map_objects,
            object_coords=coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=1,
            lane_colors=lane_colors,
            bit_shift=13,
        )
    elif map_feature_name in ["BOUNDARIES"]:

        self_ids = np.asarray(self_ids)
        self_colors: npt.NDArray[np.uint8] = np.full(
            len(self_ids), 1, dtype=np.uint8)

        other_ids = np.asarray(other_ids)
        other_colors: npt.NDArray[np.uint8] = np.full(
            len(other_ids), 1, dtype=np.uint8)
        # make in two parts for both draw_ego_route_separately is true or false

        self_map_objects = _draw_linestring_image(
            image=self_map_objects,
            object_coords=self_coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=1,
            lane_colors=self_colors,
            bit_shift=13,
        )

        other_map_objects = _draw_linestring_image(
            image=other_map_objects,
            object_coords=other_coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=1,
            lane_colors=other_colors,
            bit_shift=13,
        )
        # if self, draw a 6-channel with different color; if not, draw a
        # 3-channel with same color and overriden style
        if draw_ego_route_separately:
            map_objects = np.concatenate((other_map_objects, self_map_objects),
                                         axis=2)
        else:
            # Convert images to grayscale for bitwise operations
            gray1 = cv2.cvtColor(other_map_objects, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self_map_objects, cv2.COLOR_BGR2GRAY)

            # Combine the two images using bitwise operations
            mask = cv2.bitwise_or(gray1, gray2)
            map_objects = cv2.bitwise_and(
                other_map_objects, self_map_objects, mask=mask)
    else:
        raise TypeError(
            "Not implemented Error: select between polygon and linestring!")

    # Flip the raster map along the horizontal axis to match image conventions.
    map_objects = np.flip(map_objects, axis=0)

    # Ensure the raster map is contiguous and has the correct data type.
    map_objects = np.ascontiguousarray(map_objects, dtype=np.float32)

    return map_objects


def get_speed_limit_raster(
        focus_agent: EgoState,
        map_api: AbstractMap,
        map_feature_names: List[str],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        max_speed_normalizer: float,
        longitudinal_offset: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Construct a raster map layer of speed limit information by converting a vector map to a raster map.
    :param focus_agent: AgentState representing the ego vehicle.
    :param map_api: An instance of a map api.
    :param map_feature_names: A list of names of map features to be drawn and their corresponding colors for encoding.
    :param x_range: A tuple representing the minimum and maximum range from the edges of the grid in the x direction, in meters.
    :param y_range: A tuple representing the minimum and maximum range from the edges of the grid in the y direction, in meters.
    :param raster_shape: A tuple representing the desired shape of the target raster.
    :param resolution: The pixel size in meters.
    :param max_speed_normalizer: use max speed to normalize current speed
    :param longitudinal_offset: The longitudinal offset of the ego vehicle from the center of the raster map.
    :return: The constructed speed limit map raster layer.
    """
    # Assume the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (
        y_range[1] -
        y_range[0]), f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'

    radius = (x_range[1] - x_range[0]) / 2
    speed_limit_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    for feature_name in map_feature_names:
        coords, _, speed_limits = _get_extended_layer_coords(
            focus_agent, map_api, SemanticMapLayer[feature_name], 'polygon',
            radius, longitudinal_offset)
        for i in range(len(speed_limits)):
            speed_limit = speed_limits[i]
            speed_limit_raster = _draw_polygon_image(speed_limit_raster,
                                                     [coords[i]], radius,
                                                     resolution, speed_limit)

    # Flip the agents_raster along the horizontal axis.
    speed_limit_raster = np.flip(speed_limit_raster, axis=0)
    speed_limit_raster = np.ascontiguousarray(
        speed_limit_raster, dtype=np.float32)

    # Normalize the speed limit raster.
    speed_limit_raster /= max_speed_normalizer
    return speed_limit_raster


def _get_extended_layer_coords(
        ego_state: EgoState,
        map_api: AbstractMap,
        map_layer_name: SemanticMapLayer,
        map_layer_geometry: str,
        radius: float,
        longitudinal_offset: float = 0.0,
) -> Tuple[List[npt.NDArray[np.float64]], List[str], List[Union[float, None]]]:
    """
    Construct the map layer of the raster by converting vector map to raster map, based on the focus agent.
    :param ego_state: the focus agent used for raster generating.
    :param map_api: map api
    :param map_layer_name: name of the vector map layer to create a raster from.
    :param map_layer_geometry: geometric primitive of the vector map layer. i.e. either polygon or linestring.
    :param radius: [m] the radius of the square raster map.
    :return
        object_coords: the list of 2d coordinates which represent the shape of the map.
        lane_ids: the list of ids for the map objects.
        speed_limits: the speed limit of each lane. A None value indicates that
            speed limit is not available for that lane.
    """
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)
    ego_position = Point2D(virtual_center.x, virtual_center.y)
    nearest_vector_map = _get_proximal_map_objects(
        map_api=map_api,
        layers=[map_layer_name],
        point=ego_position,
        radius=radius,
    )

    geometry = nearest_vector_map[map_layer_name]

    if len(geometry):
        global_transform = np.linalg.inv(virtual_center.as_matrix())

        # By default the map is right-oriented, this makes it top-oriented.
        map_align_transform = R.from_euler(
            'z', 90, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform

        if map_layer_name in [SemanticMapLayer['BOUNDARIES']]:
            _object_coords = _polyline_to_coords(geometry)
        elif map_layer_geometry == 'polygon':
            _object_coords = _polygon_to_coords(geometry)
        elif map_layer_geometry == 'linestring':
            # This is actually baseline_path.linestring
            # TODO: make a better way to choose which to_coords to use.
            _object_coords = _linestring_to_coords(geometry)
        else:
            raise RuntimeError(
                f'Layer geometry {map_layer_geometry} type not supported')

        object_coords: List[npt.NDArray[np.float64]] = [
            np.vstack(coords).T for coords in _object_coords
        ]
        object_coords = [
            (transform @ _cartesian_to_projective_coords(coords).T).T[:, :2]
            for coords in object_coords
        ]
        lane_speed_limits = [
            lane.speed_limit_mps if hasattr(lane, 'speed_limit_mps') else None
            for lane in geometry
        ]
        lane_ids = [lane.id for lane in geometry]
    else:
        object_coords = []
        lane_ids = []
        lane_speed_limits = []

    return object_coords, lane_ids, lane_speed_limits


def _wod_lane_to_3d_coords(geometry: List[PolylineMapObject]) -> List[Tuple[Tuple(float)]]:
    """Get 3d coordinates from a wod lane polyline.

    :param map_feature: the map feature of wod lane.
    :return: 3d coordinates of the polyline.
    """
    all_coords = []
    for map_object in geometry:
        if len(map_object.map_feature.lane.polyline) == 1:  # handle only one point polyline situation
            init_x, init_y, init_z = map_object.map_feature.lane.polyline[0].x, map_object.map_feature.lane.polyline[0].y, map_object.map_feature.lane.polyline[0].z
            all_coords.append([(init_x, init_y, init_z), (init_x - 1e-3, init_y - 1e-3, init_z)])
        else:
            points = [(point.x, point.y, point.z) for point in map_object.map_feature.lane.polyline]
            for i in range(0, len(points), 10):
                sub_points = points[i:i + 10]
                if i + 11 >= len(points):
                    all_coords.append(tuple(points[i:]))
                else:
                    all_coords.append(tuple(sub_points))
    return all_coords


def _get_wod_layer_3d_coords(
        ego_state: EgoState,
        map_api: AbstractMap,
        map_layer_name: SemanticMapLayer,
        map_layer_geometry: str,
        radius: float,
        longitudinal_offset: float = 0.0,
) -> Tuple[List[npt.NDArray[np.float64]], List[str], List[Union[float, None]]]:
    """
    Construct the map layer of the raster by converting vector map to raster map, based on the focus agent.
    :param ego_state: the focus agent used for raster generating.
    :param map_api: map api
    :param map_layer_name: name of the vector map layer to create a raster from.
    :param map_layer_geometry: geometric primitive of the vector map layer. i.e. either polygon or linestring.
    :param radius: [m] the radius of the square raster map.
    :return
        object_coords: the list of 2d coordinates which represent the shape of the map.
        object_z: the z value of every point in the lane.
    """
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)
    ego_position = Point2D(virtual_center.x, virtual_center.y)
    nearest_vector_map = _get_proximal_map_objects(
        map_api=map_api,
        layers=[map_layer_name],
        point=ego_position,
        radius=radius,
    )

    geometry = nearest_vector_map[map_layer_name]

    if len(geometry):
        global_transform = np.linalg.inv(virtual_center.as_matrix())

        # By default the map is right-oriented, this makes it top-oriented.
        map_align_transform = R.from_euler(
            'z', 90, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform

        if map_layer_geometry == 'linestring':
            # This is actually baseline_path.linestring
            # TODO: make a better way to choose which to_coords to use.
            _object_coords = _wod_lane_to_3d_coords(geometry)
        else:
            raise RuntimeError(
                f'Layer geometry {map_layer_geometry} type not supported')

        object_coords: List[npt.NDArray[np.float64]] = [
            np.vstack(coords) for coords in _object_coords
        ]
        object_z = [coords[:,2] - 0.0 for coords in object_coords]
        object_coords = [
            (transform @ _cartesian_to_projective_coords(coords[:,:2]).T).T[:, :2]
            for coords in object_coords
        ]
    else:
        object_coords = []
        object_z = []

    return object_coords, object_z


def _get_proximal_map_objects(map_api: AbstractMap, point: Point2D,
                              radius: float, layers: List[SemanticMapLayer]
                              ) -> Dict[SemanticMapLayer, List[MapObject]]:
    """Get nearby map objects within the given radius.

    `NuPlanMap.get_proximal_map_object()` does not work for certain layers (
    e.g, `DRIVABLE_AREA` and `BOUNDARIES`).

    This function is a workaround for that.

    :param map_api: map api.
    :param point: center point.
    :param radius: [m] radius.
    :param layers: List of semantic map layers.
    :return object_map: a dictionary of map objects.
    """
    if all(layer in map_api.get_available_map_objects() for layer in layers):
        return map_api.get_proximal_map_objects(point, radius, layers)

    x_min, x_max = point.x - radius, point.x + radius
    y_min, y_max = point.y - radius, point.y + radius
    patch = geom.box(x_min, y_min, x_max, y_max)

    object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

    for layer in layers:
        object_map[layer] = _get_proximity_map_object(map_api, patch, layer)

    return object_map


def _get_proximity_map_object(map_api: AbstractMap, patch: geom.Polygon,
                              layer: SemanticMapLayer) -> List[MapObject]:
    """Gets nearby lanes within the given patch.

    Similar to `NuPlanMap.get_proximal_map_object()`, but only works for
    `DRIVABLE_AREA` and `BOUNDARIES`.

    :param patch: The area to be checked.
    :param layer: desired layer to check.
    :return: A list of map objects.
    """
    layer_df = map_api._get_vector_map_layer(layer)
    map_object_ids = layer_df[layer_df['geometry'].intersects(patch)]['fid']
    if layer == SemanticMapLayer.DRIVABLE_AREA:
        return [NuPlanPolygonMapObject(i, layer_df) for i in map_object_ids]
    elif layer == SemanticMapLayer.BOUNDARIES:
        return [NuPlanPolylineMapObject(i) for i in map_object_ids]
    else:
        raise ValueError(f"Unsupported layer: {layer}")


def get_traffic_light_dict_from_generator(
        tl_generator: Generator[TrafficLightStatusData, None, None],
) -> Dict[TrafficLightStatusType, List[str]]:
    """
    This function simply converts data format. From generator to dict.
    :param tl_generator: the generator got from map_api.
    :return: the dict for latter process.
    """
    traffic_light_by_iter_list = list(tl_generator)
    # 4 stands for 4 types: red, yellow, green and unknown.
    traffic_light_by_iter_dict = {
        TrafficLightStatusType(i_enum): []
        for i_enum in range(4)
    }
    for traffic_light_by_iter in traffic_light_by_iter_list:
        traffic_light_by_iter_dict[traffic_light_by_iter.status].append(
            traffic_light_by_iter.lane_connector_id)
    return traffic_light_by_iter_dict


def _draw_polygon_image_by_array_colors(
        image: npt.NDArray[np.float32],
        object_coords: List[npt.NDArray[np.float64]],
        radius: float,
        resolution: float,
        array_colors: npt.NDArray[np.uint8],
        bit_shift: int = 12,
) -> npt.NDArray[np.float32]:
    """
    Draw a map feature consisting of polygons using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resolution: [m] pixel size in meters.
    :param color: color of the map feature.
    :param bit_shift: bit shift of the polygon used in opencv.
    :return: the resulting raster map with the map feature.
    """
    assert len(object_coords) == len(array_colors)
    if len(object_coords):
        for coords, lane_color in zip(object_coords, array_colors):
            index_coords = (radius + coords) / resolution
            shifted_index_coords = (index_coords * 2**bit_shift).astype(
                np.int64)
            # Add int() before lane_color to address the cv2 error: color
            # should be numeric
            lane_color = int(lane_color) if np.isscalar(lane_color) else [
                int(item) for item in lane_color
            ]  # type: ignore
            cv2.fillPoly(
                image,
                shifted_index_coords[None],
                color=lane_color,
                shift=bit_shift,
                lineType=cv2.LINE_AA)

    return image


def get_lane_colors(
        lane_ids: List[str],
        traffic_light_connectors: Dict[TrafficLightStatusType, List[str]],
) -> npt.NDArray[np.unit8]:
    """
    Get lane color from lane_connector color dict.
    :param lane_ids: the list of lane_ids which we want the color for
    :param traffic_light_connectors: our dict for traffic light information
    :return: lane_colors array for input lanes.
    """
    # Get a list indicating the color of each lane
    lane_ids = [int(x) for x in lane_ids]
    lane_ids = np.asarray(lane_ids)  # type: ignore
    lane_colors: npt.NDArray[np.uint8] = np.full(
        (len(lane_ids), 3),
        BASELINE_TL_COLOR[TrafficLightStatusType.UNKNOWN],
        dtype=np.uint8)

    # If we have valid traffic light informaiton for intersection connectors
    if len(traffic_light_connectors) > 0:
        # here the for-loop is itered 4 times, for 4 types.
        for tl_status in TrafficLightStatusType:
            if tl_status != TrafficLightStatusType.UNKNOWN and len(
                    traffic_light_connectors[tl_status]) > 0:
                # lanes_in_tl_status is a 0-1 array to indicate where to draw
                # colors
                lanes_in_tl_status = np.isin(
                    lane_ids, traffic_light_connectors[tl_status])
                # draw colors to where lanes_in_tl_status is 1 to get
                # lane_colors
                lane_colors[lanes_in_tl_status] = BASELINE_TL_COLOR[tl_status]
    return lane_colors

def draw_arrow_triangle(image, x, y, heading, length, base_width, filled_color):
    """
    Draw a filled, arrow-like triangle on an image to represent direction.

    :param image: The image to draw on
    :param x: The x-coordinate of the triangle's base center
    :param y: The y-coordinate of the triangle's base center
    :param heading: The heading angle of the triangle in degrees
    :param length: The length of the triangle (from base to tip)
    :param base_width: The width of the triangle's base
    """
    # Convert heading to radians
    heading_rad = np.deg2rad(heading)
    
    # Calculate the triangle's vertices
    tip = (int(x + length * np.cos(heading_rad)), int(y + length * np.sin(heading_rad)))
    base_left = (int(x - (base_width / 2) * np.sin(heading_rad)), int(y + (base_width / 2) * np.cos(heading_rad)))
    base_right = (int(x + (base_width / 2) * np.sin(heading_rad)), int(y - (base_width / 2) * np.cos(heading_rad)))
    
    # Combine the points into an array
    triangle_cnt = np.array([tip, base_left, base_right])
    
    # Draw the filled triangle
    cv2.fillPoly(image, [triangle_cnt], filled_color)  # Blue color in BGR
    return image


def draw_traffic_light_at_connector_start(
    traffic_light_raster,
    lane_ids,
    baseline_paths_coords,
    radius,
    resolution,
    traffic_light_connectors
) -> npt.NDArray[np.unit8]:
    """
    Get lane color from lane_connector color dict.
    :param lane_ids: the list of lane_ids which we want the color for
    :param traffic_light_connectors: our dict for traffic light information
    :return: lane_colors array for input lanes.
    """
    # Get a list indicating the color of each lane
    lane_ids = [int(x) for x in lane_ids]
    traffic_index = 1
    for traffic_light_status, traffic_lane_ids in traffic_light_connectors.items():
        # if TrafficLightStatusType.UNKNOWN != traffic_light_status:
        for traffic_lane_id in traffic_lane_ids:
            if traffic_lane_id in lane_ids:
                lane_index = lane_ids.index(traffic_lane_id)
                lane_coords = baseline_paths_coords[lane_index]
                traffic_light_raster[...,0] = _draw_linestring_image(
                    image=traffic_light_raster[...,0],
                    object_coords=[lane_coords],
                    radius=radius,
                    resolution=resolution,
                    baseline_path_thickness=1,
                    lane_colors=[traffic_index],
                )
                heading = np.arctan2(lane_coords[1][1] - lane_coords[0][1], lane_coords[1][0] - lane_coords[0][0])
                x = round((radius + lane_coords[0][0]) / resolution)
                y = round((radius + lane_coords[0][1]) / resolution)
                traffic_light_raster = cv2.circle(traffic_light_raster, (int(x), int(y)), 4, traffic_index, thickness=-1) 
                traffic_index += 1
    return traffic_light_raster.squeeze(-1)

def get_traffic_light_circle_raster(
        ego_state: EgoState,
        map_api: AbstractMap,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        traffic_light_connectors: Dict[TrafficLightStatusType, List[str]],
        longitudinal_offset: float = 0.0, 
) -> npt.NDArray[np.float32]:
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')

    radius = (x_range[1] - x_range[0]) / 2

    traffic_light_raster: npt.NDArray[np.float32] = np.zeros((*raster_shape, 1), dtype=np.float32)

    baseline_paths_coords = []
    lane_ids = []
    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coord, lane_id = _get_layer_coords(
            ego_state=ego_state,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_features],
            map_layer_geometry='linestring',
            radius=radius,
            longitudinal_offset=longitudinal_offset,
        )
        baseline_paths_coords += baseline_paths_coord
        lane_ids += lane_id

    # get lane_colors for the two type.
    traffic_light_raster = draw_traffic_light_at_connector_start(
        traffic_light_raster,
        lane_ids,
        baseline_paths_coords,
        radius,
        resolution,
        traffic_light_connectors)
    traffic_light_raster = np.flip(traffic_light_raster, axis=0)
    traffic_light_raster = np.ascontiguousarray(
        traffic_light_raster, dtype=np.float32)
    return traffic_light_raster



def get_traffic_light_raster(
        ego_state: EgoState,
        map_api: AbstractMap,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        traffic_light_connectors: Dict[TrafficLightStatusType, List[str]],
        route_roadblock_ids: List[str],
        draw_by_polygon_tl: bool = True,
        draw_ego_route_separately_tl: bool = True,
        baseline_path_thickness: int = 1,
        longitudinal_offset: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Mainly this function help draw traffic light color onto baseline path.
    Construct the baseline paths layer by converting vector map to raster map.
    This function is for agents raster model, it has 3 channels for baseline path.
    Support polygon/linestring, support self/other different color.
    :param focus_agent: agent state representing ego.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param traffic_light_connectors: a dict mapping tl status type to a list of lane ids in this status.
    :param route_roadblock_ids: route block id of the current map.
    :param draw_by_polygon_tl: whether to build the raster map as polygon or not. If not, build as linestring.
    :param draw_ego_route_separately_tl: whether to extract self-route traffic light by route block ids.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')

    radius = (x_range[1] - x_range[0]) / 2

    other_paths_raster: npt.NDArray[np.float32] = np.zeros((*raster_shape, 3),
                                                           dtype=np.float32)
    self_paths_raster: npt.NDArray[np.float32] = np.zeros((*raster_shape, 3),
                                                          dtype=np.float32)

    # if draw_ego_route_separately_tl, draw a 6-channel pic; if not, draw a
    # 3-channel one
    if draw_ego_route_separately_tl:
        baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(
            (*raster_shape, 6), dtype=np.float32)
    else:
        baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(
            (*raster_shape, 3), dtype=np.float32)

    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coords, lane_ids, _ = _get_extended_layer_coords(
            ego_state=ego_state,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_features],
            map_layer_geometry='polygon',
            radius=radius,
            longitudinal_offset=longitudinal_offset,
        )

        self_lane_ids = []
        self_coords = []
        other_lane_ids = []
        other_coords = []
        # split lane_ids and baseline_paths_coords into two parts: self_ and
        # other_
        for index in range(len(lane_ids)):
            lane_id = lane_ids[index]
            coord = baseline_paths_coords[index]
            if map_features == 'LANE':
                cnt_lane = map_api._get_lane(lane_id)
            elif map_features == 'LANE_CONNECTOR':
                cnt_lane = map_api._get_lane_connector(lane_id)
            else:
                raise TypeError("Can't do for none lane type")
            cnt_block_id = cnt_lane.get_roadblock_id()
            # judge whether lane_ids and baseline_paths_coords are in route_roadblock_ids,
            # to see if it's self_ or other_
            if cnt_block_id in route_roadblock_ids:
                self_lane_ids.append(lane_id)
                self_coords.append(coord)
            else:
                other_lane_ids.append(lane_id)
                other_coords.append(coord)
        # get lane_colors for the two type.
        other_lane_colors = get_lane_colors(other_lane_ids,
                                            traffic_light_connectors)
        self_lane_colors = get_lane_colors(self_lane_ids,
                                           traffic_light_connectors)

        # for both draw_ego_route_separately_tl is true and false, we make the raster by two parts.
        # But if draw_ego_route_separately_tl is true, we use different colors.
        if draw_by_polygon_tl:
            # draw twice with different color, one for self and one for other
            # for other
            other_paths_raster = _draw_polygon_image_by_array_colors(
                other_paths_raster, other_coords, radius, resolution,
                other_lane_colors)

            # for self
            self_paths_raster = _draw_polygon_image_by_array_colors(
                self_paths_raster, self_coords, radius, resolution,
                self_lane_colors)

        else:  # default as linestring
            # draw for other
            other_paths_raster = _draw_linestring_image(
                image=other_paths_raster,
                object_coords=other_coords,
                radius=radius,
                resolution=resolution,
                baseline_path_thickness=baseline_path_thickness,
                lane_colors=other_lane_colors,
            )
            # draw for self
            self_paths_raster = _draw_linestring_image(
                image=self_paths_raster,
                object_coords=self_coords,
                radius=radius,
                resolution=resolution,
                baseline_path_thickness=baseline_path_thickness,
                lane_colors=self_lane_colors,
            )

    # if self, draw a 6-channel with different color; if not, draw a 3-channel
    # with same color and overriden style
    if draw_ego_route_separately_tl:
        baseline_paths_raster = np.concatenate(
            (other_paths_raster, self_paths_raster), axis=2)
    else:
        # Convert images to grayscale for bitwise operations
        gray1 = cv2.cvtColor(other_paths_raster, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self_paths_raster, cv2.COLOR_BGR2GRAY)

        # Combine the two images using bitwise operations
        mask = cv2.bitwise_or(gray1, gray2)
        baseline_paths_raster = cv2.bitwise_and(
            other_paths_raster, self_paths_raster, mask=mask)

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(
        baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster


def get_wod_roadmap_raster(
        focus_agent: AgentState,
        map_api: AbstractMap,
        map_features: Dict[str, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        linestring_thickness: int = 1,
) -> npt.NDArray[np.float32]:
    """
    Construct the map layer of the raster by converting vector map to raster map.
    :param focus_agent: agent state representing ego.
    :param map_api: map api.
    :param map_features: name of map features to be drawn and its color for encoding.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :return roadmap_raster: the constructed map raster layer.
    """
    # Assume the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (
        y_range[1] -
        y_range[0]), f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'

    radius = (x_range[1] - x_range[0]) / 2
    roadmap_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    for feature_name, feature_color in map_features.items():
        if feature_name in ["BASELINE_PATHS", "EXTENDED_PUDO"]:
            coords, line_ids = _get_layer_coords(
                focus_agent, map_api, SemanticMapLayer[feature_name], 'linestring', radius)
            lane_colors: npt.NDArray[np.uint8] = np.ones(
                len(line_ids)).astype(np.uint8)
            roadmap_raster = _draw_linestring_image(
                image=roadmap_raster,
                object_coords=coords,
                radius=radius,
                resolution=resolution,
                baseline_path_thickness=linestring_thickness,
                lane_colors=lane_colors)
        else:
            coords, _, _ = _get_extended_layer_coords(
                focus_agent, map_api, SemanticMapLayer[feature_name],
                'polygon', radius)
            roadmap_raster = _draw_polygon_image(
                roadmap_raster, coords, radius, resolution, feature_color)

    # Flip the agents_raster along the horizontal axis.
    roadmap_raster = np.flip(roadmap_raster, axis=0)
    roadmap_raster = np.ascontiguousarray(roadmap_raster, dtype=np.float32)

    return roadmap_raster


def get_wod_baseline_paths_raster(
        ego_state: AgentState,
        map_api: AbstractMap,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        baseline_path_thickness: int = 1,
) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This funciton is for ego raster model, the baselin path only has one channel.
    :param ego_state: SE2 state of ego.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    for map_features in ['LANE']:
        baseline_paths_coords, lane_ids, _ = _get_extended_layer_coords(
            ego_state=ego_state,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_features],
            map_layer_geometry='linestring',
            radius=radius,
        )
        lane_colors: npt.NDArray[np.uint8] = np.ones(len(lane_ids)).astype(
            np.uint8)
        baseline_paths_raster = _draw_linestring_image(
            image=baseline_paths_raster,
            object_coords=baseline_paths_coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=baseline_path_thickness,
            lane_colors=lane_colors,
        )

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(
        baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster


def get_wod_baseline_z_raster(
        ego_state: AgentState,
        map_api: AbstractMap,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        baseline_path_thickness: int = 1,
) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This funciton is for ego raster model, the baselin path only has one channel.
    :param ego_state: SE2 state of ego.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    for map_feature in ['LANE']:
        baseline_paths_coords, baseline_path_z = _get_wod_layer_3d_coords(
            ego_state=ego_state,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_feature],
            map_layer_geometry='linestring',
            radius=radius,
        )

        lane_colors: npt.NDArray[np.uint8] = np.zeros(len(baseline_paths_coords)).astype(
            np.uint8)
        for i in range(len(baseline_paths_coords)):
            lane_colors[i] = int(np.clip((np.mean(baseline_path_z[i]) / 5 * 128) + 128, 0, 255))

        baseline_paths_raster = _draw_linestring_image(
            image=baseline_paths_raster,
            object_coords=baseline_paths_coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=baseline_path_thickness,
            lane_colors=lane_colors,
        )

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(
        baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster


def get_agents_raster(
        ego_state: EgoState,
        tracked_objects: List[TrackedObjects],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        polygon_bit_shift: int = 9,
        with_instance_mask: bool = False,
        longitudinal_offset: float = 0.0,
):
    """
    Construct the agents layer of the raster by transforming all detected boxes around the agent
    and creating polygons of them in a raster grid, and also can create the corresponding instance mask.

    :param ego_state: SE2 state of ego.
    :param tracked_objects: list of TrackedObjects.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param polygon_bit_shift: bit shift of the polygon used in opencv.
    :param with_instance_mask: whether to return instance mask.
    :param longitudinal_offset: [m] longitudinal offset of the virtual center.
    :return: constructed agents raster layer.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    # ego_to_global = ego_state.rear_axle.as_matrix()
    radius = (x_range[1] - x_range[0]) / 2
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)

    ego_to_global = virtual_center.as_matrix()
    global_to_ego = np.linalg.inv(ego_to_global)

    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()
    agents_raster = np.zeros(raster_shape, dtype=np.float32)
    if with_instance_mask:
        instances_raster = np.zeros(raster_shape, dtype=np.float32)

    for tracked_object in tracked_objects:
        # Transform the box relative to agent.
        raster_object_matrix = north_aligned_transform @ global_to_ego @ tracked_object.center.as_matrix(
        )
        raster_object_pose = StateSE2.from_matrix(raster_object_matrix)
        # Filter out boxes outside the raster.
        valid_x = x_range[0] < raster_object_pose.x < x_range[1]
        valid_y = y_range[0] < raster_object_pose.y < y_range[1]
        if not (valid_x and valid_y):
            continue

        # Get the 2D coordinate of the detected agents.
        raster_oriented_box = OrientedBox(
            raster_object_pose, tracked_object.box.length,
            tracked_object.box.width, tracked_object.box.height)
        box_bottom_corners = raster_oriented_box.all_corners()
        x_corners = np.asarray(
            [corner.x for corner in box_bottom_corners])  # type: ignore
        y_corners = np.asarray(
            [corner.y for corner in box_bottom_corners])  # type: ignore

        # Discretize
        y_corners = (y_corners - ymin) / (ymax - ymin) * height  # type: ignore
        x_corners = (x_corners - xmin) / (xmax - xmin) * width  # type: ignore

        box_2d_coords = np.stack([x_corners, y_corners],
                                 axis=1)  # type: ignore
        box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

        # Draw the box as a filled polygon on the raster layer.
        value = tracked_object.tracked_object_type.value + 1
        box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)
        cv2.fillPoly(
            agents_raster,
            box_2d_coords,
            color=value,
            shift=polygon_bit_shift,
            lineType=cv2.LINE_AA)
        if with_instance_mask:
            value = tracked_object.metadata.track_id
            cv2.fillPoly(
                instances_raster,
                box_2d_coords,
                color=value,
                shift=polygon_bit_shift,
                lineType=cv2.LINE_AA)

    # Flip the agents_raster along the horizontal axis.
    agents_raster = np.asarray(agents_raster)
    agents_raster = np.flip(agents_raster, axis=0)
    agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

    if with_instance_mask:
        instances_raster = np.asarray(instances_raster)
        instances_raster = np.flip(instances_raster, axis=0)
        instances_raster = np.ascontiguousarray(
            instances_raster, dtype=np.uint16)
        return agents_raster, instances_raster
    return agents_raster


def calculate_tilt_angle(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
) -> float:
    """
    Calculate the tilt angle for 2 given points.
    :param point_a: the first point to be calculate.
    :param point_b: the second point to be calculate.
    :return: the resulting tilt angle in rad.
    """
    x1, y1 = point_a
    x2, y2 = point_b

    # Check for the same points
    if x1 == x2 and y1 == y2:
        return 0.0
        # raise ValueError("The points are the same, tilt angle cannot be calculated.")

    # Calculate the difference in x and y coordinates
    dx = x2 - x1
    dy = y2 - y1

    # Calculate tilt angle in radians
    tilt_angle = math.atan2(dy, dx)

    return tilt_angle


def _draw_linestring_image_polar_direction(
        image: npt.NDArray[np.float32],
        object_coords: npt.NDArray[np.float64],
        radius: float,
        resolution: float,
        baseline_path_thickness: int,
        bit_shift: int = 13,
) -> npt.NDArray[np.float32]:
    """
    Draw a map feature consisting of linestring using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param bit_shift: bit shift of the polylines used in opencv.
    :return: the resulting raster map with the map feature.
    """
    index_coords = (radius + object_coords) / resolution
    points = (index_coords * 2**bit_shift).astype(np.int64)
    num_segments = len(points) - 1

    for i in range(num_segments):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])

        # Calculate the color for the current segment
        segment_color = calculate_tilt_angle(
            pt1, pt2) + 2 * np.pi  # result ranging from pi to 3pi

        # Draw the segment
        cv2.line(
            image,
            pt1,
            pt2,
            segment_color,
            baseline_path_thickness,
            lineType=cv2.LINE_AA,
            shift=bit_shift)
    return image


def align_edges(
        edges: Union[List[NuPlanLane], List[NuPlanRoadBlock]],
        ids: List[str],
) -> Union[Dict[NuPlanLane, int], Dict[NuPlanRoadBlock, int]]:
    """
    Align the edges in an order of layer.
    :param point_a: the edges to be calculated.
    :param point_b: the ids of the edges.
    :return: a dict of aligned edges by incoming edges and outgoing edges.
    """
    # Create a dictionary for easy edge lookup by ID
    edge_dict = {edge.id: edge for edge in edges if edge.id in ids}

    # Filter the edges to include only those in the ids list
    for edge_id, edge in edge_dict.items():
        edge.incoming_edges = [e for e in edge.incoming_edges if e.id in ids]
        edge.outgoing_edges = [e for e in edge.outgoing_edges if e.id in ids]
        edge_dict[edge_id] = edge

    # Initialize layer assignment
    layers = {}

    # BFS for layer assignment
    def assign_layers(start_edge):
        queue = deque([(start_edge, 0)])

        while queue:
            edge, layer = queue.popleft()

            if edge.id in layers:
                layers[edge.id] = max(layer, layers[edge.id])
                continue

            if edge.id not in layers:  # or layer < layers[edge.id]
                layers[edge.id] = layer
                for next_edge in edge.outgoing_edges:
                    # can not use next_edge directly, because only in the dict its
                    # attributes incoming and outgoing are pruned
                    queue.append((edge_dict[next_edge.id], layer + 1))

    # Find all nodes with no incoming edges and start BFS from them
    no_incoming = [
        edge for edge in edge_dict.values() if not edge.incoming_edges
    ]
    for edge in no_incoming:
        assign_layers(edge)

    # If there are still nodes with no layer assigned, run BFS on them
    for edge in edge_dict.values():
        if edge.id not in layers:
            assign_layers(edge)

    return layers


def _draw_linestring_image_gradient(
        image: npt.NDArray[np.float32],
        object_coords: npt.NDArray[np.float64],
        radius: float,
        resolution: float,
        baseline_path_thickness: int,
        color_range: tuple[npt.NDArray[np.float32]],
        bit_shift: int = 13,
) -> npt.NDArray[np.float32]:
    """
    Draw a map feature consisting of linestring using a list of its coordinates.
    :param image: the raster map on which the map feature will be drawn
    :param object_coords: the coordinates that represents the shape of the map feature.
    :param radius: the radius of the square raster map.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param color_range: the range of color used for this line.
    :param bit_shift: bit shift of the polylines used in opencv.
    :return: the resulting raster map with the map feature.
    """
    index_coords = (radius + object_coords) / resolution
    points = (index_coords * 2**bit_shift).astype(np.int64)
    num_segments = len(points) - 1

    for i in range(num_segments):
        pt1 = tuple(points[i])
        pt2 = tuple(points[i + 1])

        # Calculate the color for the current segment
        if num_segments > 1:
            t = i / (num_segments - 1)
        else:
            t = 0
        segment_color = color_range[0] * (1 - t) + color_range[1] * t

        # Draw the segment
        cv2.line(
            image,
            pt1,
            pt2,
            segment_color,
            baseline_path_thickness,
            lineType=cv2.LINE_AA,
            shift=bit_shift)
    return image


def get_gradient_lane_raster(
        focus_agent: EgoState,
        map_api: AbstractMap,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        baseline_path_thickness: int = 1,
        longitudinal_offset: Optional[float] = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This funciton is for lane raster. Draw the lane raster so that the color gradient indicates the direction.
    :param focus_agent: The state of the agent representing the ego vehicle.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param longitudinal_offset: [m] the longitudinal offset of the baseline path.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    lane_coords, lane_ids, _ = _get_extended_layer_coords(
        focus_agent, map_api, SemanticMapLayer['LANE'], 'linestring', radius,
        longitudinal_offset)

    lane_connector_coords, lane_connector_ids, _ = _get_extended_layer_coords(
        focus_agent, map_api, SemanticMapLayer['LANE_CONNECTOR'], 'linestring',
        radius, longitudinal_offset)

    coords = lane_coords + lane_connector_coords
    ids = lane_ids + lane_connector_ids

    edges = [
        map_api._get_lane(lane_id)
        if lane_id in lane_ids else map_api._get_lane_connector(lane_id)
        for lane_id in ids
    ]
    edge_dict = {edge.id: edge for edge in edges if edge.id in ids}
    layer_dict = align_edges(edges, ids)
    assert len(ids) == len(edges) == len(layer_dict) == len(edge_dict)

    assert min(layer_dict.values()) == 0
    # +1 because layer is 0-index, +1 because n+1 colors to form a range for n targets
    colors = np.linspace(0.0001, 0.9999,
                         max(layer_dict.values()) + 2).astype(np.float32)
    for lane_id, layer_index in layer_dict.items():
        index = ids.index(lane_id)
        drawing = coords[index]
        color_range = colors[layer_index:layer_index + 2]
        baseline_paths_raster = _draw_linestring_image_gradient(
            image=baseline_paths_raster,
            object_coords=drawing,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=baseline_path_thickness,
            color_range=color_range,
        )

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(
        baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster


def get_polor_direction_lane_raster(
        focus_agent: EgoState,
        map_api: AbstractMap,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        baseline_path_thickness: int = 1,
        longitudinal_offset: Optional[float] = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This funciton is for lane raster, the color indicates the rad value of the yaw of each position of the lane.
    :param focus_agent: The state of the agent representing the ego vehicle.
    :param map_api: map api
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param longitudinal_offset: [m] the longitudinal offset of the baseline path.
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}')

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    lane_coords, lane_ids, _ = _get_extended_layer_coords(
        focus_agent, map_api, SemanticMapLayer['LANE'], 'linestring', radius,
        longitudinal_offset)

    lane_connector_coords, lane_connector_ids, _ = _get_extended_layer_coords(
        focus_agent, map_api, SemanticMapLayer['LANE_CONNECTOR'], 'linestring',
        radius, longitudinal_offset)

    coords = lane_coords + lane_connector_coords
    ids = lane_ids + lane_connector_ids

    for coord, id in zip(coords, ids):
        baseline_paths_raster = _draw_linestring_image_polar_direction(
            image=baseline_paths_raster,
            object_coords=coord,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=baseline_path_thickness,
        )

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(
        baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster


def rotate_dx_dy_in_complex_space(
        input_raster: npt.NDArray[np.float32],
        phi: float,
) -> npt.NDArray[np.float32]:
    """
    :param inp_raster: input raster to be rotated
    :param phi: the angle to be rotated. counterclockwise, in radians.
    """
    return_raster = np.zeros(input_raster.shape)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return_raster[0] = input_raster[0] * cos_phi - input_raster[1] * sin_phi
    return_raster[1] = input_raster[1] * cos_phi + input_raster[0] * sin_phi
    return return_raster


def rotate_tilt_angle(
        input_raster: npt.NDArray[np.float32],
        phi: float,
):
    """
    Align the values after rotating a raster of tilt angle, and ensure the range is (pi,3pi) and 0 for non-occupancy
    :param inp_raster: input raster to be rotated
    :param phi: the angle to be rotated. counterclockwise, in radians.
    """
    non_zero_mask = input_raster != 0.0
    # only modify values in non_zero_mask
    input_raster[non_zero_mask] = (input_raster[non_zero_mask] + phi) % (
        2 * np.pi) + np.pi
    return input_raster


def get_mission_goal_raster(
        target_relative_pose: npt.NDArray[np.float32],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[float, float],
        draw_at_bound_each: bool,
) -> npt.NDArray[np.float32]:
    """
    Construct mission goal of the current scenario.
    :param future_trajectory: future trajectory of ego vehicle.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param draw_at_bound_each: whether to draw the mission goal at the boundary of each frame.
    :return: constructed mission goal raster layer.
    """
    target_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    xmin, xmax = x_range
    ymin, ymax = y_range
    width, height = raster_shape

    # Discretize
    target_y = (target_relative_pose[-1, 0] - ymin) / \
        (ymax - ymin) * height  # type: ignore
    target_x = (target_relative_pose[-1, 1] - xmin) / \
        (xmax - xmin) * width  # type: ignore

    target_x, target_y = target_x.astype(np.int64), target_y.astype(np.int64)

    if draw_at_bound_each:
        target_x = np.clip(target_x, 0, height - 1)
        target_y = np.clip(target_y, 0, width - 1)
        cv2.circle(
            target_raster, (target_x, target_y),
            radius=0,
            color=1.0,
            thickness=-1)
    else:
        if 0 <= target_x < height and 0 <= target_y < width:
            cv2.circle(
                target_raster, (target_x, target_y),
                radius=0,
                color=1.0,
                thickness=-1)

    target_raster = np.flip(target_raster, axis=0)
    target_raster = np.ascontiguousarray(target_raster, dtype=np.float32)
    return target_raster


def _get_layer_coords(
        ego_state: Union[AgentState, EgoState],
        map_api: AbstractMap,
        map_layer_name: SemanticMapLayer,
        map_layer_geometry: str,
        radius: float,
        longitudinal_offset: float = 0.0,
) -> Tuple[List[npt.NDArray[np.float64]], List[str]]:
    """
    Construct the map layer of the raster by converting vector map to raster map, based on the focus agent.
    :param ego_state (AgentState, EgoState): current state of focus agent.
    :param map_api: map api
    :param map_layer_name: name of the vector map layer to create a raster from.
    :param map_layer_geometry: geometric primitive of the vector map layer. i.e. either polygon or linestring.
    :param radius: [m] the radius of the square raster map.
    :param longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center
    :return
        object_coords: the list of 2d coordinates which represent the shape of the map.
        lane_ids: the list of ids for the map objects.
    """
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)
    ego_position = Point2D(virtual_center.x, virtual_center.y)
    nearest_vector_map = map_api.get_proximal_map_objects(
        layers=[map_layer_name],
        point=ego_position,
        radius=radius,
    )
    geometry = nearest_vector_map[map_layer_name]

    if len(geometry):
        global_transform = np.linalg.inv(virtual_center.as_matrix())

        # By default the map is right-oriented, this makes it top-oriented.
        map_align_transform = R.from_euler(
            'z', 90, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform

        if map_layer_geometry == 'polygon':
            _object_coords = _polygon_to_coords(geometry)
        elif map_layer_geometry == 'linestring':
            _object_coords = _linestring_to_coords(geometry)
        else:
            raise RuntimeError(
                f'Layer geometry {map_layer_geometry} type not supported')

        object_coords: List[npt.NDArray[np.float64]] = [
            np.vstack(coords).T for coords in _object_coords
        ]
        object_coords = [
            (transform @ _cartesian_to_projective_coords(coords).T).T[:, :2]
            for coords in object_coords
        ]

        lane_ids = [lane.id for lane in geometry]
    else:
        object_coords = []
        lane_ids = []

    return object_coords, lane_ids


def get_baseline_paths_raster(
        ego_state: Union[AgentState, EgoState],
        map_api: AbstractMap,
        map_feature_names: List[str],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        baseline_path_thickness: int = 1,
        longitudinal_offset: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Construct the baseline paths layer by converting vector map to raster map.
    This funciton is for ego raster model, the baselin path only has one channel.
    :param focus_agent (AgentState, EgoState): current state of focus agent.
    :param map_api: map api
    :param map_feature_names: name of the vector map layer to create a raster from.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param baseline_path_thickness: [pixel] the thickness of polylines used in opencv.
    :param longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center
    :return baseline_paths_raster: the constructed baseline paths layer.
    """
    # Assume the raster has a square shape.
    if (x_range[1] - x_range[0]) != (y_range[1] - y_range[0]):
        raise ValueError(
            f'Raster shape is assumed to be square but got width: \
                {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'
        )

    radius = (x_range[1] - x_range[0]) / 2
    baseline_paths_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    for map_features in map_feature_names:
        baseline_paths_coords, lane_ids = _get_layer_coords(
            ego_state=ego_state,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_features],
            map_layer_geometry='linestring',
            radius=radius,
            longitudinal_offset=longitudinal_offset,
        )
        lane_colors: npt.NDArray[np.uint8] = np.ones(len(lane_ids)).astype(
            np.uint8)
        baseline_paths_raster = _draw_linestring_image(
            image=baseline_paths_raster,
            object_coords=baseline_paths_coords,
            radius=radius,
            resolution=resolution,
            baseline_path_thickness=baseline_path_thickness,
            lane_colors=lane_colors,
        )

    # Flip the agents_raster along the horizontal axis.
    baseline_paths_raster = np.flip(baseline_paths_raster, axis=0)
    baseline_paths_raster = np.ascontiguousarray(
        baseline_paths_raster, dtype=np.float32)
    return baseline_paths_raster


def get_p3c_roadmap_raster(
        focus_agent: Union[AgentState, EgoState],
        map_api: AbstractMap,
        map_features: Dict[str, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        linestring_thickness: int = 1,
        longitudinal_offset: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Construct the map layer of the raster by converting vector map to raster map.
    :param focus_agent (AgentState, EgoState): current state of agent.
    :param map_api: map api.
    :param map_features: name of map features to be drawn and its color for encoding.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param linestring_thickness: thickness of linestring
    :param longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center
    :return roadmap_raster: the constructed map raster layer.
    """
    # Assume the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (
        y_range[1] -
        y_range[0]), f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'

    radius = (x_range[1] - x_range[0]) / 2
    roadmap_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    for feature_name, feature_color in map_features.items():
        if feature_name in ['BOUNDARIES']:
            coords, _ = _get_layer_coords(
                focus_agent, map_api, SemanticMapLayer[feature_name],
                'linestring', radius, longitudinal_offset)
            roadmap_raster = _draw_linestring_image(
                image=roadmap_raster,
                object_coords=coords,
                radius=radius,
                resolution=resolution,
                baseline_path_thickness=linestring_thickness,
                lane_colors=[feature_color for i in range(len(coords))])
        elif feature_name in ['LANE_EDGE']:
            coords, colors = get_lane_edge_layer_coords(
                focus_agent, map_api, radius, longitudinal_offset)
            roadmap_raster = _draw_linestring_image(
                image=roadmap_raster,
                object_coords=coords,
                radius=radius,
                resolution=resolution,
                baseline_path_thickness=linestring_thickness,
                lane_colors=colors)

        else:
            coords, _ = _get_layer_coords(
                focus_agent, map_api, SemanticMapLayer[feature_name],
                'polygon', radius, longitudinal_offset)
            roadmap_raster = _draw_polygon_image(
                image=roadmap_raster,
                object_coords=coords,
                radius=radius,
                resolution=resolution,
                color=feature_color)

    roadmap_raster = np.flip(roadmap_raster, axis=0)
    roadmap_raster = np.ascontiguousarray(roadmap_raster, dtype=np.float32)
    return roadmap_raster


def get_p3c_goal_raster(ego_state: Union[AgentState, EgoState],
                        coords: List[float],
                        x_range: Tuple[float, float],
                        raster_shape: Tuple[int, int],
                        longitudinal_offset: float = 0.0,
                        polygon_bit_shift: int = 9,
                        color_value: int = 1):
    """
    Construct the map layer of the raster by converting vector map to raster map.

    Args:
        focus_agent (AgentState, EgoState): current state of agent.
        coords (List[float]): coordinates of goal
        x_range (Tuple[float, float]): [m] min and max range from the edges of the grid in x direction.
        raster_shape (Tuple[int, int]): shape of the target raster.
        longitudinal_offset (float, optional): [-0.5, 0.5] longitudinal offset of ego center. Defaults to 0.0.
        polygon_bit_shift (int, optional): bit shift of the polygon used in opencv. Defaults to 9.
        color_value (int, optional): pixel value of goal. Defaults to 1.

    Returns:
        goal raster: the constructed goal raster layer.
    """
    radius = (x_range[1] - x_range[0]) / 2
    goal_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)
    global_transform = np.linalg.inv(virtual_center.as_matrix())
    north_aligned_transform = StateSE2(0, 0, np.pi / 2).as_matrix()
    # map_align_transform = R.from_euler('z', 90, degrees=True).as_matrix().astype(np.float32)
    transform = north_aligned_transform @ global_transform
    trans_coords = StateSE2.from_matrix(transform @ coords.as_matrix())
    length, width, height = 2, 2, 2
    raster_oriented_box = OrientedBox(trans_coords, length, width, height)
    box_bottom_corners = raster_oriented_box.all_corners()
    x_corners = np.asarray(
        [corner.x for corner in box_bottom_corners])  # type: ignore
    y_corners = np.asarray(
        [corner.y for corner in box_bottom_corners])  # type: ignore

    x_corners = (x_corners + radius) / (2 * radius) * raster_shape[0]
    y_corners = (y_corners + radius) / (2 * radius) * raster_shape[1]

    box_2d_coords = np.stack([x_corners, y_corners], axis=1)  # type: ignore
    box_2d_coords = np.expand_dims(box_2d_coords, axis=0)

    # Draw the box as a filled polygon on the raster layer.
    box_2d_coords = (box_2d_coords * 2**polygon_bit_shift).astype(np.int32)

    cv2.fillPoly(
        goal_raster,
        box_2d_coords,
        color=color_value,
        shift=polygon_bit_shift,
        lineType=cv2.LINE_AA)
    goal_raster = np.flip(goal_raster, axis=0)
    goal_raster = np.ascontiguousarray(goal_raster, dtype=np.float32)

    return goal_raster


def get_lane_edge_layer_coords(
        ego_state: Union[AgentState, EgoState],
        map_api: AbstractMap,
        radius: float,
        longitudinal_offset: float = 0.0,
) -> Tuple[List[npt.NDArray[np.float64]], List[str]]:
    """
    Construct the lane edge layer of the raster by converting vector map to raster map, based on the agent.
    :param ego_state (AgentState, EgoState): current state of agent.
    :param map_api: map api
    :param radius: [m] the radius of the square raster map.
    :param longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center
    :return
        object_coords: the list of 2d coordinates which represent the shape of the map.
        lane_ids: the list of ids for the map objects.
    """
    type_dict = {0: 3, 2: 3, 512: 8}
    virtual_center = calc_raster_center(ego_state, radius, longitudinal_offset)
    ego_position = Point2D(virtual_center.x, virtual_center.y)

    nearest_vector_map = map_api.get_proximal_map_objects(
        layers=[SemanticMapLayer.LANE],
        point=ego_position,
        radius=radius,
    )
    geometry = nearest_vector_map[SemanticMapLayer.LANE]

    if len(geometry):
        global_transform = np.linalg.inv(virtual_center.as_matrix())

        # By default the map is right-oriented, this makes it top-oriented.
        map_align_transform = R.from_euler(
            'z', 90, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform
        origin_object_coords = []
        _coords_type = []
        for element in geometry:
            lane_left_edge = [
                line.linestring.coords.xy for line in element.left_boundary
            ]
            lane_right_edge = [
                line.linestring.coords.xy for line in element.right_boundary
            ]
            origin_object_coords = origin_object_coords + lane_left_edge + lane_right_edge
            lane_left_edge_type = [
                type_dict[line.type] for line in element.left_boundary
            ]
            lane_right_edge_type = [
                type_dict[line.type] for line in element.right_boundary
            ]
            _coords_type = _coords_type + lane_left_edge_type + lane_right_edge_type
        non_zero_index = [
            index for (index, value) in enumerate(_coords_type) if value != 8
        ]
        _object_coords = [
            origin_object_coords[index] for index in non_zero_index
        ]
        _coords_type = [_coords_type[index] for index in non_zero_index]

        object_coords: List[npt.NDArray[np.float64]] = [
            np.vstack(coords).T for coords in _object_coords
        ]
        object_coords = [
            (transform @ _cartesian_to_projective_coords(coords).T).T[:, :2]
            for coords in object_coords
        ]
    else:
        object_coords = []
        _coords_type = []

    return object_coords, _coords_type


def get_traj_raster(vcs_xy: np.ndarray,
                    raster_shape: Tuple[int, int],
                    target_pixel_size: float = 0.0,
                    longitudinal_offset: float = 0.0,
                    wp_thickness: int = 3):
    """
    Construct the pred traj layer of the raster by converting vector map to raster map.

    Args:
        vcs_xy (np.ndarray): vcs coordinates
        raster_shape (Tuple[int, int]): A tuple of the shape of the target raster (assumed to be square)
        target_pixel_size (float, optional): target pixel size. Defaults to 0.0.
        longitudinal_offset (float, optional): [-0.5, 0.5] longitudinal offset of ego center.
        wp_thickness (int, optional): pixel size of waypoint. Defaults to 3.

    Returns:
        the constructed pred traj raster
    """
    img = np.zeros(raster_shape).astype(np.uint8)
    for [vcs_x, vcs_y] in vcs_xy:

        map_y_center = int(raster_shape[1] * 0.5)
        map_x_center = int(raster_shape[0] * (0.5 + longitudinal_offset))

        bev_x = -int(vcs_x / target_pixel_size) + map_x_center
        bev_y = -int(vcs_y / target_pixel_size) + map_y_center
        cv2.circle(img, (bev_y, bev_x), wp_thickness, 1, -1)

    return img


def get_roadmap_raster(
        focus_agent: EgoState,
        map_api: AbstractMap,
        map_features: Dict[str, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        raster_shape: Tuple[int, int],
        resolution: float,
        longitudinal_offset: float = 0.0,
) -> npt.NDArray[np.float32]:
    """
    Construct the map layer of the raster by converting vector map to raster map.
    :param focus_agent: agent state representing ego.
    :param map_api: map api.
    :param map_features: name of map features to be drawn and its color for encoding.
    :param x_range: [m] min and max range from the edges of the grid in x direction.
    :param y_range: [m] min and max range from the edges of the grid in y direction.
    :param raster_shape: shape of the target raster.
    :param resolution: [m] pixel size in meters.
    :param longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center.
    :return roadmap_raster: the constructed map raster layer.
    """
    # Assume the raster has a square shape.
    assert (x_range[1] - x_range[0]) == (
        y_range[1] -
        y_range[0]), f'Raster shape is assumed to be square but got width: \
            {y_range[1] - y_range[0]} and height: {x_range[1] - x_range[0]}'

    radius = (x_range[1] - x_range[0]) / 2
    roadmap_raster: npt.NDArray[np.float32] = np.zeros(
        raster_shape, dtype=np.float32)

    for feature_name, feature_color in map_features.items():
        coords, _ = _get_layer_coords(focus_agent, map_api,
                                      SemanticMapLayer[feature_name],
                                      'polygon', radius, longitudinal_offset)
        roadmap_raster = _draw_polygon_image(roadmap_raster, coords, radius,
                                             resolution, feature_color)

    # Flip the agents_raster along the horizontal axis.
    roadmap_raster = np.flip(roadmap_raster, axis=0)
    roadmap_raster = np.ascontiguousarray(roadmap_raster, dtype=np.float32)

    return roadmap_raster
