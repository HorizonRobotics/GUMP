import math
from typing import List, Optional, Tuple, Union

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map_objects import (Lane, LaneConnector,
                                                     PolylineMapObject)
from nuplan.common.maps.nuplan_map.utils import estimate_curvature_along_path
from nuplan_extent.common.maps.wod_map.boundary_segment import \
    WodBoundarySegment
from nuplan_extent.common.maps.wod_map.lane_neighbor import WodLaneNeighbor
from nuplan_extent.common.maps.wod_map.utils import extract_discrete_baseline
from shapely.geometry import LineString, Point
from waymo_open_dataset.protos.map_pb2 import MapFeature


def _get_heading(pt1: Point, pt2: Point) -> float:
    """
    Computes the angle two points makes to the x-axis
    :param pt1: origin point
    :param pt2: end point
    :return: [rad] resulting angle
    """
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.atan2(y_diff, x_diff)


class WodLaneCenter(PolylineMapObject):
    def __init__(self,
                 map_feature: MapFeature,
                 distance_for_curvature_estimation: float = 2.0,
                 distance_for_heading_estimation: float = 0.5) -> None:
        """
        Constructor of WodLane
        :param lane_id: unique identifier of the lane
        :param lanes_df: the geopandas GeoDataframe that contains all lanes in the map
        :param lane_connectors_df: the geopandas GeoDataframe that contains all lane connectors in the map
        :param baseline_paths_df: the geopandas GeoDataframe that contains all baselines in the map
        """
        super().__init__(map_feature.id)
        self.map_feature = map_feature
        if len(self.map_feature.lane.
               polyline) == 1:  # handle only one point polyline situation
            init_x, init_y = self.map_feature.lane.polyline[
                0].x, self.map_feature.lane.polyline[0].y
            self._baseline_path = LineString([(init_x, init_y),
                                              (init_x - 1e-3, init_y - 1e-3)])
        else:
            self._baseline_path = LineString(
                [(corr.x, corr.y) for corr in self.map_feature.lane.polyline])
        assert self._baseline_path.length > 0.0, "The length of the path has to be greater than 0!"
        self._lane_type = map_feature.lane.type
        self._speed_limit_mph = float(map_feature.lane.speed_limit_mph)
        self._discrete_path = None
        self._distance_for_curvature_estimation = distance_for_curvature_estimation
        self._distance_for_heading_estimation = distance_for_heading_estimation

    @property
    def incoming_edge_ids(self) -> List[int]:
        return list(self.map_feature.lane.entry_lanes)

    @property
    def outgoing_edge_ids(self) -> List[int]:
        return list(self.map_feature.lane.exit_lanes)

    @property
    def left_neighbors(self) -> List[WodLaneNeighbor]:
        return [
            WodLaneNeighbor(lane_neighbor)
            for lane_neighbor in self.map_feature.lane.left_neighbors
        ]

    @property
    def right_neighbors(self) -> List[WodLaneNeighbor]:
        return [
            WodLaneNeighbor(lane_neighbor)
            for lane_neighbor in self.map_feature.lane.right_neighbors
        ]

    @property
    def left_boundaries(self) -> List[WodBoundarySegment]:
        return [
            WodBoundarySegment(boundary)
            for boundary in self.map_feature.lane.left_boundaries
        ]

    @property
    def right_boundaries(self) -> List[WodBoundarySegment]:
        return [
            WodBoundarySegment(boundary)
            for boundary in self.map_feature.lane.right_boundaries
        ]

    @property
    def type(self) -> str:
        """ Inherited from superclass """
        return self._lane_type

    @property
    def parent(self) -> Union[Lane, LaneConnector]:
        """ Inherited from superclass """

    @property
    def length(self) -> float:
        """ Inherited from superclass """
        return self._baseline_path.length

    @property
    def linestring(self) -> LineString:
        """ Inherited from superclass """
        return self._baseline_path

    @property
    def baseline_path(self) -> PolylineMapObject:
        return self

    @property
    def speed_limit_mps(self) -> Optional[float]:
        """ Inherited from superclass """
        return self._speed_limit_mph

    @property
    def poses(self) -> List[Tuple]:
        discrete_path = self.discrete_path()
        poses = [tuple(state.serialize()) for state in discrete_path]
        return poses

    def discrete_path(self) -> List[StateSE2]:
        """ Inherited from superclass """
        if self._discrete_path is None:
            self._discrete_path = extract_discrete_baseline(
                self._baseline_path)
        return self._discrete_path  # type: ignore

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        """ Inherited from superclass """
        return self._baseline_path.project(Point(point.x,
                                                 point.y))  # type: ignore

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        """ Inherited from superclass """
        arc_length = self.get_nearest_arc_length_from_position(point)
        state1 = self._baseline_path.interpolate(arc_length)
        state2 = self._baseline_path.interpolate(
            arc_length + self._distance_for_heading_estimation)

        if state1 == state2:
            # Handle the case where the queried position (state1) is at the end
            # of the baseline path
            state2 = self._baseline_path.interpolate(
                arc_length - self._distance_for_heading_estimation)
            heading = _get_heading(state2, state1)
        else:
            heading = _get_heading(state1, state2)

        return StateSE2(state1.x, state1.y, heading)

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        """ Inherited from superclass """
        return estimate_curvature_along_path(
            self._baseline_path,  # type: ignore
            arc_length,
            self._distance_for_curvature_estimation)
