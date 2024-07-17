import os
import pickle
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

import shapely.geometry as geom
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.maps_datatypes import (RasterLayer, RasterMap,
                                               SemanticMapLayer)
from nuplan_extent.common.maps.wod_map.baselane import WodLaneCenter
from nuplan_extent.common.maps.wod_map.crosswalk import WodCrosswalk
from nuplan_extent.common.maps.wod_map.lane_state import \
    WodTrafficSignalLaneState
from nuplan_extent.common.maps.wod_map.road_edge import WodRoadEdge
from nuplan_extent.common.maps.wod_map.road_line import WodRoadLine
from nuplan_extent.common.maps.wod_map.speed_bump import WodSpeedBump
from nuplan_extent.common.maps.wod_map.stop_sign import WodStopSign


class WodMap(AbstractMap):
    def __init__(self, data_root: str, scenario_id: str) -> None:
        """Initialize WodMap

        Args:
            data_root (str): waymo data root
            scenario_id (str): id of scenario
        """

        self.data_root = data_root
        self.scenario_id = scenario_id
        self._file_path = os.path.join(data_root, scenario_id) + ".pkl"
        self.map_objects = None
        self._map_range = [1e6, -1e6, 1e6, -1e6]

        self._map_object_getter: Dict[SemanticMapLayer,
                                      Callable[[str], MapObject]] = {
                                          SemanticMapLayer.LANE:
                                          self._get_lane_center,
                                          SemanticMapLayer.STOP_SIGN:
                                          self._get_stop_sign,
                                          SemanticMapLayer.CROSSWALK:
                                          self._get_crosswalk,
                                          SemanticMapLayer.SPEED_BUMP:
                                          self._get_speed_bump,
                                          SemanticMapLayer.BASELINE_PATHS:
                                          self._get_road_line,
                                          SemanticMapLayer.EXTENDED_PUDO:
                                          self._get_road_edge,
                                          SemanticMapLayer.STOP_LINE:
                                          self._get_lane_state
                                      }

    def initialize_all_layers(self) -> None:
        """
        Load all layers to vector map
        :param: None
        :return: None
        """
        self.load_map()

    def load_map(self):
        with open(self._file_path, 'rb') as f:
            data = pickle.load(f)
        self.map_objects: Dict[SemanticMapLayer,
                               Dict[int, MapObject]] = defaultdict(dict)
        for map_feature in data.map_features:
            if map_feature.WhichOneof('feature_data') == "lane":
                self.map_objects[SemanticMapLayer.LANE][
                    map_feature.id] = WodLaneCenter(map_feature)
                self.update_map_range(
                    self.map_objects[SemanticMapLayer.LANE][
                        map_feature.id].linestring.bounds)
            elif map_feature.WhichOneof('feature_data') == "road_edge":
                self.map_objects[SemanticMapLayer.EXTENDED_PUDO][
                    map_feature.id] = WodRoadEdge(map_feature)
            elif map_feature.WhichOneof('feature_data') == "road_line":
                self.map_objects[SemanticMapLayer.BASELINE_PATHS][
                    map_feature.id] = WodRoadLine(map_feature)
            elif map_feature.WhichOneof('feature_data') == "stop_sign":
                self.map_objects[SemanticMapLayer.STOP_SIGN][
                    map_feature.id] = WodStopSign(map_feature)
            elif map_feature.WhichOneof('feature_data') == "crosswalk":
                self.map_objects[SemanticMapLayer.CROSSWALK][
                    map_feature.id] = WodCrosswalk(map_feature)
            elif map_feature.WhichOneof('feature_data') == "speed_bump":
                self.map_objects[SemanticMapLayer.SPEED_BUMP][
                    map_feature.id] = WodSpeedBump(map_feature)
        current_dynamic_map_state = data.dynamic_map_states[data.
                                                            current_time_index]
        for lane_state in current_dynamic_map_state.lane_states:
            self.map_objects[SemanticMapLayer.STOP_LINE][
                lane_state.lane] = WodTrafficSignalLaneState(lane_state)

    def _preload_map(func):
        def wrap(*args, **kwargs):
            if args[0].map_objects is None:
                args[0].load_map()
            return func(*args, **kwargs)

        return wrap

    def update_map_range(self, new_range: List[int]):
        """Update map range

        Args:
            new_range (List[int]): minx, miny, maxx, maxy, which order is different with self._map_range
        """
        minx, miny, maxx, maxy = new_range
        raw_min_x, raw_max_x, raw_min_y, raw_max_y = self._map_range
        if minx < raw_min_x:
            raw_min_x = minx
        if miny < raw_min_y:
            raw_min_y = miny
        if maxx > raw_max_x:
            raw_max_x = maxx
        if maxy > raw_max_y:
            raw_max_y = maxy
        self._map_range = [raw_min_x, raw_max_x, raw_min_y, raw_max_y]

    @property
    def map_range(self) -> List[int]:
        """return map range

        Returns:
            List[int]: min_x, max_x, min_y, max_y
        """
        return self._map_range

    @property
    def map_name(self) -> str:
        """ Inherited, see superclass. """
        return self.scenario_id

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """ Inherited, see superclass. """
        return list(self._map_object_getter.keys())

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        """ Inherited, see superclass. """

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        """ Inherited, see superclass. """

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        """ Inherited, see superclass. """

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        """ Inherited, see superclass. """

    def get_all_map_objects(self, point: Point2D,
                            layer: SemanticMapLayer) -> List[MapObject]:
        """ Inherited, see superclass. """

    def get_distance_to_nearest_map_object(self, point: Point2D, layer: SemanticMapLayer) \
            -> Tuple[Optional[str], Optional[float]]:
        """ Inherited from superclass """

    def get_distance_to_nearest_raster_layer(self, point: Point2D,
                                             layer: SemanticMapLayer) -> float:
        """Inherited from superclass"""
        raise NotImplementedError

    def get_distances_matrix_to_nearest_map_object(
            self, points: List[Point2D],
            layer: SemanticMapLayer) -> Optional[npt.NDArray[np.float64]]:
        """Inherited from superclass"""
        raise NotImplementedError

    @_preload_map
    def get_map_object(self, object_id: str,
                       layer: SemanticMapLayer) -> MapObject:
        """ Inherited, see superclass. """
        try:
            return self._map_object_getter[layer](object_id)
        except KeyError:
            raise ValueError(
                f"Object representation for layer: {layer.name} is unavailable"
            )

    def get_one_map_object(self, point: Point2D,
                           layer: SemanticMapLayer) -> Optional[MapObject]:
        """ Inherited, see superclass. """

    @_preload_map
    def get_proximal_map_objects(self,
                                 point: Point2D,
                                 radius: float,
                                 layers: List[SemanticMapLayer],
                                 mode: str = 'intersect'
                                 ) -> Dict[SemanticMapLayer, List[MapObject]]:
        """ Inherited, see superclass. """
        x = point.x
        y = point.y
        if isinstance(radius, float):
            patch = geom.box(x - radius, y - radius, x + radius, y + radius)
        else:
            patch = geom.box(x - radius[0], y - radius[1], x + radius[2],
                             y + radius[3])
        object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)
        for layer in layers:
            object_map[layer] = self._get_proximity_map_object(
                patch, layer, mode=mode)
        return object_map

    def _get_proximity_map_object(self,
                                  patch: geom.Polygon,
                                  layer: SemanticMapLayer = None,
                                  mode: str = 'intersect') -> List[MapObject]:
        """
        Get all the record token that intersects or within a particular rectangular patch.
        :param box_coords: The rectangular patch coordinates (x_min, y_min, x_max, y_max).
        :param layer: layer that we want to retrieve in a particular patch.
            By default will always look for all non geometric layers.
        :param mode: "intersect" will return all non geometric records that intersects the patch,
            "within" will return all non geometric records that are within the patch.
        :return: Dictionary of layer_name - tokens pairs.
        """
        proximal_objects = []
        for map_object_id in self.map_objects[layer].keys():
            if (hasattr(self.map_objects[layer][map_object_id], 'linestring') and
                    self.map_objects[layer][map_object_id].linestring.intersects(patch)) or \
                    (hasattr(self.map_objects[layer][map_object_id], 'polygon') and
                     self.map_objects[layer][map_object_id].polygon.intersects(patch)) or \
                    (hasattr(self.map_objects[layer][map_object_id], 'point') and
                     self.map_objects[layer][map_object_id].point.intersects(patch)):
                proximal_objects.append(self.map_objects[layer][map_object_id])
        return proximal_objects

    @_preload_map
    def _get_lane_center(self, lane_id: str) -> WodLaneCenter:
        """return WodLaneCenter for given lane_id

        Args:
            lane_id (str): map object id of this lanecenter

        Returns:
            WodLaneCenter: WodLaneCenter
        """
        return self.map_objects[SemanticMapLayer.LANE][lane_id]

    @_preload_map
    def _get_speed_bump(self, speed_bump_id: str) -> WodSpeedBump:
        """return WodSpeedBump for given speed_bump_id

        Args:
            speed_bump_id (str): map object id of this SpeedBump

        Returns:
            WodSpeedBump: WodSpeedBump
        """
        return self.map_objects[SemanticMapLayer.SPEED_BUMP][speed_bump_id]

    @_preload_map
    def _get_stop_sign(self, stop_sign_id: str) -> WodStopSign:
        """return WodStopSign for given stop stop_sign_id

        Args:
            stop_line_id (str): map object id of this Stopsign

        Returns:
            WodStopSign: WodStopSign
        """
        return self.map_objects[SemanticMapLayer.STOP_SIGN][stop_sign_id]

    @_preload_map
    def _get_crosswalk(self, crosswalk_id: str) -> WodCrosswalk:
        """return WodCrosswalk for given crosswalk_id

        Args:
            crosswalk_id (str): map object id of this crosswalk

        Returns:
            WodCrosswalk: WodCrosswalk
        """
        return self.map_objects[SemanticMapLayer.CROSSWALK][crosswalk_id]

    @_preload_map
    def _get_road_line(self, road_line_id: str) -> WodRoadLine:
        """return WodRoadLine for given road_line_id

        Args:
            road_line_id (str): map object id of this road line

        Returns:
            WodRoadLine: WodRoadLine
        """
        return self.map_objects[SemanticMapLayer.BASELINE_PATHS][road_line_id]

    @_preload_map
    def _get_road_edge(self, road_edge_id: str) -> WodRoadEdge:
        """return WodRoadEdge for given road_edge_id

        Args:
            road_edge_id (str): map object id of this road edge

        Returns:
            WodRoadEdge: WodRoadEdge
        """
        return self.map_objects[SemanticMapLayer.EXTENDED_PUDO][road_edge_id]

    @_preload_map
    def _get_lane_state(self, lane_state_id: str) -> WodTrafficSignalLaneState:
        """return WodTrafficSignalLaneState for given lane_state_id

        Args:
            lane_state_id (str): id of this lane

        Returns:
            WodTrafficSignalLaneState: WodTrafficSignalLaneState
        """
        return self.map_objects[SemanticMapLayer.STOP_LINE][lane_state_id]
