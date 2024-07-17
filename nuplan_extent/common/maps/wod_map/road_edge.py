from typing import List, Tuple

from nuplan.common.maps.abstract_map_objects import AbstractMapObject
from shapely.geometry import LineString
from waymo_open_dataset.protos.map_pb2 import MapFeature


class WodRoadEdge(AbstractMapObject):
    def __init__(self, map_feature: MapFeature) -> None:
        super().__init__(map_feature.id)
        self.map_feature = map_feature
        self._edge_type = map_feature.road_edge.type
        if len(map_feature.road_edge.
               polyline) == 1:  # handle only one point polyline situation
            init_x, init_y = map_feature.road_edge.polyline[
                0].x, map_feature.road_edge.polyline[0].y
            self._polyline = LineString([(init_x, init_y),
                                         (init_x - 1e-3, init_y - 1e-3)])
        else:
            self._polyline = LineString(
                [(corr.x, corr.y) for corr in map_feature.road_edge.polyline])
        self._poses = [(corr.x, corr.y, 0)
                       for corr in map_feature.road_edge.polyline]

    @property
    def type(self) -> str:
        return self._edge_type

    @property
    def linestring(self) -> LineString:
        return self._polyline

    @property
    def poses(self) -> List[Tuple]:
        return self._poses

    @property
    def baseline_path(self) -> AbstractMapObject:
        return self
