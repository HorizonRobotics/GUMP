from typing import List, Tuple

from nuplan.common.maps.abstract_map_objects import PolygonMapObject
from shapely.geometry import Polygon
from waymo_open_dataset.protos.map_pb2 import MapFeature


class WodSpeedBump(PolygonMapObject):
    def __init__(self, map_feature: MapFeature) -> None:
        super().__init__(map_feature.id)
        self.map_feature = map_feature
        self._polygon = Polygon(
            [corr.x, corr.y] for corr in map_feature.speed_bump.polygon)
        self._poses = [(corr.x, corr.y, 0)
                       for corr in map_feature.speed_bump.polygon]

    @property
    def polygon(self) -> Polygon:
        """ Inherited from superclass """
        return self._polygon

    @property
    def type(self) -> str:
        return "TYPE_CROSSWALK"

    @property
    def poses(self) -> List[Tuple]:
        return self._poses
