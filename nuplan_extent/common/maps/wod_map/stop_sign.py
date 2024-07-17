from typing import List

from nuplan.common.maps.abstract_map_objects import AbstractMapObject
from shapely.geometry import Point
from waymo_open_dataset.protos.map_pb2 import MapFeature


class WodStopSign(AbstractMapObject):
    def __init__(self, map_feature: MapFeature) -> None:
        super().__init__(map_feature.id)
        self.map_feature = map_feature
        self._position = Point(map_feature.stop_sign.position.x,
                               map_feature.stop_sign.position.y)
        self._block_lanes = map_feature.stop_sign.lane

    @property
    def point(self) -> Point:
        """return stop sign position

        Returns:
            Point: stop sign point position
        """
        return self._position

    @property
    def block_lanes(self) -> List[str]:
        """ return block lanes id

        Return:
            List[str]: block lanes id
        """
        return self._block_lanes
