from nuplan.common.maps.abstract_map_objects import AbstractMapObject
from shapely.geometry import Point
from waymo_open_dataset.protos.map_pb2 import \
    TrafficSignalLaneState


class WodTrafficSignalLaneState(AbstractMapObject):
    def __init__(self, lane_state: TrafficSignalLaneState) -> None:
        super().__init__(lane_state.lane)
        self._position = Point(lane_state.stop_point.x,
                               lane_state.stop_point.y)
        self._state_type = lane_state.state

    @property
    def point(self) -> Point:
        """return stop sign position

        Returns:
            Point: stop sign point position
        """
        return self._position

    @property
    def type(self) -> str:
        return self._state_type
