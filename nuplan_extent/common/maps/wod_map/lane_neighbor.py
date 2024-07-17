from typing import List

from nuplan_extent.common.maps.wod_map.boundary_segment import \
    WodBoundarySegment
from waymo_open_dataset.protos.map_pb2 import LaneNeighbor


class WodLaneNeighbor():
    def __init__(self, lane_neighbor: LaneNeighbor):
        self._neighbor_id = lane_neighbor.feature_id
        self._self_start_index = lane_neighbor.self_start_index
        self._self_end_index = lane_neighbor.self_end_index
        self._neighbor_start_index = lane_neighbor.neighbor_start_index
        self._neighbor_end_index = lane_neighbor.neighbor_end_index
        self._boundaries = [
            WodBoundarySegment(boundary)
            for boundary in lane_neighbor.boundaries
        ]

    @property
    def neighbor_id(self) -> str:
        return self._neighbor_id

    @property
    def self_start_index(self) -> str:
        return self._self_start_index

    @property
    def self_end_index(self) -> str:
        return self._self_end_index

    @property
    def neighbor_start_index(self) -> str:
        return self._neighbor_start_index

    @property
    def neighbor_end_index(self) -> str:
        return self._neighbor_end_index

    @property
    def boundaries(self) -> List[WodBoundarySegment]:
        return self._boundaries
