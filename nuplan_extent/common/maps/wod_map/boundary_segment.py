from waymo_open_dataset.protos.map_pb2 import BoundarySegment


class WodBoundarySegment():
    def __init__(self, boundary_segment: BoundarySegment):
        self._lane_start_index = boundary_segment.lane_start_index
        self._lane_end_index = boundary_segment.lane_end_index
        self._boundary_feature_id = boundary_segment.boundary_feature_id
        self._boundary_type = boundary_segment.boundary_type

    @property
    def lane_start_index(self) -> str:
        return self._lane_start_index

    @property
    def lane_end_index(self) -> str:
        return self._lane_end_index

    @property
    def boundary_feature_id(self) -> str:
        return self._boundary_feature_id

    @property
    def boundary_type(self) -> str:
        return self._boundary_type
