from __future__ import annotations

from enum import IntEnum


class WodLaneStateType(IntEnum):
    LANE_STATE_UNKNOWN = 0
    #  States for traffic signals with arrows.
    LANE_STATE_ARROW_STOP = 1
    LANE_STATE_ARROW_CAUTION = 2
    LANE_STATE_ARROW_GO = 3

    # Standard round traffic signals.
    LANE_STATE_STOP = 4
    LANE_STATE_CAUTION = 5
    LANE_STATE_GO = 6

    # Flashing light signals.
    LANE_STATE_FLASHING_STOP = 7
    LANE_STATE_FLASHING_CAUTION = 8


class WodRoadLineType(IntEnum):
    TYPE_UNKNOWN = 0
    TYPE_BROKEN_SINGLE_WHITE = 1
    TYPE_SOLID_SINGLE_WHITE = 2
    TYPE_SOLID_DOUBLE_WHITE = 3
    TYPE_BROKEN_SINGLE_YELLOW = 4
    TYPE_BROKEN_DOUBLE_YELLOW = 5
    TYPE_SOLID_SINGLE_YELLOW = 6
    TYPE_SOLID_DOUBLE_YELLOW = 7
    TYPE_PASSING_DOUBLE_YELLOW = 8


class WodRoadEdgeType(IntEnum):
    TYPE_UNKNOWN = 0
    TYPE_ROAD_EDGE_BOUNDARY = 1
    TYPE_ROAD_EDGE_MEDIAN = 2


class WodLaneCenterType(IntEnum):
    TYPE_UNDEFINED = 0
    TYPE_FREEWAY = 1
    TYPE_SURFACE_STREET = 2
    TYPE_BIKE_LANE = 3
