from __future__ import annotations

import sys
from enum import Enum
from typing import Set, Tuple
import numpy as np


class VocabularyStateType(Enum):
    """Enum of classification types for TrackedObject with integer ranges."""

    EGO = (0, 0), 'ego', 0
    AGENTS = (1, 320), 'agents', 1
    TRAFFIC_LIGHT = (321, 324), 'traffic_light', 2
    X = (325, 1324), 'x', 3 # 1000, 0.2, -100, 100
    Y = (1325, 2324), 'y', 4 # 1000, 0.2, -100, 100
    HEADING = (2325, 2524), 'heading', 5, # 200, np.pi/100, -np.pi, np.pi
    VX = (2525, 2724), 'vx', 6 # 200, 0.25, -25, 25
    VY = (2725, 2924), 'vy', 7 # 200, 0.25, -25, 25
    WIDTH = (2925, 2939), 'width', 8 # 15
    LENGTH = (2940, 2969), 'length', 9 #30
    BOS_TOKEN = (2970, 2970), 'bos_token', 10
    NEWBORN_BEGIN_TOKEN = (2971, 2971), 'newborn_begin_token', 11
    TRAFFIC_LIGHT_END_TOKEN = (2972, 2972), 'traffic_light_end_token', 12
    PAD_TOKEN = (2973, 2973), 'pad_token', 13
    LOCATION_TOKEN = (-1000 * 1000 * 200 * 15 * 30 * 200 * 200, -1), 'location_token', 14

    @property
    def num_agent_attributes(self) -> str:
        return 14

    @property
    def vocal_size(self) -> int:
        return self.PAD_TOKEN.end+1

    @property
    def x_range(self) -> Tuple[float, float]:
        return (-100, 100)

    @property
    def y_range(self) -> Tuple[float, float]:
        return (-100, 100)

    @property
    def heading_range(self) -> Tuple[float, float]:
        return (-np.pi, np.pi)

    @property
    def vx_range(self) -> Tuple[float, float]:
        return (-25, 25)

    @property
    def vy_range(self) -> Tuple[float, float]:
        return (-25, 25)

    @property
    def width_range(self) -> Tuple[float, float]:
        return (0, 7)

    @property
    def length_range(self) -> Tuple[float, float]:
        return (0, 15)

    @property
    def x_step(self) -> float:
        return (self.x_range[1] - self.x_range[0])/self.nx

    @property
    def y_step(self) -> float:
        return (self.y_range[1] - self.y_range[0])/self.ny

    @property
    def heading_step(self) -> float:
        return (self.heading_range[1] - self.heading_range[0])/self.nh

    @property
    def vx_step(self) -> float:
        return (self.vx_range[1] - self.vx_range[0])/self.nvx

    @property
    def vy_step(self) -> float:
        return (self.vy_range[1] - self.vy_range[0])/self.nvy

    @property
    def width_step(self) -> float:
        return (self.width_range[1] - self.width_range[0])/self.nw

    @property
    def length_step(self) -> float:
        return (self.length_range[1] - self.length_range[0])/self.nl

    @property
    def nx(self) -> int:
        return self.X.end - self.X.start + 1

    @property
    def ny(self) -> int:
        return self.Y.end - self.Y.start + 1

    @property
    def nh(self) -> int:
        return self.HEADING.end - self.HEADING.start + 1

    @property
    def nvx(self) -> int:
        return self.VX.end - self.VX.start + 1

    @property
    def nvy(self) -> int:
        return self.VY.end - self.VY.start + 1

    @property
    def nw(self) -> int:
        return self.WIDTH.end - self.WIDTH.start + 1

    @property
    def nl(self) -> int:
        return self.LENGTH.end - self.LENGTH.start + 1

    @property
    def ntl(self) -> int:
        return self.TRAFFIC_LIGHT.end - self.TRAFFIC_LIGHT.start + 1

    @property
    def nagents(self) -> int:
        return self.AGENTS.end - self.AGENTS.start + 1

    @property
    def start(self) -> int:
        """Get the start value of the range."""
        return self.value[0][0]

    @property
    def end(self) -> int:
        """Get the end value of the range."""
        return self.value[0][1]

    @property
    def index(self) -> int:
        """Get the index of the range."""
        return self.value[2]

    @classmethod
    def index_to_state(cls, index: int) -> VocabularyStateType:
        """Convert an index to its corresponding VocabularyStateType."""
        for state in cls:
            if state.index == index:
                return state
        raise ValueError(f"No VocabularyStateType found for index {index}")

    def __contains__(self, value) -> bool:
        """Check if a number is within the range of this enum member."""
        if isinstance(value, VocabularyStateType):
            return self.start <= value.start <= self.end
        elif isinstance(value, int):
            return self.start <= value <= self.end
        else:
            raise TypeError("Unsupported type for containment check")

    def get_sampling_mask(self):
        sampling_mask = np.zeros(self.vocal_size)
        sampling_mask[self.start:self.end+1] = 1
        return sampling_mask > 0


class PositionalStateType(Enum):
    """Enum of classification types for TrackedObject."""

    TRAFFIC_LIGHT_STATE = 1, 'traffic_light_state'
    NEWBORN_STATE = 2, 'newborn_state'
    NORMAL_STATE = 3, 'normal_state'

    def __contains__(self, pos_state: PositionalStateType) -> bool:
        """Check if a number is within the range of this enum member."""
        return pos_state.value == self.value