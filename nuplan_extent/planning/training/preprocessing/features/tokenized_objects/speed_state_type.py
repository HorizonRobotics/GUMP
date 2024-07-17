from __future__ import annotations

import sys
from enum import Enum
from typing import Set, Tuple
import numpy as np


class SpeedVocabularyStateType(Enum):
    """Enum of classification types for TrackedObject with integer ranges."""

    EGO = (0, 0), 'ego', 0
    AGENTS = (1, 521), 'agents', 1
    CLASS_TYPE = (522, 524), 'class_type', 2
    SPEED_TOKEN = (525, 9071), 'speed_token', 3
    BOS_TOKEN = (9072, 9072), 'bos_token', 4
    NEWBORN_BEGIN_TOKEN = (9073, 9073), 'newborn_bos_token', 5
    PAD_TOKEN = (9074, 9074), 'pad_token', 6

    @property
    def dx_range(self):
        return [-1, 8.0]

    @property
    def dy_range(self):
        return [-0.75, 0.75]
    
    @property
    def dh_range(self):
        return [-0.75, 0.75]
    
    @property
    def nx(self):
        return 37
    
    @property
    def ny(self):
        return 11

    @property
    def nh(self):
        return 21

    @property
    def dx_step(self):
        return (self.dx_range[1] - self.dx_range[0]) / (self.nx - 1)
    
    @property
    def dy_step(self):
        return (self.dy_range[1] - self.dy_range[0]) / (self.ny - 1)

    @property
    def dh_step(self):
        return (self.dh_range[1] - self.dh_range[0]) / (self.nh - 1)

    @property
    def num_agent_attributes(self) -> str:
        return 1

    @property
    def vocal_size(self) -> int:
        return self.PAD_TOKEN.end+1

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