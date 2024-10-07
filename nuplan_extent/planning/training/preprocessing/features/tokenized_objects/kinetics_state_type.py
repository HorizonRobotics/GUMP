from __future__ import annotations

import sys
from enum import Enum
from typing import Set, Tuple
import numpy as np


class KineticsVocabularyStateType(Enum):
    """Enum of classification types for TrackedObject with integer ranges."""

    BLANK = (0, 0), 'blank', 0
    CONTROL = (1, 33 * 65), 'control', 1            # 64 * 64 = 4096,  
    
    @property
    def control_acc_range(self) -> Tuple[float, float]:
        return (-6, 6)
    
    @property
    def control_steering_range(self) -> Tuple[float, float]:
        return (-1.0, 1.0)
    
    @property
    def ncontrol_acc(self) -> int:
        return 32
    
    @property
    def ncontrol_steering(self) -> int:
        return 64
    
    @property
    def control_acc_step(self) -> float:
        return (self.control_acc_range[1] - self.control_acc_range[0]) / self.ncontrol_acc
        
    @property
    def control_steering_step(self) -> float:
        return (self.control_steering_range[1] - self.control_steering_range[0]) / self.ncontrol_steering

    @property
    def num_agent_attributes(self) -> str:
        return 1

    @property
    def vocal_size(self) -> int:
        return self.CONTROL.end+1

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