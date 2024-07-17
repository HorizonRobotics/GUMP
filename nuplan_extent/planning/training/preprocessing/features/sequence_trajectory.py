from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


@dataclass
class SequenceTrajectory(Trajectory):
    """
    Dataclass that holds trajectory signals produced from the model or from the dataset for supervision.

    :param data: either a [num_batches, num_mode, num_states, 3] or [num_mode, num_states, 3] representing the trajectory
                 where se2_state is [x, y, heading] with units [meters, meters, radians].
    """
    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        array_dims = self.num_dimensions
        state_size = self.data.shape[-1]

        if (array_dims != 3) and (array_dims != 4):
            raise RuntimeError(f'Invalid trajectory array. Expected 2 or 3 dims, got {array_dims}.')

        if state_size != self.state_size():
            raise RuntimeError(
                f'Invalid trajectory array. Expected {self.state_size()} variables per state, got {state_size}.'
            )

    def to_device(self, device: torch.device) -> Trajectory:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return SequenceTrajectory(data=self.data.to(device=device))

    def to_feature_tensor(self) -> Trajectory:
        """Inherited, see superclass."""
        return SequenceTrajectory(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Trajectory:
        """Implemented. See interface."""
        return SequenceTrajectory(data=data["data"])

    def unpack(self) -> List[Trajectory]:
        """Implemented. See interface."""
        return [SequenceTrajectory(data[None]) for data in self.data]

    @property
    def num_batches(self) -> Optional[int]:
        """
        :return: number of batches in the trajectory, None if trajectory does not have batch dimension
        """
        return None if self.num_dimensions <= 2 else self.data.shape[0]
        
    @property
    def num_timesteps(self) -> Optional[int]:
        """Number of timesteps in the feature."""
        return self.data.shape[0] if len(self.data.shape) < 4 else self.data.shape[1]

    @classmethod
    def append_to_trajectory(cls, trajectory: Trajectory, new_state: torch.Tensor) -> Trajectory:
        """
        Extend trajectory with a new state, in this case we require that both trajectory and new_state has dimension
        of 3, that means that they both have batch dimension
        :param trajectory: to be extended
        :param new_state: state with which trajectory should be extended
        :return: extended trajectory
        """
        assert trajectory.num_dimensions == 3, f"Trajectory dimension {trajectory.num_dimensions} != 3!"
        assert len(
            new_state.shape) == 3, f"New state dimension {new_state.shape} != 3!"

        if new_state.shape[0] != trajectory.data.shape[0]:
            raise RuntimeError(
                f"Not compatible shapes {new_state.shape} != {trajectory.data.shape}!")

        if new_state.shape[-1] != trajectory.data.shape[-1]:
            raise RuntimeError(
                f"Not compatible shapes {new_state.shape} != {trajectory.data.shape}!")

        return Trajectory(data=torch.cat((trajectory.data, new_state.clone()), dim=2))
