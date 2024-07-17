from __future__ import annotations

from functools import cached_property
from typing import Optional

import torch
from dataclasses import dataclass
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    FeatureDataType)
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


@dataclass
class FlattenFeature(Tensor):
    """
    Dataclass that holds general flattened 1d feature.

    :param data: either a [num_batches, num_states, num_features] or [num_states, num_features]
                 representing the 1d feature.
    """

    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        array_dims = self.num_dimensions

        if (array_dims != 2) and (array_dims != 3):
            raise RuntimeError(
                f'Invalid 1d feature. Expected 2 or 3 dims, got {array_dims}.')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return len(self.data) > 0 and self.data.shape[
            -2] > 0 and self.data.shape[-1] == self.state_size

    @property
    def state_size(self) -> int:
        """
        Size of each SE2 state of the trajectory.
        """
        return self.data.shape[-1]

    @property
    def num_of_iterations(self) -> int:
        """
        :return: number of states in a trajectory
        """
        return int(self.data.shape[-2])

    @property
    def num_batches(self) -> Optional[int]:
        """
        :return: number of batches in the trajectory, None if trajectory does not have batch dimension
        """
        return None if self.num_dimensions <= 2 else self.data.shape[0]

    def state_at_index(self, index: int) -> FeatureDataType:
        """
        Query state at index along trajectory horizon
        :param index: along horizon
        :return: state corresponding to the index along trajectory horizon
        @raise in case index is not within valid range: 0 < index <= num_of_iterations
        """
        assert 0 <= index < self.num_of_iterations, f"Index is out of bounds! 0 <= {index} < {self.num_of_iterations}!"
        return self.data[..., index, :]

    def detach(self) -> FlattenFeature:
        validate_type(self.data, torch.Tensor)
        return FlattenFeature(data=self.data.detach())
