from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, List

import torch
from dataclasses import dataclass
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import \
    AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    FeatureDataType, to_tensor)


@dataclass
class Tensor(AbstractModelFeature):
    """
    Dataclass that holds general tensor with any possible shape.

    :param data: tensor with any shape, the first dimension could be batch.
    """

    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return True

    def to_device(self, device: torch.device) -> Tensor:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Tensor(data=self.data.to(device=device))

    def to_feature_tensor(self) -> Tensor:
        """Inherited, see superclass."""
        return Tensor(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Tensor:
        """Implemented. See interface."""
        return Tensor(data=data["data"])

    def unpack(self) -> List[Tensor]:
        """Implemented. See interface."""
        return [Tensor(data[None]) for data in self.data]

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)
