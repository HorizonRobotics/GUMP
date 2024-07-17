from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor


@dataclass
class Token(AbstractModelFeature):
    """
    Dataclass that holds general tensor with any possible shape.

    :param data: tensor with any shape, the first dimension could be batch.
    """

    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        pass

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return True

    def to_device(self, device: torch.device) -> Token:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Token(data=self.data.to(device=device))

    def to_feature_tensor(self) -> Token:
        """Inherited, see superclass."""
        return Token(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Token:
        """Implemented. See interface."""
        return Token(data=data["data"])

    def unpack(self) -> List[Token]:
        """Implemented. See interface."""
        return [Token(data[None]) for data in self.data]

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)

    @classmethod
    def collate(cls, batch: List[AbstractModelFeature]) -> AbstractModelFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        serialized = [sample.serialize() for sample in batch]
        return [cls.deserialize(s) for s in serialized]
