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
class AttributeTable(AbstractModelFeature):
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

    def to_device(self, device: torch.device) -> Tensor:
        """Implemented. See interface."""
        # validate_type(self.data, torch.Tensor)
        return AttributeTable(data=self.data)

    def to_feature_tensor(self) -> Tensor:
        """Inherited, see superclass."""
        return AttributeTable(data=self.data)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Tensor:
        """Implemented. See interface."""
        return AttributeTable(data=data["data"])

    def unpack(self) -> List[Tensor]:
        """Implemented. See interface."""
        return [AttributeTable(data) for data in self.data]

    @classmethod
    def collate(cls, batch: List[AbstractModelFeature]
                ) -> AbstractModelFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        serialized = [sample.serialize() for sample in batch]
        return [cls.deserialize(s) for s in serialized]
