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
class WaymoTensor(AbstractModelFeature):
    """
    Dataclass that holds general tensor with any possible shape.

    :param data: tensor with any shape, the first dimension could be batch.
    """

    data: FeatureDataType
    scenario_id: FeatureDataType
    agent_id: int
    local_to_global_transform: FeatureDataType    

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return True

    def to_device(self, device: torch.device) -> WaymoTensor:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return WaymoTensor(data=self.data.to(device=device), scenario_id=self.scenario_id, agent_id=self.agent_id, local_to_global_transform=self.local_to_global_transform)

    def to_feature_tensor(self) -> WaymoTensor:
        """Inherited, see superclass."""
        return WaymoTensor(data=to_tensor(self.data), scenario_id=self.scenario_id, agent_id=self.agent_id, local_to_global_transform=self.local_to_global_transform)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> WaymoTensor:
        """Implemented. See interface."""
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        if "data_dict" in data:
            data = data["data_dict"]
        return WaymoTensor(
            data=data["ego_states"],
            scenario_id=data["scenario_id"],
            agent_id=data["ego_id"],
            local_to_global_transform=data["local_to_global_transform"],)

    def unpack(self) -> List[WaymoTensor]:
        """Implemented. See interface."""
        unpacked = []
        for data, scenario_id, agent_id, local_to_global_transform in zip(self.data, self.scenario_id, self.agent_id, self.local_to_global_transform):
            unpacked.append(
                WaymoTensor(
                    data=data[None],
                    scenario_id=scenario_id,
                    agent_id=agent_id,
                    local_to_global_transform=local_to_global_transform,
                ))

        return unpacked

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)
