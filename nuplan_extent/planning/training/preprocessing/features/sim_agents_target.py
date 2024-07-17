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
class SimAgentsTarget(AbstractModelFeature):
    """
    Dataclass that holds general tensor with any possible shape.

    :param data: tensor with any shape, the first dimension could be batch.
    """

    pkl_path: str
    scenario_id: str
    agent_idx: str
    local_to_global_transform: torch.Tensor


    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return True

    def to_device(self, device: torch.device) -> SimAgentsTarget:
        """Implemented. See interface."""
        return SimAgentsTarget(self.pkl_path, self.scenario_id, self.agent_idx, self.local_to_global_transform)

    def to_feature_tensor(self) -> SimAgentsTarget:
        """Inherited, see superclass."""
        return SimAgentsTarget(self.pkl_path, self.scenario_id, self.agent_idx, self.local_to_global_transform)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SimAgentsTarget:
        """Implemented. See interface."""
        return SimAgentsTarget(pkl_path=data['pkl_path'], scenario_id=data['scenario_id'], agent_idx=data['agent_idx'], local_to_global_transform=to_tensor(data['local_to_global_transform']))

    def unpack(self) -> List[SimAgentsTarget]:
        """Implemented. See interface."""
        unpacked = []
        for pkl_path_i, scenario_id_i, agent_idx_i, local_to_global_transform in zip(self.pkl_path, self.scenario_id, self.agent_idx, self.local_to_global_transform):
            unpacked.append(
                SimAgentsTarget(pkl_path=pkl_path_i, scenario_id=scenario_id_i, agent_idx=agent_idx_i, local_to_global_transform=local_to_global_transform)
                )

        return unpacked

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return 0
