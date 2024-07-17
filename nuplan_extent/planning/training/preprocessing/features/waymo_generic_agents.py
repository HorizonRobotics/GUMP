from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents


@dataclass
class WaymoGenericAgents(GenericAgents):


    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        if not all([len(self.ego) == len(agent) for agent in self.agents.values()]):
            raise AssertionError("Batch size inconsistent across features!")

        if len(self.ego) == 0:
            raise AssertionError("Batch size has to be > 0!")

        if self.ego[0].ndim != 2:
            raise AssertionError(
                "Ego feature samples does not conform to feature dimensions! "
                f"Got ndim: {self.ego[0].ndim} , expected 2 [num_frames, 7]"
            )

        if 'EGO' in self.agents.keys():
            raise AssertionError("EGO not a valid agents feature type!")
        # for feature_name in self.agents.keys():
        #     if feature_name not in TrackedObjectType._member_names_:
        #         raise ValueError(f"Object representation for layer: {feature_name} is unavailable!")

        # for agent in self.agents.values():
        #     if not isinstance(agent[0], Dict) and agent[0].ndim != 3:
        #         raise AssertionError(
        #             "Agent feature samples does not conform to feature dimensions! "
        #             f"Got ndim: {agent[0].ndim} , "
        #             f"expected 3 [num_frames, num_agents, 8]"
        #         )

        # for sample_idx in range(len(self.ego)):
        #     if int(self.ego[sample_idx].shape[0]) != self.num_frames or not all(
        #         [int(agent[sample_idx].shape[0]) == self.num_frames for agent in self.agents.values()
        #          if not isinstance(agent[sample_idx], Dict)]
        #     ):
        #         raise AssertionError("Agent feature samples have different number of frames!")


    def to_feature_tensor(self) -> WaymoGenericAgents:
        """Implemented. See interface."""
        return WaymoGenericAgents(
            ego=[to_tensor(sample) for sample in self.ego],
            agents={agent_name: [to_tensor(sample) for sample in agent] for agent_name, agent in self.agents.items()},
        )

    @cached_property
    def is_valid(self) -> bool:
        return True

    @classmethod
    def collate(cls, batch: List[WaymoGenericAgents]) -> WaymoGenericAgents:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        agents: Dict[str, List[FeatureDataType]] = defaultdict(list)
        for sample in batch:
            for agent_name, agent in sample.agents.items():
                agents[agent_name] += [agent[0]]
        return WaymoGenericAgents(ego=[item.ego[0] for item in batch], agents=agents)


    def to_device(self, device: torch.device) -> WaymoGenericAgents:
        """Implemented. See interface."""
        return WaymoGenericAgents(
            ego=[to_tensor(ego).to(device=device) for ego in self.ego],
            agents={
                agent_name: [to_tensor(sample).to(device=device) for sample in agent]
                for agent_name, agent in self.agents.items()
            },
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> WaymoGenericAgents:
        """Implemented. See interface."""
        return WaymoGenericAgents(ego=data["ego"], agents=data["agents"])

    def unpack(self) -> List[WaymoGenericAgents]:
        """Implemented. See interface."""
        return [
            WaymoGenericAgents(
                ego=[self.ego[sample_idx]],
                agents={agent_name: [agent[sample_idx]] for agent_name, agent in self.agents.items()},
            )
            for sample_idx in range(self.batch_size)
        ]

    # @cached_property
    # def is_valid(self) -> bool:
    #     """Inherited, see superclass."""
    #     return (
    #         len(self.ego) > 0
    #         and all([len(agent) > 0 for agent in self.agents.values()])
    #         and all([len(self.ego) == len(agent) for agent in self.agents.values()])
    #         and len(self.ego[0]) > 0
    #         and all([len(agent[0]) > 0 for agent in self.agents.values()])
    #         and all([len(self.ego[0]) == len(agent[0]) > 0 for agent in self.agents.values()])
    #         and self.ego[0].shape[-1] == self.ego_state_dim()
    #         and all([agent[0].shape[-1] == self.agents_states_dim() for agent in self.agents.values()])
    #     )

    @staticmethod
    def ego_state_dim() -> int:
        """
        :return: ego state dimension.
        """
        return WaymoEgoInternalIndex.dim()

    @staticmethod
    def agents_states_dim() -> int:
        """
        :return: agent state dimension.
        """
        return WaymoAgentInternalIndex.dim()

    @property
    def ego_feature_dim(self) -> int:
        """
        :return: ego feature dimension.
        """
        return GenericAgents.ego_state_dim() * self.num_frames


    def get_ego_agents_center_in_sample(self, sample_idx: int) -> FeatureDataType:
        """
        Return ego center in the given sample index.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: 2>. (x, y) positions of the ego's center at sample index.
        """
        self._validate_ego_query(sample_idx)
        return self.get_present_ego_in_sample(sample_idx)[: WaymoEgoInternalIndex.y() + 1]

    def agent_processing_by_type(
        self, processing_function: Callable[[str, int], FeatureDataType], sample_idx: int
    ) -> FeatureDataType:
        """
        Apply agent processing functions across all agent types in features for given batch sample.
        :param processing_function: function to apply across agent types
        :param sample_idx: the batch index of interest.
        :return Processed agent feature across agent types.
        """
        agents: List[FeatureDataType] = []
        for agent_type in self.agents.keys():
            if self.has_agents(agent_type, sample_idx):
                agents.append(processing_function(agent_type, sample_idx))
        if len(agents) == 0:
            if isinstance(self.ego[sample_idx], torch.Tensor):
                return torch.empty(
                    (0, len(self.agents.keys()) * self.num_frames * WaymoAgentInternalIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                    device=self.ego[sample_idx].device,
                )
            else:
                return np.empty(
                    (0, len(self.agents.keys()) * self.num_frames * WaymoAgentInternalIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                )
        elif isinstance(agents[0], torch.Tensor):
            return torch.cat(agents, dim=0)
        else:
            return np.concatenate(agents, axis=0)



    def get_flatten_agents_features_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Flatten agents' features of specified type by stacking the agents' states along the num_frame dimension
        <np.ndarray: num_frames, num_agents, 8>] -> <np.ndarray: num_agents, num_frames x 8>].

        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, num_frames x 8>] agent feature.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.num_agents_in_sample(agent_type, sample_idx) == 0:
            if isinstance(self.ego[sample_idx], torch.Tensor):
                return torch.empty(
                    (0, self.num_frames * WaymoAgentInternalIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                    device=self.ego[sample_idx].device,
                )
            else:
                return np.empty(
                    (0, self.num_frames * WaymoAgentInternalIndex.dim()),
                    dtype=self.ego[sample_idx].dtype,
                )

        data = self.agents[agent_type][sample_idx]
        axes = (1, 0) if isinstance(data, torch.Tensor) else (1, 0, 2)
        return data.transpose(*axes).reshape(data.shape[1], -1)


    def get_agents_centers_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's centers in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents, 2>. (x, y) positions of the agents' centers at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, : WaymoAgentInternalIndex.y() + 1]


    def get_agents_length_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's length at the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents>. lengths of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty.
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, WaymoAgentInternalIndex.length()]

    def get_agents_width_by_type_in_sample(self, agent_type: str, sample_idx: int) -> FeatureDataType:
        """
        Returns all agents of specified type's width in the given sample index.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :return: <FeatureDataType: num_agents>. width of all the agents at the sample index.
        :raise RuntimeError if feature at given sample index is empty
        """
        self._validate_agent_query(agent_type, sample_idx)
        if self.agents[agent_type][sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")
        return self.get_present_agents_by_type_in_sample(agent_type, sample_idx)[:, WaymoAgentInternalIndex.width()]


class WaymoEgoInternalIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
      in the Ego Trajectory Tensors.

    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the ego x position.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the ego y position.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the ego heading.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the ego x velocity.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the ego y velocity.
        :return: index
        """
        return 4

    @staticmethod
    def ax() -> int:
        """
        The dimension corresponding to the ego x acceleration.
        :return: index
        """
        return 5

    @staticmethod
    def ay() -> int:
        """
        The dimension corresponding to the ego y acceleration.
        :return: index
        """
        return 6
    
    @staticmethod
    def width() -> int:
        """
        The dimension corresponding to the ego width.
        :return: index
        """
        return 7

    @staticmethod
    def length() -> int:
        """
        The dimension corresponding to the ego length.
        :return: index
        """
        return 8   

    @staticmethod
    def z() -> int:
        """
        The dimension corresponding to the ego y position.
        :return: index
        """
        return 9

    @staticmethod
    def type() -> int:
        """
        The dimension corresponding to the ego type.
        :return: index
        """
        return 10

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the EgoInternal buffer.
        :return: number of features.
        """
        return 11
    

class WaymoAgentInternalIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
      in the tensors used to compute the final Agent Feature.


    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def track_token() -> int:
        """
        The dimension corresponding to the track_token for the agent.
        :return: index
        """
        return 0

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the agent.
        :return: index
        """
        return 2

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the agent.
        :return: index
        """
        return 3

    @staticmethod
    def width() -> int:
        """
        The dimension corresponding to the width of the agent.
        :return: index
        """
        return 4

    @staticmethod
    def length() -> int:
        """
        The dimension corresponding to the length of the agent.
        :return: index
        """
        return 5

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x position of the agent.
        :return: index
        """
        return 6

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y position of the agent.
        :return: index
        """
        return 7
    
    @staticmethod
    def z() -> int:
        """
        The dimension corresponding to the z position of the agent.
        :return: index
        """
        return 8
    

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the AgentsInternal buffer.
        :return: number of features.
        """
        return 9
