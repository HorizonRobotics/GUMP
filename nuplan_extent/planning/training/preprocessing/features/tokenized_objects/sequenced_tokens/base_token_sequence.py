from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from abc import ABC, abstractmethod

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.traffic_light_end_token import TrafficLightEndToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.newborn_begin_token import NewBornBeginToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType


class BaseTokenSequence(ABC):
    """
    Abstract base class for a structured sequence of tokens in a tokenized transformer model.

    This class provides the foundation for creating a sequence of tokens that follows a 
    specific order: "BOS, traffic light, traffic light end, agent(s), newborn begin, agent(s), BOS". 
    It ensures any subclass implementation maintains this sequence structure for proper model processing.

    Subclasses must implement specific methods to add tokens and validate the sequence according to this pattern.
    """

    def __init__(self):
        self.tokens = []  # List to store the sequence of tokens

    @abstractmethod
    def add_token(self, token: Union[TrafficLightToken, AgentToken, TrafficLightEndToken, BOSToken, NewBornBeginToken]):
        """
        Adds a token to the token sequence while enforcing the specific sequence structure.

        Args:
            token (Union[TrafficLightToken, AgentToken, TrafficLightEndToken, BOSToken, NewBornBeginToken]): 
                The token to be added to the sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_sequence(self) -> bool:
        """
        Validates whether the current token sequence adheres to the specified structure.

        Returns:
            bool: True if the sequence is valid as per the defined structure, False otherwise.
        """
        raise NotImplementedError

    def get_sequence(self) -> List[Union[TrafficLightToken, AgentToken, TrafficLightEndToken, BOSToken, NewBornBeginToken]]:
        """
        Retrieves the current token sequence.

        Returns:
            List[Union[TrafficLightToken, AgentToken, TrafficLightEndToken, BOSToken, NewBornBeginToken]]: 
                The current sequence of tokens.
        """
        return self.tokens
    
    def __len__(self):
        """
        Returns the length of the token sequence.
        """
        return len(self.tokens)
    
    # def to_device(self, device: torch.device):
    #     """Implemented. See interface."""
    #     validate_type(self.data, torch.Tensor)
    #     return Control(data=self.data.to(device=device))

    # def to_feature_tensor(self):
    #     """Inherited, see superclass."""
    #     return Control(data=to_tensor(self.data))

    # @classmethod
    # def deserialize(cls, data: Dict[str, Any]):
    #     """Implemented. See interface."""
    #     return Control(data=data["data"])

    # def unpack(self):
    #     """Implemented. See interface."""
    #     return [Control(data[None]) for data in self.data]

