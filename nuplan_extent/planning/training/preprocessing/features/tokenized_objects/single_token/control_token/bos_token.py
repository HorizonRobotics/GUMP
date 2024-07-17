from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np
from abc import ABC, abstractmethod

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.base_single_token import BaseSingleToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType


class BOSToken(BaseSingleToken):
    """
    Class for the Beginning of Sequence (BOS) token.

    This class inherits from BaseSingleToken and provides a template for the BOS token
    in a tokenized transformer model. The BOS token is used to indicate the start of a 
    sequence of tokens in the model.

    Attributes:
        value (str): The string representation of the token.
        token_type (str): A string that categorizes the token into a type, 
                          e.g., 'control', 'agent', etc.
    """

    def __init__(self, frame_index=0, num_traffic_light=0, num_agents=0):
        """
        Initializes the BOSToken instance with a value and a token type.

        The value of the BOS token is '<BOS>' and the token type is 'control'.
        """
        super().__init__(value=VocabularyStateType.BOS_TOKEN.start, token_type='control', frame_index=frame_index)
        self.num_traffic_light = num_traffic_light
        self.num_agents = num_agents
        self.traffic_light_tokens = None

    @property
    def name(self):
        """
        Gets the name or identifier of the BOS token.

        Returns:
            str: A string representing the unique name or identifier of the token.
        """
        return 'bos'

    @property
    def token_index_symbol(self):
        return 6

    @property
    def abbreviation(self):
        """
        Gets the abbreviation of the token.

        Returns:
            str: A string representing the abbreviation of the token.
        """
        return 'bos'

    @property
    def num_tokens(self):
        return 1

    def __str__(self):
        """
        Returns a string representation of the token.

        Overrides the default string representation method to provide a human-readable
        description of the token, including its value and type.
        """
        return f"({self.value},{self.abbreviation})"
    
    def __eq__(self, token) -> bool:
        if not isinstance(token, BOSToken):
            return False
        return self.frame_index == token.frame_index
    
    def __lt__(self, other):
        """
        Compare this AgentToken instance with another object for sorting, based solely on the 
        'frame_index' attribute. This implementation defines an ordering where objects with 
        a smaller 'frame_index' are considered "less than" those with a larger 'frame_index'.

        Parameters:
            other (AgentToken): The AgentToken object to compare to. It is assumed that
                                this object also has a 'frame_index' attribute.

        Returns:
            bool: True if this object's 'frame_index' is less than the 'frame_index' of 'other',
                  False otherwise.
        """
        # Comparison is based only on 'frame_index'
        return self.frame_index < other.frame_index
            
    
    def next_possible_token_type(self):
        """
        Gets the next possible tokens in the sequence.

        Returns:
            List[BaseSingleToken]: A list of the next possible tokens in the sequence.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken
        return [TrafficLightToken]

    def get_control_state(self):
        """
        Gets the control state of the token.

        Returns:
            int: The control state of the token.
        """
        return VocabularyStateType.BOS_TOKEN.start

    def get_control_idxes(self, output=False):
        """
        Gets the control index of the token.

        Returns:
            int: The control index of the token.
        """
        assert self.index_in_sequence is not None, "index_in_sequence is None"
        return self.index_in_sequence
        
    def to_numpy_array(self):
        return np.array(
            [self.token_index_symbol,
            *[0.0]*26,
            self.frame_index,
            0.0]
        )

    @staticmethod
    def from_numpy(array):
        frame_index = int(array[27])
        token = BOSToken(
            frame_index=frame_index
        )
        return token