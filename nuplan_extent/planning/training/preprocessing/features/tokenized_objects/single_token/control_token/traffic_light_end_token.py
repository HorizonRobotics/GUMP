from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np
from abc import ABC, abstractmethod

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.base_single_token import BaseSingleToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType


class TrafficLightEndToken(BaseSingleToken):
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

    def __init__(self, frame_index=0):
        """
        Initializes the BOSToken instance with a value and a token type.

        The value of the BOS token is '<BOS>' and the token type is 'control'.
        """
        super().__init__(value=VocabularyStateType.TRAFFIC_LIGHT_END_TOKEN.start, token_type='control', frame_index=frame_index)
    
    @property
    def name(self):
        """
        Gets the name or identifier of the BOS token.

        Returns:
            str: A string representing the unique name or identifier of the token.
        """
        return 'traffic_light_end'

    @property
    def abbreviation(self):
        """
        Gets the abbreviation of the token.

        Returns:
            str: A string representing the abbreviation of the token.
        """
        return 'tl_end'

    @property
    def num_tokens(self):
        return 1

    @property
    def token_index_symbol(self):
        return 4
    
    def __str__(self):
        """
        Returns a string representation of the token.

        Overrides the default string representation method to provide a human-readable
        description of the token, including its value and type.
        """
        return f"({self.value},{self.abbreviation})"
    
    def __eq__(self, token) -> bool:
        if not isinstance(token, TrafficLightEndToken):
            return False
        return self.frame_index == token.frame_index

    def __lt__(self, other):
        """
        Compare this instance with another object for sorting purposes, based on 
        'frame_index' and specific object types.

        The comparison logic is as follows:
        - Primarily compare based on 'frame_index'. An object with a lower 'frame_index'
          is considered "less than" an object with a higher 'frame_index'.
        - If 'frame_index' values are equal, the comparison then checks the type of 'other'.
        - If 'other' is an instance of 'BOSToken' or 'TrafficLightToken', it is considered 
          "greater than" this object (thus, this object is "less than" 'other').
        - In all other cases when 'frame_index' values are equal, this object is considered
          "less than" 'other'.

        Parameters:
            other (object): The object to compare to. This method expects 'other' to have
                            a 'frame_index' attribute and to be potentially an instance of
                            'BOSToken' or 'TrafficLightToken'.

        Returns:
            bool: True if this object is considered "less than" 'other', False otherwise.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken

        # Compare based on 'frame_index' first
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index > other.frame_index:
            return False
        else:
            # If 'frame_index' are equal, check the type of 'other'
            return not (isinstance(other, BOSToken) or isinstance(other, TrafficLightToken))

    
    def next_possible_token_type(self):
        """
        Gets the next possible tokens in the sequence.

        Returns:
            List[BaseSingleToken]: A list of the next possible tokens in the sequence.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
        return [AgentToken]

    def get_control_state(self):
        """
        Gets the control state of the token.

        Returns:
            str: A string representing the control state of the token.
        """
        return VocabularyStateType.TRAFFIC_LIGHT_END_TOKEN.start

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
        token = TrafficLightEndToken(
            frame_index=frame_index
        )
        return token