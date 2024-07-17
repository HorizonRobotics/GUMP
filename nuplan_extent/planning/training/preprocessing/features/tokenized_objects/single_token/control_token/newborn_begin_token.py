from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np
from abc import ABC, abstractmethod

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.base_single_token import BaseSingleToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType


class NewBornBeginToken(BaseSingleToken):
    """
    Represents the 'New Born Begin' token in a tokenized transformer model.

    This class inherits from BaseSingleToken and provides a specific implementation for 
    the 'New Born Begin' token, typically used to signal the start of a new entity or 
    element in the model. Unlike a generic BOS token, it is specialized for scenarios 
    where new beginnings or initiations are important to demarcate, such as in 
    scenarios dealing with birth or creation events.

    Attributes:
        value (str): The string representation of the token, typically set to a 
                     predefined value indicating the start of a new entity.
        token_type (str): A string categorizing the token as a 'control' type, 
                          distinguishing it from other token types like 'agent'.
    """

    def __init__(self, frame_index=0):
        """
        Initializes the NewBornBeginToken instance with a predefined value and token type.

        The value is typically set to represent the start of a new entity in the model, 
        and the token type is 'control' to signify its role in controlling the flow 
        or structure within the model.
        """
        super().__init__(value=VocabularyStateType.NEWBORN_BEGIN_TOKEN.start, token_type="control", frame_index=frame_index)
    
    @property
    def name(self):
        """
        Gets the name or identifier of the New Born Begin token.

        Returns:
            str: A string representing the unique name or identifier of the token, 
                 which in this case is 'new_born_begin'.
        """
        return 'newborn_begin'

    @property
    def abbreviation(self):
        """
        Gets the abbreviation of the token.

        Returns:
            str: A string representing the abbreviation of the token.
        """
        return 'nb_begin'
    
    @property
    def token_index_symbol(self):
        return 2

    @property
    def num_tokens(self):
        return 1

    def __str__(self):
        """
        Returns a string representation of the New Born Begin token.

        Overrides the default string representation method to provide a human-readable
        description of the token, including its value and type.
        """
        return f"({self.value}, {self.abbreviation})"
    
    def __eq__(self, token) -> bool:
        if not isinstance(token, NewBornBeginToken):
            return False
        return self.frame_index == token.frame_index

    def __lt__(self, other):
        """
        Compare this AgentToken instance with another object for sorting purposes, primarily 
        based on the 'frame_index' attribute. If 'frame_index' values are equal, the comparison
        logic then considers the type and specific attributes of the 'other' object.

        The detailed comparison logic is as follows:
        - Objects are first compared based on their 'frame_index'. An object with a lower 
          'frame_index' is considered "less than" an object with a higher 'frame_index'.
        - If 'frame_index' values are equal, and 'other' is an instance of 'AgentToken' and
          has 'NEWBORN_STATE' as its 'agent_type', then 'other' is considered "greater than" 
          this object.
        - If 'frame_index' values are equal, and 'other' is an instance of 'BOSToken', 
          then 'other' is considered "greater than" this object.
        - In all other cases when 'frame_index' values are equal, this object is not 
          considered "less than" 'other'.

        Parameters:
            other (object): The object to compare to. This method expects that 'other'
                            will have a 'frame_index' attribute and may be an instance of 
                            either 'AgentToken' or 'BOSToken'.

        Returns:
            bool: True if this object is considered "less than" 'other', False otherwise.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
        # Compare based on 'frame_index' first
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index > other.frame_index:
            return False
        else:
            # Additional comparisons when 'frame_index' values are equal
            if isinstance(other, AgentToken):
                # Compare based on 'agent_type' if 'other' is an 'AgentToken'
                return other.agent_type == PositionalStateType.NEWBORN_STATE
            else:
                # In all other cases, this object is not "less than" 'other'
                return False

    
    def next_possible_token_type(self):
        """
        Gets the next possible tokens in the sequence.

        Returns:
            List[BaseSingleToken]: A list of the next possible tokens in the sequence.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
        return [AgentToken, BOSToken]


    def get_control_state(self):
        """
        Gets the control state of the token.

        Returns:
            str: A string representing the control state of the token.
        """
        return VocabularyStateType.NEWBORN_BEGIN_TOKEN.start

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
        token = NewBornBeginToken(
            frame_index=frame_index
        )
        return token