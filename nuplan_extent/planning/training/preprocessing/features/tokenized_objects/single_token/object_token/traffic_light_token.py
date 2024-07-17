from __future__ import annotations

import dataclasses
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from copy import deepcopy

from abc import ABC, abstractmethod

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.base_single_token import BaseSingleToken


class TrafficLightToken(BaseSingleToken):
    """
    Represents a token for a traffic light in a tokenized transformer model.

    This class encapsulates information about a traffic light in a transportation 
    or simulation environment, including its location, current status, and associated 
    lane ID. It extends the BaseSingleToken to leverage common token functionalities 
    while adding specific attributes and methods relevant to traffic light tokens.

    Attributes:
        value (str): Inherited from BaseSingleToken, represents the string 
                     representation of the traffic light token.
        token_type (str): Inherited from BaseSingleToken, categorizes the token 
                          as a 'traffic_light' type.
        location (tuple, optional): A tuple representing the physical coordinates 
                                    (x, y) of the traffic light.
        status (TrafficLightStatusType): An enum value representing the current 
                                         status of the traffic light, which can be 
                                         RED, YELLOW, GREEN, or UNKNOWN.
        lane_id (str): The identifier of the lane that the traffic light controls 
                       or is associated with.

    The status of the traffic light is mapped as follows:
        TrafficLightStatusType.RED: 0,
        TrafficLightStatusType.YELLOW: 1,
        TrafficLightStatusType.GREEN: 2,
        TrafficLightStatusType.UNKNOWN: 3
    """
    def __init__(self, traffic_light_status, tl_index, lane_id, lane_coords, frame_index=0):
        super().__init__(traffic_light_status, frame_index, token_type='object')
        self.traffic_light_status = traffic_light_status
        self.tl_index = tl_index
        self.lane_id = lane_id
        self.lane_coords = deepcopy(lane_coords)
        self.lane_coords[:, 0], self.lane_coords[:, 1] = deepcopy(lane_coords[:, 1]), deepcopy(-lane_coords[:, 0])
    
    @property
    def name(self):
        """
        Gets the name of the token.

        Returns:
            str: A string representing the name of the token.
        """
        return 'traffic_light'

    @property
    def abbreviation(self):
        """
        Gets the abbreviation of the token.

        Returns:
            str: A string representing the abbreviation of the token.
        """
        return 'tl'
    
    @property
    def num_tokens(self):
        return 1
    
    @property
    def token_index_symbol(self):
        5

    def __str__(self):
        """
        Returns a string representation of the token.

        Overrides the default string representation method to provide a human-readable
        description of the token, including its value and type.
        """
        return f'{self.value}{self.abbreviation}{self.tl_index}'
    
    def __eq__(self, token) -> bool:
        if not isinstance(token, TrafficLightToken):
            return False
        return self.tl_index == token.tl_index

    def __lt__(self, other):
        """
        Compare this instance with another object for sorting purposes, based on 
        'frame_index' and, if equal, on specific object types and attributes.

        The detailed comparison logic is as follows:
        - Objects are first compared based on their 'frame_index'. An object with a lower 
          'frame_index' is considered "less than" an object with a higher 'frame_index'.
        - If 'frame_index' values are equal and 'other' is an instance of 'BOSToken', 
          'other' is considered "greater than" this object.
        - If 'frame_index' values are equal and 'other' is an instance of 'TrafficLightToken', 
          the comparison is based on the 'tl_index' attribute.
        - In all other cases when 'frame_index' values are equal, this object is considered
          "less than" 'other'.

        Parameters:
            other (object): The object to compare to. It is assumed that 'other' will have
                            a 'frame_index' attribute and may be an instance of 'BOSToken'
                            or 'TrafficLightToken'.

        Returns:
            bool: True if this object is considered "less than" 'other', False otherwise.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken

        # Primary comparison based on 'frame_index'
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index > other.frame_index:
            return False
        else:
            # Secondary comparison based on the type of 'other'
            if isinstance(other, BOSToken):
                return False
            elif isinstance(other, TrafficLightToken):
                # Tertiary comparison for TrafficLightToken instances based on 'tl_index'
                return self.tl_index < other.tl_index
            else:
                return True
        
    def next_possible_token_type(self):
        """
        Gets the next possible tokens in the sequence.

        Returns:
            List[BaseSingleToken]: A list of the next possible tokens in the sequence.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.traffic_light_end_token import TrafficLightEndToken
        return [TrafficLightToken, TrafficLightEndToken]

    def get_traffic_light_state(self):
        """
        return the traffic light state
        """
        x = self.lane_coords[0, 0]
        y = self.lane_coords[0, 1]
        heading = np.arctan2(self.lane_coords[1, 1] - self.lane_coords[0, 1], self.lane_coords[1, 0] - self.lane_coords[0, 0])
        return x, y, heading, self.traffic_light_status.value, self.tl_index

    def get_traffic_light_idxes(self):
        """
        return the traffic light state
        """
        assert self.index_in_sequence is not None, "index_in_sequence is None"
        return self.index_in_sequence
    