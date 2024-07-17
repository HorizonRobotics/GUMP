from __future__ import annotations

import dataclasses
import logging
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from copy import deepcopy
from numba import jit
from abc import ABC, abstractmethod

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.base_single_token import BaseSingleToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType

X_START, X_END, X_RANGE, X_STEP = VocabularyStateType.X.start, VocabularyStateType.X.end, VocabularyStateType.X.x_range, VocabularyStateType.X.x_step
Y_START, Y_END, Y_RANGE, Y_STEP = VocabularyStateType.Y.start, VocabularyStateType.Y.end, VocabularyStateType.Y.y_range, VocabularyStateType.Y.y_step
HEADING_START, HEADING_END, HEADING_RANGE, HEADING_STEP = VocabularyStateType.HEADING.start, VocabularyStateType.HEADING.end, VocabularyStateType.HEADING.heading_range, VocabularyStateType.HEADING.heading_step
WIDTH_START, WIDTH_END, WIDTH_RANGE, WIDTH_STEP = VocabularyStateType.WIDTH.start, VocabularyStateType.WIDTH.end, VocabularyStateType.WIDTH.width_range, VocabularyStateType.WIDTH.width_step
LENGTH_START, LENGTH_END, LENGTH_RANGE, LENGTH_STEP = VocabularyStateType.LENGTH.start, VocabularyStateType.LENGTH.end, VocabularyStateType.LENGTH.length_range, VocabularyStateType.LENGTH.length_step


class AgentToken(BaseSingleToken):
    """
    Represents a token for a dynamic agent in a tokenized transformer model.

    This class is designed to hold information about various types of dynamic agents 
    such as self-driving vehicles, standard vehicles, pedestrians, cyclists, or other 
    movable objects in a simulated or real-world environment. It extends the 
    BaseSingleToken to leverage common functionalities while adding specific attributes 
    and methods relevant to dynamic agents.

    Attributes:
        value (str): Inherited from BaseSingleToken, represents the string 
                     representation of the agent token.
        token_type (str): Inherited from BaseSingleToken, categorizes the token 
                          as an 'agent' type.
        location (tuple): A tuple representing the physical coordinates (x, y, heading) 
                          of the agent.
        past_trajectory (list of tuples): A list of tuples representing the agent's 
                                          historical trajectory.
        future_trajectory (list of tuples): A list of tuples representing the agent's 
                                            predicted future trajectory.
        track_id (str): A unique identifier for tracking the agent.
        size (tuple): A tuple representing the dimensions of the agent (width, length, height).
        speed (float): The current speed of the agent.
        acceleration (float): The current acceleration of the agent.
        yaw_rate (float): The rate of change of the agent's heading (rotational velocity).

    These attributes capture the state and dynamics of agents, crucial for understanding 
    and predicting their behavior in the model.
    """
    def __init__(self, value, track_id, token_id, type_idx, 
        x, y, z, heading, x_1sec, y_1sec, heading_1sec, 
        x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, 
        width, length, past_trajectory, future_trajectory, vx, vy, frame_index=0, is_ego=False, raw_id=None, detokenize_initial=False):
        super().__init__(value, frame_index=frame_index, token_type='object')
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        self.vx = vx
        self.vy = vy
        self.v = np.sqrt(vx**2 + vy**2)

        self.x_1sec = x_1sec
        self.y_1sec = y_1sec
        self.heading_1sec = heading_1sec

        self.x_1_5sec = x_1_5sec
        self.y_1_5sec = y_1_5sec
        self.heading_1_5sec = heading_1_5sec

        self.x_2sec = x_2sec
        self.y_2sec = y_2sec
        self.heading_2sec = heading_2sec

        self._prev_x = [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        self._prev_y = [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        self._prev_heading = [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]

        self._before_precondition_x = x
        self._before_precondition_y = y
        self._before_precondition_heading = heading
        self.width = width
        self.length = length
        self.past_trajectory = np.stack(past_trajectory)
        self.future_trajectory = np.stack(future_trajectory)
        self.track_id = int(track_id)
        self.token_id = int(token_id)
        self.type_idx = int(type_idx)
        self.agent_type = PositionalStateType.NORMAL_STATE
        self.predicted_future_trajectory = None
        self.pred_future_prob = None
        self.is_ego = is_ego
        self.raw_id = raw_id
        self.detokenize_initial = detokenize_initial
    
    @property
    def token_index_symbol(self):
        return 1

    @staticmethod
    def from_ego_array(ego_array, frame_index=0, dataset='nuplan', raw_id=None):
        """
        Creates an AgentToken instance from an ego array.

        Args:
            ego_array (np.array): A numpy array representing the ego vehicle.

        Returns:
            AgentToken: An AgentToken instance representing the ego vehicle.
        """
        current_ego_array = ego_array[frame_index]
        if frame_index+1 < len(ego_array):
            next_ego_array = ego_array[frame_index+1]
            x_1sec, y_1sec, heading_1sec = next_ego_array[:3]
        else:
            x_1sec, y_1sec, heading_1sec = np.nan, np.nan, np.nan

        if frame_index+2 < len(ego_array):
            next_ego_array = ego_array[frame_index+2]
            x_1_5sec, y_1_5sec, heading_1_5sec = next_ego_array[:3]
        else:
            x_1_5sec, y_1_5sec, heading_1_5sec = np.nan, np.nan, np.nan


        if frame_index+3 < len(ego_array):
            next_ego_array = ego_array[frame_index+3]
            x_2sec, y_2sec, heading_2sec = next_ego_array[:3]
        else:
            x_2sec, y_2sec, heading_2sec = np.nan, np.nan, np.nan

        if dataset == 'nuplan':
            x, y, heading, vx, vy, ax, ay = current_ego_array
            width, length = 2.297, 5.176
            track_id = 0
            token_id = 0
            type_idx = 0
            z = 0
        elif dataset == 'waymo':
            # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
            x, y, heading, vx, vy, ax, ay, width, length, z, type_idx = current_ego_array
            track_id = 0
            token_id = 0
            type_idx = type_idx - 1
        else:
            raise NotImplementedError

        value = AgentToken.tokenize(x, y, heading, x_1sec, y_1sec, heading_1sec, 
            x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec,
            width, length, track_id)
        past_trajectory = AgentToken.extract_past_trajectory(ego_array, frame_index)
        future_trajectory = AgentToken.extract_future_trajectory(ego_array, frame_index)
        return AgentToken(value, track_id, token_id, type_idx, x, y, z, heading, x_1sec, y_1sec, heading_1sec, 
            x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, width, length, past_trajectory, future_trajectory, vx, vy, frame_index=frame_index, is_ego=True, raw_id=raw_id)

    @staticmethod
    def from_agent_array(agent_array, frame_index=0, type_idx=0, dataset='nuplan', raw_id=None):
        """
        Creates an AgentToken instance from an agent array.

        Args:
            agent_array (np.array): A numpy array representing the agent.

        Returns:
            AgentToken: An AgentToken instance representing the agent.
        """
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        if frame_index+1 < len(agent_array):
            next_agent_array = agent_array[frame_index+1]
            x_1sec, y_1sec, heading_1sec = next_agent_array[6], next_agent_array[7], next_agent_array[3]
        else:
            x_1sec, y_1sec, heading_1sec = np.nan, np.nan, np.nan

        if frame_index+2 < len(agent_array):
            next_agent_array = agent_array[frame_index+2]
            x_1_5sec, y_1_5sec, heading_1_5sec = next_agent_array[6], next_agent_array[7], next_agent_array[3]
        else:
            x_1_5sec, y_1_5sec, heading_1_5sec = np.nan, np.nan, np.nan

        if frame_index+3 < len(agent_array):
            next_agent_array = agent_array[frame_index+3]
            x_2sec, y_2sec, heading_2sec = next_agent_array[6], next_agent_array[7], next_agent_array[3]
        else:
            x_2sec, y_2sec, heading_2sec = np.nan, np.nan, np.nan

        if dataset == 'nuplan':
            track_token, vx, vy, heading, width, length, x, y = agent_array[frame_index]
            z = 0
        else:
            track_token, vx, vy, heading, width, length, x, y, z = agent_array[frame_index]

        track_token = int(track_token)
        value = AgentToken.tokenize(x, y, heading, x_1sec, y_1sec, heading_1sec, 
             x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, width, length, track_token)
        agent_array = agent_array[:, [6, 7, 3]] # convert to x, y, heading
        past_trajectory = AgentToken.extract_past_trajectory(agent_array, frame_index)
        future_trajectory = AgentToken.extract_future_trajectory(agent_array, frame_index)  
        token_id = track_token  # we are not using token_id for now, use track_token as a placeholder
        return AgentToken(value, track_token, token_id, type_idx, x, y, z, heading, x_1sec, y_1sec, heading_1sec, 
             x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, width, length, past_trajectory, future_trajectory, vx, vy, frame_index=frame_index, is_ego=False, raw_id=raw_id)      

    @staticmethod
    def from_unrolled_nextstep(next_state, prev_agent_token, frame_index=0):
        dx, dy, heading = next_state
        prev_x, prev_y, prev_heading = prev_agent_token.x, prev_agent_token.y, prev_agent_token.heading
        x, y = prev_x + dx, prev_y + dy
        
        # Assuming np.nan values are placeholders, initialize these once if they're constant
        nan_values = np.nan, np.nan, np.nan
        
        width, length, track_token, type_idx = (
            prev_agent_token.width, prev_agent_token.length, 
            prev_agent_token.track_id, prev_agent_token.type_idx
        )

        # Utilize a method or constructor that directly accepts the structured data
        value = AgentToken.tokenize(x, y, heading, *nan_values*3, width, length, track_token)

        token_id, is_ego, raw_id = prev_agent_token.token_id, prev_agent_token.is_ego, prev_agent_token.raw_id

        # Simplify updates to _prev_x, _prev_y, _prev_heading
        update_prev = lambda prev_list, new_value: [prev_list[1], prev_list[2], new_value]
        past_trajectory, future_trajectory = deepcopy(prev_agent_token.past_trajectory), deepcopy(prev_agent_token.future_trajectory)
        next_agent = AgentToken(
            value, track_token, token_id, type_idx, x, y, prev_agent_token.z, heading, 
            *nan_values*3, width, length, past_trajectory, future_trajectory, 0.0, 0.0, frame_index, is_ego, raw_id
        )

        # Assigning previous conditions more succinctly
        next_agent._before_precondition_x, next_agent._before_precondition_y, next_agent._before_precondition_heading = prev_x, prev_y, prev_heading
        
        # Optimized updates
        next_agent._prev_x = update_prev(prev_agent_token._prev_x, [prev_agent_token.x_1sec, prev_agent_token.x_1_5sec, prev_agent_token.x_2sec])
        next_agent._prev_y = update_prev(prev_agent_token._prev_y, [prev_agent_token.y_1sec, prev_agent_token.y_1_5sec, prev_agent_token.y_2sec])
        next_agent._prev_heading = update_prev(prev_agent_token._prev_heading, [prev_agent_token.heading_1sec, prev_agent_token.heading_1_5sec, prev_agent_token.heading_2sec])

        return next_agent

    @staticmethod
    @jit(nopython=True)
    def tokenize(x, y, heading, x_1sec, y_1sec, heading_1sec, x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, width, length, track_id):
        # Ensure the primary values are within bounds
        def calculate_token(value, value_range, start, step):
            """
            Helper function to calculate the token value or return -1 if the input value is None.
            """
            if np.isnan(value):
                return -1
            value = np.maximum(value_range[0], np.minimum(value_range[1], value))
            return start + int(round((value - value_range[0]) / step))
        # Use the helper function to calculate tokens or set to -1 if value is None
        x_token = calculate_token(x, X_RANGE, X_START, X_STEP)
        y_token = calculate_token(y, Y_RANGE, Y_START, Y_STEP)
        heading_token = calculate_token(heading, HEADING_RANGE, HEADING_START, HEADING_STEP)
        
        x_token_1sec = calculate_token(x_1sec, X_RANGE, X_START, X_STEP)
        y_token_1sec = calculate_token(y_1sec, Y_RANGE, Y_START, Y_STEP)
        heading_token_1sec = calculate_token(heading_1sec, HEADING_RANGE, HEADING_START, HEADING_STEP)

        x_token_1_5sec = calculate_token(x_1_5sec, X_RANGE, X_START, X_STEP)
        y_token_1_5sec = calculate_token(y_1_5sec, Y_RANGE, Y_START, Y_STEP)
        heading_token_1_5sec = calculate_token(heading_1_5sec, HEADING_RANGE, HEADING_START, HEADING_STEP)

        x_token_2sec = calculate_token(x_2sec, X_RANGE, X_START, X_STEP)
        y_token_2sec = calculate_token(y_2sec, Y_RANGE, Y_START, Y_STEP)
        heading_token_2sec = calculate_token(heading_2sec, HEADING_RANGE, HEADING_START, HEADING_STEP)

        width_token = calculate_token(width, WIDTH_RANGE, WIDTH_START, WIDTH_STEP)
        length_token = calculate_token(length, LENGTH_RANGE, LENGTH_START, LENGTH_STEP)

        return (width_token, length_token, x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec, x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec, track_id)

    @staticmethod
    @jit(nopython=True)
    def detokenize(x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec, x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec, width_token, length_token):
        # bound the values
        x_token = np.maximum(X_START, np.minimum(X_END, x_token))
        y_token = np.maximum(Y_START, np.minimum(Y_END, y_token))
        heading_token = np.maximum(HEADING_START, np.minimum(HEADING_END, heading_token))
        x_token_1sec = np.maximum(X_START, np.minimum(X_END, x_token_1sec))
        y_token_1sec = np.maximum(Y_START, np.minimum(Y_END, y_token_1sec))
        heading_token_1sec = np.maximum(HEADING_START, np.minimum(HEADING_END, heading_token_1sec))
        x_token_1_5sec = np.maximum(X_START, np.minimum(X_END, x_token_1_5sec))
        y_token_1_5sec = np.maximum(Y_START, np.minimum(Y_END, y_token_1_5sec))
        heading_token_1_5sec = np.maximum(HEADING_START, np.minimum(HEADING_END, heading_token_1_5sec))
        x_token_2sec = np.maximum(X_START, np.minimum(X_END, x_token_2sec))
        y_token_2sec = np.maximum(Y_START, np.minimum(Y_END, y_token_2sec))
        heading_token_2sec = np.maximum(HEADING_START, np.minimum(HEADING_END, heading_token_2sec))
        width_token = np.maximum(WIDTH_START, np.minimum(WIDTH_END, width_token))
        length_token = np.maximum(LENGTH_START, np.minimum(LENGTH_END, length_token))

        # convert tokens back to values
        x = X_RANGE[0] + (x_token - X_START) * X_STEP
        y = Y_RANGE[0] + (y_token - Y_START) * Y_STEP
        heading = HEADING_RANGE[0] + (heading_token - HEADING_START) * HEADING_STEP
        x_1sec = X_RANGE[0] + (x_token_1sec - X_START) * X_STEP
        y_1sec = Y_RANGE[0] + (y_token_1sec - Y_START) * Y_STEP
        heading_1sec = HEADING_RANGE[0] + (heading_token_1sec - HEADING_START) * HEADING_STEP       
        x_1_5sec = X_RANGE[0] + (x_token_1_5sec - X_START) * X_STEP
        y_1_5sec = Y_RANGE[0] + (y_token_1_5sec - Y_START) * Y_STEP
        heading_1_5sec = HEADING_RANGE[0] + (heading_token_1_5sec - HEADING_START) * HEADING_STEP
        x_2sec = X_RANGE[0] + (x_token_2sec - X_START) * X_STEP
        y_2sec = Y_RANGE[0] + (y_token_2sec - Y_START) * Y_STEP
        heading_2sec = HEADING_RANGE[0] + (heading_token_2sec - HEADING_START) * HEADING_STEP
        width = WIDTH_RANGE[0] + (width_token - WIDTH_START) * WIDTH_STEP
        length = LENGTH_RANGE[0] + (length_token - LENGTH_START) * LENGTH_STEP
        return x, y, heading, x_1sec, y_1sec, heading_1sec, x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, width, length

    @staticmethod
    def extract_past_trajectory(ego_array, frame_index):
        """
        Extracts the past trajectory of the ego vehicle.

        Args:
            ego_array (np.array): A numpy array representing the ego vehicle.
            frame_index (int): The frame index of the ego vehicle.

        Returns:
            List[tuple]: A list of tuples representing the past trajectory of the 
                         ego vehicle.
        """
        max_past_trajectory_step = 6
        past_trajectory = []
        cx, cy, ch = ego_array[frame_index][:3]
        for i in range(frame_index-1, 0, -1):
            if len(past_trajectory) == max_past_trajectory_step:
                break
            px, py, ph = ego_array[i][:3]
            # we only care about the relative position, and absolute heading
            past_trajectory.append((px-cx, py-cy, ph))
        while len(past_trajectory) < max_past_trajectory_step:
            past_trajectory.append((np.nan, np.nan, np.nan))
        return past_trajectory

    @staticmethod
    def extract_future_trajectory(ego_array, frame_index):
        """
        Extracts the future trajectory of the ego vehicle.

        Args:
            ego_array (np.array): A numpy array representing the ego vehicle.
            frame_index (int): The frame index of the ego vehicle.

        Returns:
            List[tuple]: A list of tuples representing the future trajectory of the 
                         ego vehicle.
        """
        max_future_trajectory_step = 16
        future_trajectory = []
        cx, cy, ch = ego_array[frame_index][:3]
        for i in range(frame_index+1, len(ego_array)):
            if len(future_trajectory) == max_future_trajectory_step:
                break
            px, py, ph = ego_array[i][:3]
            # we only care about the relative position, and absolute heading
            future_trajectory.append((px-cx, py-cy, ph))
        while len(future_trajectory) < max_future_trajectory_step:
            future_trajectory.append((np.nan, np.nan, np.nan))
        return future_trajectory

    def tozeros(self):
        """
        Converts the token to a zero token.

        Returns:
            AgentToken: A zero token representing the agent.
        """
        self.value = list(AgentToken.tokenize(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, self.track_id))
        self.x, self.y, self.heading, self.width, self.length, self.type_idx = 0, 0, 0, 0, 0, 0

    @staticmethod
    def within_range(current_agent_array, cls_idx):
        if len(current_agent_array) == 9:
            track_token, vx, vy, heading, width, length, x, y, z = current_agent_array
        else:
            track_token, vx, vy, heading, width, length, x, y = current_agent_array
            z = 0
        if cls_idx == 1:
            if x > 40 or x < -40:
                return False
            if y > 40 or y < -40:
                return False
        else:
            if x > 100 or x < -100:
                return False
            if y > 100 or y < -100:
                return False
        # else:
        #     if x > X_RANGE[1] or x < X_RANGE[0]:
        #         return False
        #     if y > Y_RANGE[1] or y < Y_RANGE[0]:
        #         return False
        return True

    def _within_range(self):
        delta = 4.0
        if self.x >= (X_RANGE[1] - delta) or self.x <= (X_RANGE[0] + delta):
            return False
        if self.y > (Y_RANGE[1] - delta) or self.y < (Y_RANGE[0] + delta):
            return False
        return True

    def sampled_fake_distribution(self):
        """
        Creates a fake distribution for the agent.

        Returns:
            np.array: A numpy array representing the fake distribution for the agent.
        """
        vocal_size = VocabularyStateType.X.vocal_size
        fake_distribution = - np.ones((3, vocal_size)) * 1000
        x, y, heading = self.value[2], self.value[3], self.value[4]
        fake_distribution[0, int(x)] = 1
        fake_distribution[1, int(y)] = 1
        fake_distribution[2, int(heading)] = 1
        return fake_distribution

    @property
    def abbreviation(self):
        """
        Gets the abbreviation of the token.

        Returns:
            str: A string representing the abbreviation of the token.
        """
        classes = ['v', 'p', 'c', 'u']
        return classes[self.type_idx]

    @property
    def name(self):
        """
        Gets the name of the token.

        Returns:
            str: A string representing the name of the token.
        """
        return 'agent'

    @property
    def num_tokens(self):
        return 2

    # @property
    # def is_imagined(self):
    #     return self.predicted_future_trajectory is not None

    def __lt__(self, other):
        """
        Compare this instance with another object for sorting purposes, based on 
        'frame_index' and, if equal, on specific object types and attributes.

        The detailed comparison logic is as follows:
        - Objects are first compared based on their 'frame_index'. An object with a lower 
          'frame_index' is considered "less than" an object with a higher 'frame_index'.
        - If 'frame_index' values are equal, the method then considers the type of 'other'.
          If 'other' is an instance of 'BOSToken', 'TrafficLightEndToken', or 
          'TrafficLightToken', it is considered "greater than" this object.
        - If 'frame_index' values are equal and 'other' is an 'AgentToken', the comparison
          is based on a tuple comparison of (type_idx, track_id) between this object and 'other'.
        - In all other cases when 'frame_index' values are equal, this object is considered
          "less than" 'other'.

        Parameters:
            other (object): The object to compare to. It is assumed that 'other' will have
                            a 'frame_index' attribute and may be an instance of various token types.

        Returns:
            bool: True if this object is considered "less than" 'other', False otherwise.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.traffic_light_end_token import TrafficLightEndToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.newborn_begin_token import NewBornBeginToken

        # Primary comparison based on 'frame_index'
        if self.frame_index < other.frame_index:
            return True
        elif self.frame_index > other.frame_index:
            return False
        else:
            # Secondary comparison based on the type of 'other'
            if isinstance(other, (BOSToken, TrafficLightEndToken, TrafficLightToken)):
                return False
            elif isinstance(other, AgentToken):
                if self.is_ego:
                    return True
                if other.is_ego:
                    return False
                # Tertiary comparison for AgentToken instances based on type_idx and track_id
                return (self.type_idx, self.token_id) < (other.type_idx, other.token_id)
            elif isinstance(other, NewBornBeginToken):
                return self.agent_type == PositionalStateType.NORMAL_STATE
            else:
                return True

    def __eq__(self, token) -> bool:
        if not isinstance(token, AgentToken):
            return False
        return self.track_id == token.track_id and self.type_idx == token.type_idx and self.is_ego == token.is_ego

    def __hash__(self):
        # Hash based on a unique identifier for the object
        return hash((int(self.type_idx), int(self.track_id), self.is_ego))

    def __str__(self):
        """
        Returns a string representation of the token.

        Overrides the default string representation method to provide a human-readable
        description of the token, including its value and type.
        """
        # if ~np.isnan(self.x_1sec):
        #     return f'{self.token_id}{self.abbreviation}{self.track_id}({self.x:.2f}, {self.y:.2f};{self.x_1sec:.2f}, {self.y_1sec:.2f}){self.frame_index}'    
        # else:
        return f'{self.token_id}{self.abbreviation}{self.track_id}({self.x:.2f}, {self.y:.2f}, {self.heading:.2f}){self.frame_index}'    
    
    def next_possible_token_type(self):
        """
        Gets the next possible tokens in the sequence.

        Returns:
            List[BaseSingleToken]: A list of the next possible tokens in the sequence.
        """
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.newborn_begin_token import NewBornBeginToken
        from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
        return [AgentToken, NewBornBeginToken, BOSToken]

    def get_agent_state(self, return_detokenized_value=True):
        """
        return the agent state
        """
        if return_detokenized_value:
            if not self.detokenize_initial:
                width_token, length_token, x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec, x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec, track_id = self.value
                x, y, heading, x_1sec, y_1sec, heading_1sec, x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, width, length = AgentToken.detokenize(x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec, x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec, width_token, length_token)
                self.detoken_x, self.detoken_y, self.detoken_heading, self.detoken_width, self.detoken_length = x, y, heading, width, length
                self.detokenize_initial = True
            return self.detoken_x, self.detoken_y, self.detoken_heading, self.detoken_width, self.detoken_length, self.type_idx
        return self.x, self.y, self.heading, self.width, self.length, self.type_idx

    def get_object_state(self):
        return self.track_id, self.vx, self.vy, self.heading, self.width, self.length, self.x, self.y

    def get_tokenized_agent_state(self):
        """
        return the tokenized agent state
        """
        return [int(v) for v in self.value]

    def get_agent_idxes(self):
        assert self.index_in_sequence is not None, 'index_in_sequence is not assigned, please call assign_position first'
        return self.index_in_sequence + 1
    
    def get_control_state(self):
        return self.token_id
    
    def get_control_idxes(self, output=False):
        assert self.index_in_sequence is not None, 'index_in_sequence is not assigned, please call assign_position first'
        if output:
            return self.index_in_sequence + 1
        return self.index_in_sequence

    def update_closeloop_trajectory(self, x, y, heading):
        prev_future_trajectory = deepcopy(self.future_trajectory)
        prev_past_trajectory = deepcopy(self.past_trajectory)
        prev_future_trajectory[:, 0] += self._before_precondition_x
        prev_future_trajectory[:, 1] += self._before_precondition_y
        
        prev_past_trajectory[:, 0] += self._before_precondition_x
        prev_past_trajectory[:, 1] += self._before_precondition_y
        future_trajectory = np.full((prev_future_trajectory.shape[0], prev_future_trajectory.shape[1]), np.nan)
        future_trajectory[:-1, :] = prev_future_trajectory[1:, :]
        future_trajectory[:, 0] -= x 
        future_trajectory[:, 1] -= y
        
        # # past we should add 0,0 in the front and then shift all to dx, dy
        past_trajectory = np.full((prev_past_trajectory.shape[0], prev_past_trajectory.shape[1]), np.nan)
        past_trajectory[1:, :] = prev_past_trajectory[:-1, :]
        past_trajectory[0, 0] = self._before_precondition_x
        past_trajectory[0, 1] = self._before_precondition_y
        past_trajectory[0, 2] = self._before_precondition_heading
        past_trajectory[:, 0] -= x
        past_trajectory[:, 1] -= y
        self.past_trajectory, self.future_trajectory = past_trajectory, future_trajectory

    def update_agent_attribute(self, x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec, 
        x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec,
        width_token, length_token, closeloop_training=False):
        """
        Creates an AgentToken instance from an agent array.

        Args:
            agent_array (np.array): A numpy array representing the agent.

        Returns:
            AgentToken: An AgentToken instance representing the agent.
        """
        enable_temporal_aggregation = True
        gamma = 1.2
        ratio = [1, gamma, gamma**2, gamma**3]

        # (x_token, y_token, heading_token, width_token, length_token, track_id)
        track_id = int(self.track_id)
        
        # do not update width and length
        # width_token, length_token = self.value[5], self.value[6]
        value = [width_token, length_token, x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec, x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec, track_id]
        self.value = value
        x, y, heading, x_1sec, y_1sec, heading_1sec, x_1_5sec, y_1_5sec, heading_1_5sec, x_2sec, y_2sec, heading_2sec, width, length = AgentToken.detokenize(x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec, x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec, width_token, length_token)
        # dx, dy = x - self._before_precondition_x, y - self._before_precondition_y
        if closeloop_training:
            self.update_closeloop_trajectory(x, y, heading)
        
        if (self.x - float(x))**2 + (self.y - float(y))**2 > 5**2:
            return 
        self.x, self.y, self.heading = float(x), float(y), float(heading)
        

        if enable_temporal_aggregation:
            # Pre-compute for efficiency
            prev_positions = np.array([[self._prev_x[i][2-i], self._prev_y[i][2-i], self._prev_heading[i][2-i]] for i in range(3)])
            # print(prev_positions)
            valid_indices = np.ones((prev_positions.shape[0] + 1))>0
            valid_indices[1:] = ~(np.isnan(prev_positions[:, :]).any(axis=1)) # Assuming validity check is similar for x, y, heading

            # Filter valid previous positions
            valid_prev_positions = prev_positions[valid_indices[1:]]
            ratios = np.array(ratio)[valid_indices]
            self.x, self.y, self.heading = self.temporal_aggregation(self.x, self.y, self.heading, valid_prev_positions, ratios)
        
        self.x_1sec, self.y_1sec, self.heading_1sec = float(x_1sec), float(y_1sec), float(heading_1sec)
        self.x_1_5sec, self.y_1_5sec, self.heading_1_5sec = float(x_1_5sec), float(y_1_5sec), float(heading_1_5sec)
        self.x_2sec, self.y_2sec, self.heading_2sec = float(x_2sec), float(y_2sec), float(heading_2sec)

    @staticmethod
    @jit(nopython=True)
    def temporal_aggregation(x, y, heading, prev_positions, ratios, threshold=5.0):
        output_x, output_y = x * ratios[0], y * ratios[0]
        output_hcos, output_hsin = np.cos(heading) * ratios[0], np.sin(heading) * ratios[0]
        
        # Compute distances and apply threshold
        dists = np.sqrt((prev_positions[:, 0] - x)**2 + (prev_positions[:, 1] - y)**2)
        valid = dists < threshold
        
        # Filter valid positions and compute weighted sums
        valid_positions = prev_positions[valid]
        valid_ratios = ratios[1:][valid]
        output_x += np.sum(valid_ratios * valid_positions[:, 0])
        output_y += np.sum(valid_ratios * valid_positions[:, 1])
        output_hcos += np.sum(valid_ratios * np.cos(valid_positions[:, 2]))
        output_hsin += np.sum(valid_ratios * np.sin(valid_positions[:, 2]))
        
        # Compute final outputs
        accumulate_ratio = np.sum(valid_ratios) + ratios[0]
        final_x, final_y = output_x / accumulate_ratio, output_y / accumulate_ratio
        final_heading = np.arctan2(output_hsin, output_hcos)
    
        return final_x, final_y, final_heading
        
    def update_predicted_trajectory(
        self,
        predicted_future_trajectory: np.ndarray,
        pred_future_prob: np.ndarray,
    ):
        self.predicted_future_trajectory = predicted_future_trajectory
        self.pred_future_prob = pred_future_prob

    def to_npagentstate(self):
        """Get the bounding box of the ego vehicle.

        :return The bounding box of the ego vehicle, represented by 5 numbers:
            center_x, center_y, heading, half_length, half_width
        """
        return np.array([self.track_id, self.vx, self.vy, self.heading, self.width, self.length, self.x, self.y])
    
    def to_numpy_array(self):
        return np.array(
            [self.token_index_symbol,
            *[int(v) for v in self.value],
            self.track_id,
            self.token_id,
            self.type_idx,
            self.x,
            self.y,
            self.z,
            self.heading,
            self.width,
            self.length,
            self.vx,
            self.vy,
            self.frame_index,
            int(self.is_ego)]
        )
    
    @staticmethod
    def from_numpy(array):
        token_index_symbol = array[0]
        value = array[1:16]
        track_id = int(array[16])
        token_id = int(array[17])
        type_idx = int(array[18])
        x = array[19]
        y = array[20]
        z = array[21]
        heading = array[22]
        width = array[23]
        length = array[24]
        vx = array[25]
        vy = array[26]
        frame_index = int(array[27])
        is_ego = bool(array[28])
        past_trajectory = np.zeros((6, 3))
        future_trajectory = np.zeros((16, 3))
        nan_values = np.nan, np.nan, np.nan
        token = AgentToken(
            value, track_id, token_id, type_idx, x, y, z, heading, 
            *nan_values*3, width, length, past_trajectory, future_trajectory, 0.0, 0.0, frame_index, is_ego, None
        )
        return token
    
        
