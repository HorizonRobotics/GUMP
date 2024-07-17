from __future__ import annotations

import dataclasses
import logging
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from abc import ABC, abstractmethod


from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.base_token_sequence import BaseTokenSequence
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.token_sequence import TokenSequence
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.batch_token_sequence import BatchTokenSequence
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.traffic_light_end_token import TrafficLightEndToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.newborn_begin_token import NewBornBeginToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.pad_token import PADToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType


class OutputTokenSequence(ABC):
    """
    Abstract base class for a structured sequence of tokens in a tokenized transformer model.

    This class provides the foundation for creating a sequence of tokens that follows a 
    specific order: "BOS, traffic light, traffic light end, agent(s), newborn begin, agent(s), BOS". 
    It ensures any subclass implementation maintains this sequence structure for proper model processing.

    Subclasses must implement specific methods to add tokens and validate the sequence according to this pattern.
    """

    def __init__(self, input_sequence_tokens: BatchTokenSequence):
        self.input_sequence_tokens = input_sequence_tokens  # List to store the sequence of tokens
    
    ### forward
    def update_traffic_light(self, pred_tl_logits, pred_tl_tokens):
        """
        Adds a traffic light token to the sequence.

        Args:
            traffic_light_tokens (List[TrafficLightToken]): A list of traffic light tokens to add to the sequence.
        """
        self.pred_tl_logits = pred_tl_logits
        self.pred_tl_tokens = pred_tl_tokens

    def update_agents(self, pred_agent_logits, pred_agent_tokens):
        """
        Adds a list of agent tokens to the sequence.

        Args:
            agent_tokens (List[AgentToken]): A list of agent tokens to add to the sequence.
        """
        self.pred_agent_logits = pred_agent_logits
        self.pred_agent_tokens = pred_agent_tokens
    
    def update_trajectory(self, 
                          lvl_pred_past_trajectory, 
                          lvl_pred_future_trajectory, 
                          lvl_pred_past_sine_trajectory, 
                          lvl_pred_future_sine_trajectory, 
                          lvl_pred_log_prob, 
                          lvl_layer_index,
                          resampled_agent_tokens,
                          batch_indexes):
        """
        Adds a list of agent tokens to the sequence.

        Args:
            trajectory_tokens (List[AgentToken]): A list of agent tokens to add to the sequence.
        """
        self.lvl_pred_past_trajectory = lvl_pred_past_trajectory
        self.lvl_pred_future_trajectory = lvl_pred_future_trajectory
        self.lvl_pred_past_sine_trajectory = lvl_pred_past_sine_trajectory
        self.lvl_pred_future_sine_trajectory = lvl_pred_future_sine_trajectory
        self.lvl_pred_log_prob = lvl_pred_log_prob
        self.lvl_layer_index = lvl_layer_index
        self.resampled_agent_tokens = resampled_agent_tokens
        self.batch_indexes = batch_indexes
        
    
    def update_normal_tokens(self, pred_normal_logits, pred_normal_tokens):
        """
        Adds a list of agent tokens to the sequence.

        Args:
            control_tokens (List[AgentToken]): A list of agent tokens to add to the sequence.
        """
        self.pred_normal_logits = pred_normal_logits
        self.pred_normal_tokens = pred_normal_tokens
        
    ### objective
        
    def get_traffic_prediction_target(self):
        """
        extract the target of traffic light prediction, as a tensor of shape (num_tl, num_frames)
        """
        bos_tokens = self.input_sequence_tokens.get_bos_tokens()
        pred_logits = []
        target_token = []
        for i,bos_token in enumerate(bos_tokens):
            tl_tokens = bos_token.traffic_light_tokens
            for j, tl_token in enumerate(tl_tokens):
                pred_logits.append(self.pred_tl_logits[j][0, i])
                target_token.append(tl_token.traffic_light_status.value)
        if len(pred_logits) == 0:
            return None, None
        pred_logits = torch.stack(pred_logits, dim=0)
        target_token = torch.tensor(target_token, device=pred_logits.device).long()
        return pred_logits, target_token
    
    def get_agent_state_prediction_target(self, return_ego_only=False):
        pred_logits = torch.cat(self.pred_agent_logits, dim=0)
        self.input_sequence_tokens.device = pred_logits.device
        target_tokens = self.input_sequence_tokens.get_tokenized_agent_state(return_ego_only=return_ego_only)[:-1] # x_token, y_token, heading_token, width_token, length_token, track_id, we do not need the last one
        pred_logits = pred_logits.reshape(-1, pred_logits.size(-1))
        return pred_logits, target_tokens
    
    def get_control_prediction_target(self):
        pred_logits = self.pred_normal_logits
        target_tokens = []
        self.input_sequence_tokens.device = pred_logits.device
        normal_decoding_tokens = self.input_sequence_tokens.get_normal_decoding_tokens()
        for normal_decoding_token in normal_decoding_tokens:
            target_tokens.append(normal_decoding_token.get_control_state())
        target_tokens = torch.tensor(target_tokens, device=pred_logits.device).long()
        return pred_logits, target_tokens
    
    def get_past_trajectory_prediction_target(self):
        lvl_pred_past_trajectory = torch.stack(self.lvl_pred_past_sine_trajectory, dim=0)
        target_past_trajectory = np.stack([agent_token.past_trajectory for agent_token in self.resampled_agent_tokens])
        target_past_trajectory = torch.from_numpy(target_past_trajectory).to(lvl_pred_past_trajectory.device).to(lvl_pred_past_trajectory.dtype)
        return lvl_pred_past_trajectory, target_past_trajectory
    
    def get_future_trajectory_prediction_target(self):
        lvl_pred_future_trajectory = torch.stack(self.lvl_pred_future_sine_trajectory, dim=0)
        lvl_pred_log_prob = torch.stack(self.lvl_pred_log_prob, dim=0)
        target_future_trajectory = np.stack([agent_token.future_trajectory for agent_token in self.resampled_agent_tokens])
        target_future_trajectory = torch.from_numpy(target_future_trajectory).to(lvl_pred_log_prob.device).to(lvl_pred_log_prob.dtype)
        return lvl_pred_log_prob, lvl_pred_future_trajectory, target_future_trajectory
    