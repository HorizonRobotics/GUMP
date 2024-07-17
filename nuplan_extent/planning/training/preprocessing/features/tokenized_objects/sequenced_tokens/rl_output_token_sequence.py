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
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.generation_token_sequence import GenerationTokenSequence


class RLOutputTokenSequence(GenerationTokenSequence):
    """
    Abstract base class for a structured sequence of tokens in a tokenized transformer model.

    This class provides the foundation for creating a sequence of tokens that follows a 
    specific order: "BOS, traffic light, traffic light end, agent(s), newborn begin, agent(s), BOS". 
    It ensures any subclass implementation maintains this sequence structure for proper model processing.

    Subclasses must implement specific methods to add tokens and validate the sequence according to this pattern.
    """
    
    
    ### forward
    def update_ego_embed(self, x):
        """
        Adds a traffic light token to the sequence.

        Args:
            traffic_light_tokens (List[TrafficLightToken]): A list of traffic light tokens to add to the sequence.
        """
        all_sequence_tokens = self.input_sequence_tokens + self.output_sequence_tokens
        ego_tokens, batch_indexes = all_sequence_tokens.get_agent_tokens(return_last=True, return_ego=True)
        control_idxes = [ego_token.get_control_idxes() for ego_token in ego_tokens]
        self.ego_embed = x[batch_indexes, control_idxes]
    
    def get_ego_action(self, all_sequence_tokens):
        """
        Gets the action of the ego agent.
        """
        ego_tokens, batch_indexes = all_sequence_tokens.get_agent_tokens(return_last=True, return_ego=True)
        log_action_distribution = torch.stack([ego_token.log_action_distribution for ego_token in ego_tokens], dim=0)
        log_action_distribution = log_action_distribution[:, [2,3,4]]
        fake_action_distribution = np.stack([ego_token.sampled_fake_distribution() for ego_token in ego_tokens], axis=0)
        fake_action_distribution = torch.from_numpy(fake_action_distribution).to(log_action_distribution.device).to(log_action_distribution.dtype)
        log_action_distribution = torch.cat([log_action_distribution, fake_action_distribution], dim=1)
        return log_action_distribution
            
    