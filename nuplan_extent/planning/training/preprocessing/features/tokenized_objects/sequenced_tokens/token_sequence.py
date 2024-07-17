from __future__ import annotations

import dataclasses
import logging
import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from abc import ABC, abstractmethod


from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.base_token_sequence import BaseTokenSequence
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.traffic_light_end_token import TrafficLightEndToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.newborn_begin_token import NewBornBeginToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.pad_token import PADToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType

token_index_symbol_dict = {
    6: BOSToken,
    1: AgentToken,
    2: NewBornBeginToken,
    3: PADToken,
    4: TrafficLightEndToken,
    5: TrafficLightToken
}

class TokenSequence(BaseTokenSequence):
    """
    Abstract base class for a structured sequence of tokens in a tokenized transformer model.

    This class provides the foundation for creating a sequence of tokens that follows a 
    specific order: "BOS, traffic light, traffic light end, agent(s), newborn begin, agent(s), BOS". 
    It ensures any subclass implementation maintains this sequence structure for proper model processing.

    Subclasses must implement specific methods to add tokens and validate the sequence according to this pattern.
    """

    def __init__(self, num_frames=0, current_frame_index=0, block_size=2048):
        self.tokens = []  # List to store the sequence of tokens
        self.num_frames = num_frames  # Number of frames in the sequence
        self.current_frame_index = current_frame_index  # Index of the current frame in the sequence
        self.block_size = block_size # The maximum number of tokens in a sequence
        self.tl_index = None # The index of the traffic light tokens
        self.traffic_light_info = None
        self.total_num_tokens = None
        self.training = True
    
    @property
    def start_frame_index(self):
        return self.tokens[0].frame_index
    
    @property
    def end_frame_index(self):
        return self.tokens[-1].frame_index
        
    def num_bos_token(self):
        num = 0
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], BOSToken):
                num += 1
        return num

    def add_token(self, token: Union[TrafficLightToken, AgentToken, TrafficLightEndToken, BOSToken, NewBornBeginToken, PADToken]):
        """
        Adds a token to the token sequence while enforcing the specific sequence structure.

        Args:
            token (Union[TrafficLightToken, AgentToken, TrafficLightEndToken, BOSToken, NewBornBeginToken]): 
                The token to be added to the sequence.
        """
        self.tokens.append(token)
        
    def add_tokens(self, tokens: TokenSequence):
        """
        Adds a token sequence to the current token sequence.

        Args:
            tokens (TokenSequence): The token sequence to be added to the current sequence.
        """
        self.tokens.extend(tokens.tokens)
        
    def __add__(self, other):
        """
        concatenate two token sequences
        """
        assert isinstance(other, TokenSequence), "other must be a TokenSequence"
        new_sequence = TokenSequence(num_frames=self.num_frames + other.num_frames, current_frame_index=self.current_frame_index, block_size=self.block_size)
        new_sequence.tokens = self.tokens + other.tokens
        new_sequence.assign_position()
        return new_sequence
    
    def __str__(self) -> str:
        """
        Returns a string representation of the token sequence.

        Overrides the default string representation method to provide a human-readable
        description of the token sequence, including the string representation of each token.
        """
        return ' '.join([str(token) for token in self.tokens])
    
    def validate_sequence(self) -> bool:
        """
        Validates whether the current token sequence adheres to the specified structure.

        Returns:
            bool: True if the sequence is valid as per the defined structure, False otherwise.
        """
        return True
    
    def reindexing_agent_tokens(self, scenario_agent_set):
        """
        Reindex the agent tokens.
        """
        scenario_agent_list = list(set(scenario_agent_set))
        scenario_agent_list = sorted(scenario_agent_list, key=lambda x: (x.type_idx, x.track_id, x.is_ego))
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and not self.tokens[i].is_ego:
                self.tokens[i].token_id = scenario_agent_list.index(self.tokens[i]) + 1 # 0 is reserved for ego
            
    def reorganize_newborn_tokens(self):
        """
        find the newborn tokens and reorganize them
        """
        prev_agent_token = {}
        prev_max_id = -1
        prev_frame_index = 0
        current_agent_token = {}
        current_max_id = -1
        current_newborn_token = []
        reorg_tokens = []
        for i in range(len(self.tokens)):
            current_frame_index = self.tokens[i].frame_index
            test = TokenSequence()
            test.tokens = reorg_tokens
            # print(self)
            # print(test)
            # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
            if current_frame_index > prev_frame_index:
                reorg_tokens.append(NewBornBeginToken(frame_index=prev_frame_index))
                prev_agent_token = current_agent_token
                prev_max_id = current_max_id
                prev_frame_index = current_frame_index
                current_agent_token = {}
                current_max_id = -1
                for j in range(len(current_newborn_token)):
                    reorg_tokens.append(current_newborn_token[j])
                current_newborn_token = []
                
            if not isinstance(self.tokens[i], AgentToken):
                reorg_tokens.append(self.tokens[i])
                continue
                
            current_agent_token[self.tokens[i]] = i
            
            if current_frame_index == 0:
                reorg_tokens.append(self.tokens[i])
                current_max_id = max(current_max_id, self.tokens[i].token_id)
                continue
            
            # find the newborn tokens
            if self.tokens[i] in prev_agent_token:
                prev_index = prev_agent_token[self.tokens[i]]
                self.tokens[i].token_id = self.tokens[prev_index].token_id
                self.tokens[i].agent_type = PositionalStateType.NORMAL_STATE
                reorg_tokens.append(self.tokens[i])
            else:
                prev_max_id += 1
                self.tokens[i].token_id = prev_max_id
                self.tokens[i].agent_type = PositionalStateType.NEWBORN_STATE
                current_newborn_token.append(self.tokens[i])
                
            current_max_id = max(current_max_id, self.tokens[i].token_id)
        self.tokens = reorg_tokens
        
    def sort(self):
        """
        Sort the tokens based on the frame index and token id
        """
        self.tokens.sort()

    def drop_ego(self, ego_dropout_rate=0.3):
        """
        Drop the ego agent token
        """
        new_tokens = []
        drop_mask = np.random.rand(self.num_frames) < ego_dropout_rate
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and self.tokens[i].is_ego and self.tokens[i].frame_index < self.num_frames and drop_mask[self.tokens[i].frame_index]:
                self.tokens[i] = PADToken(frame_index=self.tokens[i].frame_index)
        
    def drop_frames(self, frame_dropout_rate: float):
        """
        Drop frames from the token sequence based on the specified dropout rate.
        :param frame_dropout_rate: The rate at which frames should be dropped from the sequence.
        """
        frame_drop_mask = np.random.rand(self.num_frames) < frame_dropout_rate
        for i in range(len(self.tokens)):
            if self.tokens[i].frame_index < self.num_frames and frame_drop_mask[self.tokens[i].frame_index]:
                self.tokens[i] = PADToken(frame_index=self.tokens[i].frame_index)
    
    def resample(self):
        """
        sampling the token sequence, starting from a random frame index
        """
        start_index = self.start_frame_index + np.random.randint(0, max(self.tokens[-1].frame_index - 3, 1))
        resampled_sequence = TokenSequence(num_frames=self.num_frames - start_index, current_frame_index=self.current_frame_index)
        for i in range(len(self.tokens)):
            if self.tokens[i].frame_index >= start_index:
                resampled_sequence.add_token(self.tokens[i])
        resampled_sequence.assign_position()
        return resampled_sequence
    
    def sample_frame_segment(self, start_local_index=0, end_local_index=None):
        """
        sampling the token sequence, starting from 0 to frame_index, [0, frame_index]
        if frame_index is None, cut off till the current frame index (last included)
        """
        start_frame_index = self.start_frame_index + start_local_index
        end_frame_index = self.start_frame_index + end_local_index
        
        resampled_sequence = TokenSequence(num_frames=end_frame_index - start_frame_index + 1, current_frame_index=self.current_frame_index)
        for i in range(len(self.tokens)):
            if self.tokens[i].frame_index >= start_frame_index and self.tokens[i].frame_index <= end_frame_index:
                resampled_sequence.add_token(self.tokens[i])
        resampled_sequence.block_size = self.block_size
        return resampled_sequence
    
    def get_ego_agent_token(self):
        """
        get the ego agent token
        """
        ego_tokens = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and self.tokens[i].is_ego and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                ego_tokens.append(self.tokens[i])
        return ego_tokens

    def get_prev_ego_token(self):
        """
        get the ego agent token
        """
        position = 0
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], AgentToken) and self.tokens[i].is_ego:
                position += 1
                if position == 2:
                    return self.tokens[i]
        return None
    
    def get_object_state(self):
        agent_tokens = []
        all_agent_tokens = self.get_agent_tokens(return_last=True)

        for i in range(len(all_agent_tokens)):
            if all_agent_tokens[i].is_ego:
                continue
            agent_tokens.append(all_agent_tokens[i])
        num_objects = len(agent_tokens)
        object_state = np.zeros((num_objects, 8), dtype=np.float32)
        for i in range(num_objects):
            object_state[i] = np.array(agent_tokens[i].get_object_state())
        return object_state

    def get_agent_token_by_raw_id(self, raw_id: int):
        """
        get the agent token by raw id
        """
        agent_tokens = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and self.tokens[i].raw_id == raw_id:
                agent_tokens.append(self.tokens[i])
        assert len(agent_tokens) == 1, f"agent_tokens should have only one element, get {len(agent_tokens)} elements" 
        return agent_tokens

    def assign_position(self):
        """
        Tokenize the token sequence more efficiently.
        """
        index_in_sequence = 0
        total_num_tokens = 0
        block_size = self.block_size  # Access block_size once and use a local variable
        
        for token in self.tokens:  # Removed enumerate since `i` is unused
            num_tokens = token.num_tokens  # Access once per iteration
            
            # Directly calculate validity and update total_num_tokens if valid
            if index_in_sequence + num_tokens < block_size:
                token.valid_for_training = True
                total_num_tokens += num_tokens
            else:
                token.valid_for_training = False
            
            token.index_in_sequence = index_in_sequence
            index_in_sequence += num_tokens
        
        self.total_num_tokens = total_num_tokens

    def update_ego_state(self, ego_state):
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], AgentToken) and self.tokens[i].is_ego:
                self.tokens[i].x = float(ego_state.x)
                self.tokens[i].y = float(ego_state.y)
                self.tokens[i].heading = float(ego_state.heading)
                self.tokens[i].value[2:4] = list(AgentToken.tokenize(ego_state.x, ego_state.y, ego_state.heading, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0))[2:4]
                break
        return self
            
    def get_agent_state(self):
        agent_state = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                agent_state.append(self.tokens[i].get_agent_state())
        return agent_state
    
    def get_tokenized_agent_state(self, return_ego_only=False):
        agent_state = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                if return_ego_only:
                    if self.tokens[i].is_ego:
                        agent_state.append(self.tokens[i].get_tokenized_agent_state())
                else:
                    agent_state.append(self.tokens[i].get_tokenized_agent_state())
        return agent_state

    def get_agent_idxes(self, batch_idx):
        agent_idxes = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                agent_idxes.append([batch_idx, self.tokens[i].get_agent_idxes()])
        return agent_idxes

    def get_ego_agent_control_idxes(self, batch_idx):
        agent_idxes = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and self.tokens[i].is_ego and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                agent_idxes.append([batch_idx, self.tokens[i].get_control_idxes()])
        return agent_idxes
    
    def get_agent_control_idxes(self, batch_idx, return_last_frame=False, return_last_agent=False, return_ego=False):
        if return_last_frame:
            return self.get_last_frame_control_idxes(batch_idx, return_ego=return_ego)
        if return_last_agent:
            return self.get_last_agent_control_idxes(batch_idx)
        if return_ego:
            return self.get_ego_agent_control_idxes(batch_idx)
        agent_idxes = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                agent_idxes.append([batch_idx, self.tokens[i].get_control_idxes()])
        return agent_idxes

    def get_last_frame_control_idxes(self, batch_idx, return_ego=False):
        last_frame_index = None
        agent_idxes = []
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], AgentToken):
                if return_ego and not self.tokens[i].is_ego:
                    continue
                if last_frame_index is None:
                    last_frame_index = self.tokens[i].frame_index
                if last_frame_index != self.tokens[i].frame_index:
                    break
                agent_idxes.append([batch_idx, self.tokens[i].get_control_idxes()])
        return agent_idxes

    def get_last_agent_control_idxes(self, batch_idx):
        agent_idxes = []
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], AgentToken) and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                agent_idxes.append([batch_idx, self.tokens[i].get_control_idxes()])
                break
        return agent_idxes
    
    def get_normal_decoding_idxes(self, batch_idx):
        agent_idxes = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], (AgentToken, TrafficLightEndToken, NewBornBeginToken, PADToken)) and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                agent_idxes.append([batch_idx, self.tokens[i].get_control_idxes(output=True)])
        return agent_idxes

    def get_normal_decoding_tokens(self):
        agent_idxes = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], (AgentToken, TrafficLightEndToken, NewBornBeginToken, PADToken)) and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                agent_idxes.append(self.tokens[i])
        return agent_idxes

    def get_control_state(self):
        control_state = []
        for i in range(len(self.tokens)):
            if hasattr(self.tokens[i], 'get_control_state') and ((self.training and self.tokens[i].valid_for_training) or not self.training):
                control_state.append(self.tokens[i].get_control_state())
        return control_state
    
    def get_control_idxes(self, batch_idx):
        control_idxes = []
        for i in range(len(self.tokens)):
            if ((self.training and self.tokens[i].valid_for_training) or not self.training):
                control_idxes.append([batch_idx, self.tokens[i].get_control_idxes()])
        return control_idxes

    def get_traffic_light_state(self, return_all=False):
        traffic_light_state = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], TrafficLightToken):
                if return_all or self.tokens[i].valid_for_training:
                    traffic_light_state.append(self.tokens[i].get_traffic_light_state())
        return traffic_light_state

    def get_traffic_light_idxes(self, batch_idx):
        traffic_light_idxes = []
        self.tl_index = set()
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], TrafficLightToken) and self.tokens[i].valid_for_training:
                traffic_light_idxes.append([batch_idx, self.tokens[i].get_traffic_light_idxes()])
                self.tl_index.add(int(self.tokens[i].tl_index))
        return traffic_light_idxes

    def get_bos_idxes(self, batch_idx, return_last=False):
        if return_last:
            return self.get_last_bos_idxes(batch_idx)
        bos_idxes = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], BOSToken) and self.tokens[i].valid_for_training:
                bos_idxes.append([batch_idx, self.tokens[i].get_control_idxes()])
        return bos_idxes

    def get_last_bos_idxes(self, batch_idx):
        bos_idxes = []
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], BOSToken) and self.tokens[i].valid_for_training:
                bos_idxes.append([batch_idx, self.tokens[i].get_control_idxes()])
                break
        return bos_idxes
    
    def get_bos_tokens(self, return_last=False):
        if return_last:
            return self.get_last_bos_tokens()
        bos_tokens = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], BOSToken) and self.tokens[i].valid_for_training:
                bos_tokens.append(self.tokens[i])
        return bos_tokens

    def get_last_bos_tokens(self):
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], BOSToken) and self.tokens[i].valid_for_training:
                return [self.tokens[i]]

    def get_agent_tokens(self, return_last=False, return_all=False, return_ego=False, return_prev=False):
        if return_last:
            return self.get_last_agent_tokens(return_ego=return_ego)
        if return_prev:
            return self.get_prev_agent_tokens()
        if return_ego:
            return self.get_ego_agent_token()
        agent_tokens = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken):
                if return_all:
                    agent_tokens.append(self.tokens[i])
                else:
                    if self.tokens[i].valid_for_training:
                        agent_tokens.append(self.tokens[i])
        return agent_tokens
    
    def get_last_agent_tokens(self, return_ego):
        agent_tokens = []
        last_frame_index = None
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], AgentToken):
                if return_ego and not self.tokens[i].is_ego:
                    continue
                if last_frame_index is None:
                    last_frame_index = self.tokens[i].frame_index
                if last_frame_index != self.tokens[i].frame_index:
                    break
                agent_tokens.append(self.tokens[i])
        return agent_tokens

    def get_prev_agent_tokens(self):
        agent_tokens = []
        last_frame_index = None
        count = 0
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], AgentToken):
                if last_frame_index is None:
                    last_frame_index = self.tokens[i].frame_index
                if last_frame_index != self.tokens[i].frame_index:
                    count += 1
                if count > 1:
                    break
                if count > 0:
                    agent_tokens.append(self.tokens[i])
                last_frame_index = self.tokens[i].frame_index
        return agent_tokens

    def remove_ego_history(self):
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], AgentToken) and self.tokens[i].is_ego:
                self.tokens[i].tozeros()
                
    
    def get_tl_index(self):
        assert self.tl_index is not None, "self.tl_index is not initialized, Please call get_traffic_light_idxes() first"
        return self.tl_index
    
    def aggregate_traffic_light_info(self):
        traffic_light_tokens = []
        self.traffic_light_info = []
        for i in range(len(self.tokens)-1, -1, -1):
            if isinstance(self.tokens[i], TrafficLightToken):
                traffic_light_tokens.append(self.tokens[i])
            if isinstance(self.tokens[i], BOSToken):
                traffic_light_tokens = sorted(traffic_light_tokens)
                self.tokens[i].traffic_light_tokens = traffic_light_tokens
                self.traffic_light_info = self.tokens[i].traffic_light_tokens
                traffic_light_tokens = []
                
    def get_traffic_light_status_list(self, max_num_traffic_light=64):
        traffic_light_status_list = []
        for i in range(len(self.tokens)):
            if isinstance(self.tokens[i], BOSToken) and self.tokens[i].valid_for_training:
                traffic_light_status_list.append([])
                for tl_token in self.tokens[i].traffic_light_tokens:
                    traffic_light_status_list[-1].append(int(tl_token.traffic_light_status.value))
                traffic_light_status_list[-1] = traffic_light_status_list[-1] + [0] * (max_num_traffic_light - len(traffic_light_status_list[-1]))
        return traffic_light_status_list
                
    def get_traffic_light_info(self):
        assert self.traffic_light_info is not None, "self.traffic_light_info is not initialized, Please call aggregate_traffic_light_info() first"
        return self.traffic_light_info
    
    def mark_as_imagined(self):
        for i in range(len(self.tokens)):
            self.tokens[i].is_imagined = True

    def get_imagined_token_seqeunce(self):
        imagined_sequence = TokenSequence(num_frames=self.num_frames, current_frame_index=self.current_frame_index, block_size=self.block_size)
        frame_index = -1
        num_frames = 0
        for i in range(len(self.tokens)):

            if self.tokens[i].is_imagined:
                imagined_sequence.add_token(self.tokens[i])
                if self.tokens[i].frame_index > frame_index:
                    frame_index = self.tokens[i].frame_index
                    num_frames += 1
        imagined_sequence.num_frames = num_frames
        return imagined_sequence
    
    def replace_with_imagined_tokens(self, imagined_sequence):
        start_index = imagined_sequence.start_frame_index
        end_index = imagined_sequence.end_frame_index 
        segment_one = self.sample_frame_segment(start_local_index=0, end_local_index=start_index-1)
        original_segment_two = self.sample_frame_segment(start_local_index=start_index, end_local_index=end_index)
        left_tokens = []
        
        for i in range(len(original_segment_two.tokens)):
            matched = False
            for token in imagined_sequence.tokens:
                if token.frame_index == original_segment_two.tokens[i].frame_index and token == original_segment_two.tokens[i]:
                    left_tokens.append(token)
                    matched = True
                    break
            if not matched:
                left_tokens.append(original_segment_two.tokens[i])
                
        imagined_sequence.tokens = left_tokens
        segment_two = imagined_sequence
        segment_three = self.sample_frame_segment(start_local_index=end_index+1, end_local_index=self.end_frame_index)
        
        new_sequence = segment_one + segment_two + segment_three
        new_sequence.assign_position()
        
        if len(new_sequence) != len(self):
            return self
        for i in range(len(self.tokens)):
            assert self.tokens[i] == new_sequence.tokens[i], f"self.tokens[i] should be {new_sequence.tokens[i]}, get {self.tokens[i]}"
        return new_sequence
    
    def get_valid_num_frames_for_training(self):
        num_frames = 0
        for i in range(len(self.tokens)):
            if self.tokens[i].valid_for_training:
                num_frames = self.tokens[i].frame_index + 1
        return num_frames

    def numpy(self):
        numpy_array = []
        for i in range(len(self.tokens)):
            numpy_array.append(self.tokens[i].to_numpy_array())
        return np.stack(numpy_array)

    def from_numpy(self, numpy_array):
        if isinstance(numpy_array, torch.Tensor) :
            numpy_array = numpy_array.cpu().numpy()
        self.tokens = []
        for i in range(numpy_array.shape[0]):
            token_index_symbol = int(numpy_array[i][0])
            if token_index_symbol <= 0:
                break
            token = token_index_symbol_dict[token_index_symbol].from_numpy(numpy_array[i])
            self.tokens.append(token)
        self.assign_position()
        return self
        