from __future__ import annotations

import dataclasses
import logging
import torch
import numpy as np
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from abc import ABC, abstractmethod

from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

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
from nuplan_extent.planning.training.modeling.models.utils import boxes_overlap, boxes_overlap_axis_align


class GenerationTokenSequence(ABC):
    """
    Abstract base class for a structured sequence of tokens in a tokenized transformer model.

    This class provides the foundation for creating a sequence of tokens that follows a 
    specific order: "BOS, traffic light, traffic light end, agent(s), newborn begin, agent(s), BOS". 
    It ensures any subclass implementation maintains this sequence structure for proper model processing.

    Subclasses must implement specific methods to add tokens and validate the sequence according to this pattern.
    """

    def __init__(self, input_sequence_tokens: BatchTokenSequence):
        self.input_sequence_tokens = input_sequence_tokens  # List to store the sequence of tokens
        self.input_sequence_tokens.aggregate_traffic_light_info()
        self.batch_size = len(input_sequence_tokens)
        self.init_output()  # Initialize the output sequence of tokens


        self.pred_tl_logits = None
        self.pred_tl_tokens = None
        self.pred_log_prob = None
        
    def init_output(self):
        self.output_sequence_tokens = BatchTokenSequence()
        for i in range(self.batch_size):
            current_frame_index = self.input_sequence_tokens.data[i].current_frame_index
            self.output_sequence_tokens.add_batch(TokenSequence(num_frames=1, current_frame_index=current_frame_index))
        self.output_sequence_tokens.update_block_size(self.input_sequence_tokens.block_size)
        
    def update_input(self, input_sequence_tokens):
        self.input_sequence_tokens = input_sequence_tokens
        self.init_output()

    def __len__(self):
        return len(self.output_sequence_tokens)

    def append_bos_token(self):
        """
        Adds a BOS token to the sequence.
        """
        for i in range(self.batch_size):
            if len(self.output_sequence_tokens.data[i].tokens) > 0:
                frame_index = self.output_sequence_tokens.data[i].tokens[-1].frame_index + 1
            else:
                try:
                    frame_index = self.input_sequence_tokens.data[i].tokens[-1].frame_index + 1
                except:
                    frame_index = 0

            # num_traffic_light = len(self.input_sequence_tokens.data[i].get_traffic_light_info())
            self.output_sequence_tokens.data[i].add_token(BOSToken(frame_index=frame_index, num_traffic_light=None, num_agents=None))

    def append_tl_end_token(self):
        """
        Adds a traffic light end token to the sequence.
        """
        for i in range(self.batch_size):
            frame_index = self.output_sequence_tokens.data[i].tokens[-1].frame_index
            self.output_sequence_tokens.data[i].add_token(TrafficLightEndToken(frame_index=frame_index))

    def append_newborn_begin_token(self):
        """
        Adds a newborn begin token to the sequence.
        """
        for i in range(self.batch_size):
            frame_index = self.output_sequence_tokens.data[i].tokens[-1].frame_index
            self.output_sequence_tokens.data[i].add_token(NewBornBeginToken(frame_index=frame_index))

    def get_forward_sequence_tokens(self):
        """
        combine input_sequence_tokens and output_sequence_tokens
        """
        return self.input_sequence_tokens + self.output_sequence_tokens

    def update_traffic_light(self, pred_tl_logits, pred_tl_tokens):
        self.pred_tl_logits = pred_tl_logits
        self.pred_tl_tokens = pred_tl_tokens

        for batch_index, output_sequence_token in enumerate(self.output_sequence_tokens.data):
            frame_index = self.output_sequence_tokens.data[batch_index].tokens[-1].frame_index
            tl_infos = self.input_sequence_tokens.data[batch_index].get_traffic_light_info()
            num_traffic_light = len(tl_infos)
            for i in range(num_traffic_light):
                traffic_light_status = TrafficLightStatusType(pred_tl_tokens[i][batch_index].item())
                tl_index, lane_id, lane_coords = tl_infos[i].tl_index, tl_infos[i].lane_id, tl_infos[i].lane_coords
                output_sequence_token.add_token(TrafficLightToken(traffic_light_status, tl_index, lane_id, lane_coords, frame_index=frame_index))

    def update_trajectory(
        self,
        lvl_pred_past_trajectory,
        lvl_pred_future_trajectory,
        lvl_pred_past_sine_trajectory,
        lvl_pred_future_sine_trajectory,
        lvl_pred_log_prob,
        lvl_layer_index,
        agent_tokens,
        batch_indexes
    ):
        """
        Adds a list of agent tokens to the sequence.

        Args:
            trajectory_tokens (List[AgentToken]): A list of agent tokens to add to the sequence.
        """
        layer_index = -1
        self.pred_past_trajectory = lvl_pred_past_trajectory[layer_index] # [nagent, 6, 3]xnlayer
        self.pred_future_trajectory = lvl_pred_future_trajectory[layer_index] # [nagent, mode, 16, 3]xnlayer
        self.pred_past_sine_trajectory = lvl_pred_past_sine_trajectory[layer_index]
        self.pred_future_sine_trajectory = lvl_pred_future_sine_trajectory[layer_index]
        self.pred_log_prob = lvl_pred_log_prob[layer_index]
        self.lvl_layer_index = lvl_layer_index
        self.agent_tokens = agent_tokens
        self.batch_indexes = batch_indexes

    def update_agents(
        self,
        pred_agent_logits, 
        pred_agent_tokens,
        closeloop_training=False,
        return_ego=False,
        no_ego=False):
        self.pred_agent_logits = torch.stack(pred_agent_logits, dim=0)
        self.pred_agent_tokens = torch.stack(pred_agent_tokens, dim=0)
        pred_agent_tokens = self.pred_agent_tokens.float().cpu().numpy()
        agent_tokens, batch_indexes = self.output_sequence_tokens.get_agent_tokens(return_last=True, return_ego=return_ego)
        for i, agent_token in enumerate(agent_tokens):
            if no_ego and agent_token.is_ego:
                continue
            x_token = pred_agent_tokens[2][i]
            y_token = pred_agent_tokens[3][i]
            heading_token = pred_agent_tokens[4][i]
            x_token_1sec = pred_agent_tokens[5][i]
            y_token_1sec = pred_agent_tokens[6][i]
            heading_token_1sec = pred_agent_tokens[7][i]
            x_token_1_5sec = pred_agent_tokens[8][i]
            y_token_1_5sec = pred_agent_tokens[9][i]
            heading_token_1_5sec = pred_agent_tokens[10][i]
            x_token_2sec = pred_agent_tokens[11][i]
            y_token_2sec = pred_agent_tokens[12][i]
            heading_token_2sec = pred_agent_tokens[13][i]
            width_token = pred_agent_tokens[0][i]
            length_token = pred_agent_tokens[1][i]
            agent_token.update_agent_attribute(x_token, y_token, heading_token, x_token_1sec, y_token_1sec, heading_token_1sec,
                x_token_1_5sec, y_token_1_5sec, heading_token_1_5sec, x_token_2sec, y_token_2sec, heading_token_2sec,
                width_token, length_token, closeloop_training=closeloop_training)
            if agent_token.is_ego:
                agent_token.log_action_distribution = self.pred_agent_logits[:, 0, i, :]
            else:
                agent_token.log_action_distribution = None
    
    def collisions_exist(self):
        agent_tokens = self.output_sequence_tokens.get_agent_tokens_in_batchlist(return_last=True, return_all=True)
        collision_list = [False for _ in range(len(agent_tokens))]
        for batch_index in range(len(agent_tokens)):
            if len(agent_tokens[batch_index]) < 2:
                continue
            newborn_token = agent_tokens[batch_index][0]
            for i in range(1, len(agent_tokens[batch_index])):
                if newborn_token.is_overlap(agent_tokens[batch_index][i]):
                    collision_list[batch_index] = True
                    break
        return collision_list
    
    def make_groups(self):
        """
        Makes groups of tokens for each frame in the sequence.
        """
        self.groups = {}
        for i in range(len(self.agent_tokens)):
            agent_token = self.agent_tokens[i]
            batch_index = self.batch_indexes[i]
            frame_index = agent_token.frame_index
            if (batch_index, frame_index) not in self.groups:
                self.groups[(batch_index, frame_index)] = []
            self.groups[(batch_index, frame_index)].append(agent_token)
                   
    def precondition_normal_tokens(
        self,
        topk=6,
        next_ego_state=None,
    ):
        """
        Overwrites the ego state if it is provided. Otherwise, use the prediction from world model.
        next_ego_state: numpy array of [batchsize, 3] (x, y, heading)
        """
        pred_prob = torch.exp(self.pred_log_prob)
        # take argmax
        sampled_index = torch.argmax(pred_prob, dim=-1, keepdim=True)
        sampled_mode_index_squeezed = sampled_index.squeeze(-1)
        selected_trajectories = self.pred_future_trajectory[range(sampled_mode_index_squeezed.shape[0]), sampled_mode_index_squeezed, :, :].float().detach().cpu().numpy()
        pred_future_prob_np = torch.exp(self.pred_log_prob).float().detach().cpu().numpy()
        pred_future_trajectory_np = self.pred_future_trajectory.float().detach().cpu().numpy()
        for i, batch_index in enumerate(self.batch_indexes):
            selected_trajectory = selected_trajectories[i]
            pred_future_trajectory = pred_future_trajectory_np[i]
            pred_future_prob = pred_future_prob_np[i]
            prev_agent_token = self.agent_tokens[i]
            frame_index = self.output_sequence_tokens.data[batch_index].tokens[-1].frame_index
            if next_ego_state is not None and prev_agent_token.is_ego:
                next_state = next_ego_state[batch_index, :]
            else:
                next_state = selected_trajectory[0, :]
            next_agent_token = AgentToken.from_unrolled_nextstep(next_state, prev_agent_token, frame_index)
            prev_agent_token.update_predicted_trajectory(pred_future_trajectory, pred_future_prob)
            if next_agent_token._within_range() or next_agent_token.is_ego:
            # if next_agent_token._within_range():
                self.output_sequence_tokens.data[batch_index].add_token(next_agent_token)

        self.output_sequence_tokens.sort()
        
            
    
