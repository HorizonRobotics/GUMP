import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange
from functools import partial
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken
from nuplan_extent.planning.training.modeling.models.transition_models.base_transition_model import BaseTransitionModel
from nuplan_extent.planning.training.modeling.models.utils import get_sine_embedding_2d
import loralib as lora




class EmbeddingManager(nn.Module):
    def __init__(self,
                 map_range=(-56, -56, 56, 56),
                 n_embd=768,
                 max_num_traffic_light=64
                 ):
        super().__init__()
        self.map_range = map_range
        self.n_embd=n_embd
        self.max_num_traffic_light= max_num_traffic_light
        
        self.pos_embedding_mlp = nn.Sequential(
            nn.Linear(self.n_embd, 2 * self.n_embd),
            nn.LayerNorm(2 * self.n_embd),
            nn.ReLU(),
            nn.Linear(2 * self.n_embd, self.n_embd))
        self.token_embedding = nn.Embedding(VocabularyStateType.PAD_TOKEN.vocal_size, self.n_embd)
        self.agent_type_embedding = nn.Embedding(3, self.n_embd)
        self.traffic_light_status_embedding = nn.Embedding(4, self.n_embd)
        self.tl_index_embedding = nn.Embedding(self.max_num_traffic_light, self.n_embd)
        # self.speed_mlp = nn.Sequential(
        #     lora.Linear(1, self.n_embd, r=32),
        #     nn.LayerNorm(self.n_embd),
        #     nn.ReLU(),
        #     lora.Linear(self.n_embd, self.n_embd, r=32)
        # )
        self.merged = False
        
        
    def get_vision_embeddings(self, input):
        """
        Get positional embeddings for the given image features.
        input: image_feat: torch.Tensor, shape (B, self.n_embed, H, W)
        output: pos_embeddings: torch.Tensor, shape (B, self.n_embed, H, W)
        """
        vision = input['vision_x']
        # speed = input['speed']
        # import pdb; pdb.set_trace()
        # speed_embed = self.speed_mlp(speed)
        h, w = 13, 13
        # Determine the spacing for x and y based on image size
        x_space = (self.map_range[2] - self.map_range[0]) / w
        y_space = (self.map_range[3] - self.map_range[1]) / h

        # Create a grid for x and y
        y_map, x_map = torch.meshgrid(torch.linspace(self.map_range[2] + y_space / 2, self.map_range[3] - y_space / 2, h),
                                      torch.linspace(self.map_range[0] + x_space / 2, self.map_range[1] - x_space / 2, w))
        x_map = x_map.to(vision.device).to(vision.dtype)
        y_map = y_map.to(vision.device).to(vision.dtype)
        # Move the maps to the appropriate device and adjust dimensions

        x_map = x_map[None, ..., None].repeat(vision.shape[0], 1, 1, 1)
        y_map = y_map[None, ..., None].repeat(vision.shape[0], 1, 1, 1)

        x_map, y_map = -y_map, -x_map  # align with the agents system
        # Embed each position
        pos_embeddings = self.pos_embedding_mlp(get_sine_embedding_2d(x_map, y_map, None, None, None, self.n_embd).detach())
        # if not self.merged:
        #     vision = vision + speed_embed[:, None, :]
        vision = vision + rearrange(pos_embeddings, 'b h w c -> b (h w) c')
        vision = vision[:, :, None, :]  # fake dim n to shape:  b, t, n, d, for compatibility with visual transformer
        return vision
    
    def get_token_embeddings(self, sequence_tokens):
        """
        Get positional embeddings for the given sequence tokens.
        """
        device = self.pos_embedding_mlp[0].weight.device
        dtype = self.pos_embedding_mlp[0].weight.dtype
        sequence_tokens.update_device(device)
        sequence_tokens.update_dtype(dtype)
        sequence_tokens.assign_position()
        # Initialize the embeddings tensor
        token_embeddings = torch.zeros(sequence_tokens.num_batches, sequence_tokens.get_max_token_size(), self.n_embd, device=device, dtype=dtype)
        
        # embed for other tokens
        control_tokens = sequence_tokens.get_control_state()
        valid_batch_idxes, valid_seq_idxes = sequence_tokens.get_control_idxes()
        token_embeddings[valid_batch_idxes, valid_seq_idxes] = token_embeddings[valid_batch_idxes, valid_seq_idxes] + self.token_embedding(control_tokens)
        
        # sine embedding for agent state tokens
        x, y, heading, width, length, type_idx = sequence_tokens.get_agent_state()
        if x is None:
            return token_embeddings
        width_token, length_token, x_token, y_token, heading_token, _,_,_,_,_,_, _, _, _, _ = sequence_tokens.get_tokenized_agent_state()
        
        x_token = torch.clamp(x_token, 0, VocabularyStateType.PAD_TOKEN.vocal_size-1)
        y_token = torch.clamp(y_token, 0, VocabularyStateType.PAD_TOKEN.vocal_size-1)
        heading_token = torch.clamp(heading_token, 0, VocabularyStateType.PAD_TOKEN.vocal_size-1)
        width_token = torch.clamp(width_token, 0, VocabularyStateType.PAD_TOKEN.vocal_size-1)
        length_token = torch.clamp(length_token, 0, VocabularyStateType.PAD_TOKEN.vocal_size-1)

        token_embed = self.token_embedding(x_token) + self.token_embedding(y_token) + self.token_embedding(heading_token) + self.token_embedding(width_token) + self.token_embedding(length_token)
        token_embed = token_embed.squeeze(1)
        valid_batch_idxes, valid_seq_idxes = sequence_tokens.get_agent_idxes()
        agent_pos_embed = self.pos_embedding_mlp(get_sine_embedding_2d(x, y, heading, width, length, self.n_embd))
        # TODO : check if type_idx is correct
        type_idx = torch.clamp(type_idx, 0, 2)
        agent_type_embed = self.agent_type_embedding(type_idx.squeeze(1).long())
        
        # we put agent type embedding to the previous token
        token_embeddings[valid_batch_idxes, valid_seq_idxes-1] = token_embeddings[valid_batch_idxes, valid_seq_idxes-1] + agent_type_embed
        
        token_embeddings[valid_batch_idxes, valid_seq_idxes] = token_embeddings[valid_batch_idxes, valid_seq_idxes] + agent_pos_embed + token_embed
        
        # embed for traffic light tokens
        x, y, heading, tl_state, tl_index = sequence_tokens.get_traffic_light_state()
        if x is None:
            return token_embeddings
        valid_batch_idxes, valid_seq_idxes = sequence_tokens.get_traffic_light_idxes()
        tl_pos_embed = self.pos_embedding_mlp(get_sine_embedding_2d(x, y, heading, None, None, self.n_embd))
        tl_state_embed = self.traffic_light_status_embedding(tl_state.long().squeeze(1))
        tl_index_embed = self.tl_index_embedding(tl_index.long().squeeze(1))
        token_embeddings[valid_batch_idxes, valid_seq_idxes] = token_embeddings[valid_batch_idxes, valid_seq_idxes] + tl_pos_embed + tl_state_embed + tl_index_embed
        return token_embeddings
    
    def get_traffic_light_embedding(self, sequence_tokens, return_last=False):
        device = self.pos_embedding_mlp[0].weight.device
        dtype = self.pos_embedding_mlp[0].weight.dtype
        max_num_traffic_light = np.max(sequence_tokens.get_num_traffic_light(return_last=return_last))
        valid_batch_idxes, valid_seq_idxes = sequence_tokens.get_bos_idxes(return_last=return_last)
        bos_tokens = sequence_tokens.get_bos_tokens(return_last=return_last)
        tl_embed = torch.zeros(len(bos_tokens), max_num_traffic_light, self.n_embd, device=device, dtype=dtype)
        batched_traffic_light_info = sequence_tokens.get_traffic_light_info()
        for batch_index, traffic_light_info in enumerate(batched_traffic_light_info):
            traffic_light_info = sorted(traffic_light_info)
            for i, tl_token in enumerate(traffic_light_info):
                x, y, heading, _, tl_index = tl_token.get_traffic_light_state()
                
                x, y, heading = torch.tensor(x).to(device).to(dtype), torch.tensor(y).to(device).to(dtype), torch.tensor(heading).to(device).to(dtype)
                tl_index = torch.tensor(tl_index).to(device).long()
                tl_pos_embed = self.pos_embedding_mlp(get_sine_embedding_2d(x.unsqueeze(0), y.unsqueeze(0), heading.unsqueeze(0), None, None, self.n_embd))
                tl_index_embed = self.tl_index_embedding(tl_index.unsqueeze(0))
                
                tl_embed[valid_batch_idxes==batch_index, i] = tl_pos_embed + tl_index_embed
        return tl_embed
        