from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.kinetics_state_type import KineticsVocabularyStateType
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils
import nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils as kutils
from nuplan_extent.planning.training.modeling.models.utils import get_sine_embedding_2d
from nuplan_extent.planning.training.modeling.models.utils.positional_encoding import get_sine_embedding_kinetics_2d

class KineticsEmbedder(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                 map_range=(-56, -56, 56, 56),
                 n_embd=768,
                 block_size=512,
                 max_frames=21,
                 hidden=128,
                 ):
        super().__init__()
        self.map_range = map_range
        self.n_embd = n_embd
        self.block_size = block_size
        self.max_frames = max_frames
        self.inital_pos_nembed = 1024
        
        self.visual_mlp = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd * 2),
            nn.GELU(),
            nn.Linear(self.n_embd * 2, self.n_embd))

        self.token_embedding = nn.Embedding(KineticsVocabularyStateType.CONTROL.vocal_size, self.n_embd)

        self.hidden = hidden
        # self.vision_pos = nn.Sequential(
        #     nn.Linear(self.n_embd, self.n_embd))
        
        self.pos_embed = nn.Sequential(
            nn.Linear(self.inital_pos_nembed, self.n_embd))
        self.state_embed = nn.Sequential(
            nn.Linear(self.inital_pos_nembed, self.n_embd))
        
        self.class_embedding = nn.Embedding(3, self.n_embd)
        # self.agent_index_embedding = nn.Embedding(416, self.n_embd)
        self.frame_pos_embedding = nn.Embedding(max_frames, self.n_embd)
        

    def get_vision_embeddings(self, vision):
        """
        Get positional embeddings for the given image features.
        input: image_feat: torch.Tensor, shape (B, self.n_embed, H, W)
        output: pos_embeddings: torch.Tensor, shape (B, self.n_embed, H, W)
        """
        h = w = int(math.sqrt(vision.shape[1]))
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
        pos_embeddings = self.pos_embed(get_sine_embedding_2d(x_map, y_map, None, None, None, None, self.inital_pos_nembed).detach())
        pos_embeddings = rearrange(pos_embeddings, 'b h w c -> b (h w) c')
        # vision = vision + rearrange(pos_embeddings, 'b h w c -> b (h w) c')
        # vision = vision[:, :, None, :]  # fake dim n to shape:  b, t, n, d, for compatibility with visual transformer
        return self.visual_mlp(vision), pos_embeddings

    def get_token_embeddings(self, tokenized_arrays, dtype, device, dynamic_size=True):
        """
        Get positional embeddings for the given sequence tokens.
        """

        tokenized_embedding_features, state_embedding_features, class_type_features, valid_mask = kutils.get_tokenized_features(tokenized_arrays)
        # convert to tensor
        state_embedding_features = torch.tensor(state_embedding_features, dtype=dtype, device=device)
        tokenized_embedding_features = torch.tensor(tokenized_embedding_features, dtype=torch.long, device=device)
        class_type_features = torch.tensor(class_type_features, dtype=torch.long, device=device)

        x, y, h, w, l, speed = state_embedding_features[..., [0]], state_embedding_features[..., [1]], state_embedding_features[..., [2]], state_embedding_features[..., [3]], state_embedding_features[..., [4]], state_embedding_features[..., [5]]

        token_embeddings = self.state_embed(get_sine_embedding_2d(x, y, h, w, l, speed, self.inital_pos_nembed).detach())
        pos_embeddings = self.pos_embed(get_sine_embedding_2d(x, y, h, None, None, None, self.inital_pos_nembed).detach()).detach()
        token_embeddings = self.token_embedding(tokenized_embedding_features) + token_embeddings
        token_embeddings = token_embeddings + self.class_embedding(class_type_features)

        # agent_index = torch.arange(token_embeddings.shape[2], dtype=torch.long, device=device)
        # token_embeddings = token_embeddings + self.agent_index_embedding(agent_index)[None, None]

        if dynamic_size:
            max_len = valid_mask.sum(axis=2).max()
        token_embeddings = token_embeddings[:, :, :max_len]
        pos_embeddings = pos_embeddings[:, :, :max_len]
        valid_mask = valid_mask[:, :, :max_len]

        return token_embeddings, valid_mask, pos_embeddings