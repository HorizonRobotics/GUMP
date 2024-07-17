from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type_v1_1 import VocabularyStateType, PositionalStateType
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils
from nuplan_extent.planning.training.modeling.models.utils import get_sine_embedding_2d


class GUMPEmbedder(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                 map_range=(-56, -56, 56, 56),
                 n_embd=768,
                 block_size=1024,
                 max_frames=100,
                 ):
        super().__init__()
        self.map_range = map_range
        self.n_embd = n_embd
        self.block_size = block_size
        self.max_frames = max_frames

        self.visual_mlp = nn.Sequential(
            nn.Linear(self.n_embd, 512),
            nn.GELU(),
            nn.Linear(512, self.n_embd))

        self.token_embedding = nn.Embedding(VocabularyStateType.PAD_TOKEN.vocal_size, self.n_embd)
        self.frame_pe = nn.Embedding(self.max_frames, self.n_embd)

        # self.w1 = nn.Linear(self.n_embd, self.n_embd)
        # self.w2 = nn.Linear(self.n_embd, self.n_embd)
        self.w1 = nn.Sequential(
            nn.Linear(self.n_embd, 256),
            nn.GELU(),
            nn.Linear(256, self.n_embd))

        self.w2 = nn.Sequential(
            nn.Linear(self.n_embd, 256),
            nn.GELU(),
            nn.Linear(256, self.n_embd))



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
        pos_embeddings = self.w1(get_sine_embedding_2d(x_map, y_map, None, None, None, self.n_embd).detach())
        vision = vision + rearrange(pos_embeddings, 'b h w c -> b (h w) c')
        # vision = vision[:, :, None, :]  # fake dim n to shape:  b, t, n, d, for compatibility with visual transformer
        return self.visual_mlp(vision)

    def get_token_embeddings(self, tokenized_arrays, latent_features, dtype, device, dynamic_forward=False, return_valid_mask=False):
        """
        Get positional embeddings for the given sequence tokens.
        """

        batch_size, seq_len, _ = tokenized_arrays.shape
        token_embeddings = torch.zeros((batch_size, self.block_size, self.n_embd), dtype=dtype, device=device)
        (
            ctrl_embedding_inds, ctrl_embedding_features_np, 
            query_embedding_inds, query_embedding_features_np, 
            state_embedding_inds, state_embedding_features_np,
            tokenized_embedding_features_np
        ) = gutils.get_tokenized_inds(tokenized_arrays, self.block_size)
        # convert to tensor
        ctrl_embedding_inds = torch.tensor(ctrl_embedding_inds, dtype=torch.long, device=device)
        query_embedding_inds = torch.tensor(query_embedding_inds, dtype=torch.long, device=device)
        state_embedding_inds = torch.tensor(state_embedding_inds, dtype=torch.long, device=device)
        ctrl_embedding_features = torch.tensor(ctrl_embedding_features_np, dtype=torch.long, device=device)
        query_embedding_features = torch.tensor(query_embedding_features_np, dtype=torch.long, device=device)
        state_embedding_features = torch.tensor(state_embedding_features_np, dtype=dtype, device=device)
        tokenized_embedding_features = torch.tensor(tokenized_embedding_features_np, dtype=torch.long, device=device)

        state_embeddings = self.w2(get_sine_embedding_2d(*state_embedding_features[:, :5].T[..., None], self.n_embd).detach())
        state_embeddings = self.token_embedding(tokenized_embedding_features).sum(dim=1) + state_embeddings

        token_embeddings[ctrl_embedding_inds[:, 0], ctrl_embedding_inds[:, 1], :] = self.token_embedding(ctrl_embedding_features).to(dtype)
        token_embeddings[query_embedding_inds[:, 0], query_embedding_inds[:, 1], :] = self.token_embedding(query_embedding_features).sum(dim=1).to(dtype)
        token_embeddings[state_embedding_inds[:, 0], state_embedding_inds[:, 1], :] = state_embeddings.to(dtype)

        frame_index = torch.tensor(gutils.get_frame_index(tokenized_arrays, self.block_size, self.max_frames), dtype=torch.long, device=device)
        assert (frame_index < (self.max_frames -1)).sum() == ctrl_embedding_inds.shape[0] + query_embedding_inds.shape[0] + state_embedding_inds.shape[0], \
            "shape mismatch between frame index and token embeddings."
        frame_index_pos_emb = self.frame_pe(frame_index)

        token_embeddings = token_embeddings + frame_index_pos_emb.to(dtype)
        if dynamic_forward:
            max_valid_len = torch.cat([ctrl_embedding_inds, query_embedding_inds, state_embedding_inds])[:, 1].max() + 1
            token_embeddings = token_embeddings[:, :max_valid_len, :]
        if return_valid_mask:
            valid_mask = torch.zeros((batch_size, self.block_size), dtype=torch.bool, device=device)
            valid_mask[torch.cat([ctrl_embedding_inds, query_embedding_inds, state_embedding_inds])[:, 0], torch.cat([ctrl_embedding_inds, query_embedding_inds, state_embedding_inds])[:, 1]] = True
            valid_mask = valid_mask[:, :token_embeddings.shape[1]]
            return token_embeddings, valid_mask
        return token_embeddings