from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
import loralib as lora
import torch.nn.functional as F
from einops import rearrange, repeat
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type_v1_1 import VocabularyStateType
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils

class GUMPTokenDecoder(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                n_embed=768,
                num_rnn_layers=1,
                num_agent_attributes=3,
                block_size=1024,
                temperature=1.0,
                topk=40):
        super().__init__()
        self.n_embed = n_embed
        self.num_rnn_layers = num_rnn_layers
        self.num_agent_attributes = num_agent_attributes
        self.block_size = block_size
        self.temperature = temperature
        self.topk = topk
        self.init_agents_head()

    def init_agents_head(self):
        self.agent_rnn_decoder = nn.GRU(
            input_size=self.n_embed,
            hidden_size=self.n_embed,
            num_layers=self.num_rnn_layers)
        self.lm_head = nn.Sequential(
            nn.Linear(self.n_embed, VocabularyStateType.AGENTS.vocal_size, bias=False)
            # lora.Linear(self.n_embd, VocabularyStateType.AGENTS.vocal_size, r=8, bias=False)
        )
        self.agents_pos_embedding = nn.Embedding(self.num_agent_attributes, self.n_embed)

    def decoding_controls(self, embedder, output_features, tokenized_arrays) -> Dict:
        """
        Dummy forward
        """
        ctrl_inds, target_ctrl_tokens = gutils.get_ctrl_target_inds(tokenized_arrays, self.block_size)
        target_ctrl_tokens = torch.tensor(target_ctrl_tokens, dtype=torch.long).to(output_features.device)
        ctrl_inds = torch.tensor(ctrl_inds, dtype=torch.long).to(output_features.device)
        pred_control_logits = self.lm_head(output_features[ctrl_inds[:, 0], ctrl_inds[:, 1], :])
        return pred_control_logits, target_ctrl_tokens

    def decoding_agents(self, embedder, output_features, tokenized_arrays, last_frame_only=False) -> Dict:
        """
        Dummy forward
        """
        query_embedding_inds, state_embedding_features = gutils.get_agent_target_inds(tokenized_arrays, self.block_size, last_frame_only=last_frame_only)
        query_embedding_inds = torch.tensor(query_embedding_inds, dtype=torch.long).to(output_features.device)
        target_tokenized_state = torch.tensor(state_embedding_features, dtype=torch.long).to(output_features.device)
        
        target_tokenized_state = target_tokenized_state[:, :self.num_agent_attributes]
        hidden = output_features[query_embedding_inds[:, 0], query_embedding_inds[:, 1], :]
        hidden = repeat(hidden, 'b c -> l b c', l=self.num_rnn_layers)
        pred_agent_logits, pred_agent_tokens = [], []

        input = torch.zeros_like(hidden[0:1])
        for i in range(self.num_agent_attributes):
            pos_embed = self.agents_pos_embedding(torch.tensor([[i]], device=hidden.device).long())
            input = input + pos_embed
            output, hidden = self.agent_rnn_decoder(input, hidden.contiguous())
            logits = self.lm_head(output)
            pred_agent_logits.append(logits)
            assert output.shape[0] == 1

            sampling_mask = self.get_sampling_mask(attr_index=i, num_agents=logits.shape[1])
            sampling_mask = torch.from_numpy(sampling_mask).to(logits.device)

            if self.training:
                agent_tokens = target_tokenized_state.T[i]
                pred_agent_tokens_cur = self.sampling_logits(logits[0], sampling_mask=sampling_mask)[..., 0]
                pred_agent_tokens.append(pred_agent_tokens_cur)
            else:
                agent_tokens = self.sampling_logits(logits[0], sampling_mask=sampling_mask)[..., 0]
                pred_agent_tokens.append(agent_tokens)
            token_embed = embedder.token_embedding(agent_tokens)[None, ...]
            input = token_embed
        pred_agent_logits = torch.cat(pred_agent_logits, dim=0).permute([1, 0, 2])
        pred_agent_tokens = torch.stack(pred_agent_tokens, dim=-1)

        # assert hidden.shape[1] == output_features.shape[0]
        return pred_agent_logits, pred_agent_tokens, target_tokenized_state, hidden

    def sampling_logits(self, logits, sampling_mask=None):
        # apply softmax to convert to probabilities
        logits = torch.nan_to_num(logits, nan=-10)
        temperature = self.temperature
        logits = logits[:, :] / temperature
        # optionally crop the logits to only the top k options
        if sampling_mask is not None:
            logits[~sampling_mask] = -float('Inf')
        if self.topk is not None:
            v, _ = torch.topk(logits, min(self.topk, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')


        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        tokens = torch.multinomial(probs, num_samples=1)
        return tokens

    def get_sampling_mask(self, attr_index, num_agents=None):
        # Mapping of attribute indices to the VocabularyStateType methods
        attr_to_type = {
            0: VocabularyStateType.X,
            1: VocabularyStateType.Y,
            2: VocabularyStateType.HEADING,
        }

        # Check if the attribute index is valid
        if attr_index not in attr_to_type:
            raise NotImplementedError("Attribute index not implemented")

        # Get the sampling mask for the specified attribute index
        sampling_mask = attr_to_type[attr_index].get_sampling_mask()

        # If agent_tokens is not None, tile the sampling mask accordingly
        if num_agents is not None:
            sampling_mask = np.tile(sampling_mask[None], (num_agents, 1))
        return sampling_mask

