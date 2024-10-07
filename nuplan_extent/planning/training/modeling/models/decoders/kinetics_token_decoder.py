from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
import loralib as lora
import torch.nn.functional as F
from einops import rearrange, repeat
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.kinetics_state_type import KineticsVocabularyStateType
import nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils as kutils
from copy import deepcopy

class KineticsTokenDecoder(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                n_embed=768,
                num_rnn_layers=1,
                num_agent_attributes=1,
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
        self.lm_head = nn.Sequential(
            nn.Linear(self.n_embed, KineticsVocabularyStateType.CONTROL.vocal_size, bias=False)
        )

    def decoding_agents(self, embedder, output_features, tokenized_arrays, last_frame_only=False) -> Dict:
        """
        Dummy forward
        """
        target_tokenized_features = kutils.get_agent_target_features(deepcopy(tokenized_arrays))
        target_tokenized_state = torch.tensor(target_tokenized_features, dtype=torch.long).to(output_features.device)
        
        if last_frame_only:
            output_features = output_features[:, [-1], :, :]

        bs, frame_len, agent_len, _ = output_features.shape
        output_features = output_features.contiguous().view(bs * frame_len * agent_len, -1)

        logits = self.lm_head(output_features)
        agent_tokens = self.sampling_logits(logits, sampling_mask=None)
        agent_tokens = agent_tokens.contiguous().view(bs, frame_len, agent_len, 1)

        logits = logits.contiguous().view(bs, frame_len, agent_len, -1)
        target_tokenized_state = target_tokenized_state[:, :, :agent_len, :]
        return logits, agent_tokens, target_tokenized_state, output_features

    def sampling_logits(self, logits, sampling_mask=None):
        # apply softmax to convert to probabilities
        logits = torch.nan_to_num(logits, nan=-10)
        logits = logits / self.temperature

        # logits[:, :33] = -float('Inf')
        # logits[:, -33:] = -float('Inf')
        logits[:, 0] = -float('Inf') # ignore the blank token
        # optionally crop the logits to only the top k options
        if sampling_mask is not None:
            logits[~sampling_mask] = -float('Inf')
        if self.topk is not None:
            v, _ = torch.topk(logits, min(self.topk, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # import pdb; pdb.set_trace()
        # print(v)
        # sample from the distribution
        tokens = torch.multinomial(probs, num_samples=1)
        return tokens
