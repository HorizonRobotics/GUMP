"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import copy
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many

from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.base_model import GPTConfig, GPTBase
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.output_token_sequence import OutputTokenSequence
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.rl_output_token_sequence import RLOutputTokenSequence

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
from torch.utils.checkpoint import checkpoint

class GPT(GPTBase):
    def __init__(self, config, temperature=1.0, top_k=None):
        super().__init__(config)
        self.temperature = temperature
        self.top_k = top_k
        self.max_num_newborn_object = 1

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        # assert all(k == 'dropout' for k in override_args)
        # from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            # 350M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # print("forcing vocab_size=50257, block_size=1024, bias=True")
        # config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        # config_args['block_size'] = 1024 # always 1024 for GPT model
        # checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        for k in override_args.keys():
            print(f"overriding {k} rate to {override_args[k]}")
            config_args[k] = override_args[k]
        print(config_args)
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        return model

    def forward(self, sequence_tokens, vision, embedding_manager, decoding_manager, generate_mode=False):
        # forward the GPT model itself
        x = embedding_manager.get_token_embeddings(sequence_tokens)
        x = self.transformer.drop(x)
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if i % self.visual_ca_ds_ratio == 0:
                x = self.transformer.hv[i // self.visual_ca_ds_ratio](x, vision)
            if i in decoding_manager.intermedia_layer_index_for_trajectory:
                decoding_manager.update_intermedia_feature(x, intermedia_layer_index = i, generate_mode=generate_mode)
        x = self.transformer.ln_f(x)  # (b, t, n_embd)
        return x

    def forward_train(self, sequence_tokens, vision, embedding_manager, decoding_manager):

        output_sequence_tokens = OutputTokenSequence(sequence_tokens)
        x = self(sequence_tokens, vision, embedding_manager, decoding_manager)
        # output_sequence_tokens.update_traffic_light(*decoding_manager.decoding_traffic_light(x, sequence_tokens))
        output_sequence_tokens.update_agents(*decoding_manager.decoding_agents(x, sequence_tokens, return_ego_only=False))
        output_sequence_tokens.update_normal_tokens(*decoding_manager.decoding_normal_tokens(x, sequence_tokens))
        output_sequence_tokens.update_trajectory(*decoding_manager.decoding_trajectory(x, sequence_tokens))

        # import pdb;pdb.set_trace()

        return output_sequence_tokens 

    def generate(self, sequence_tokens, generation_sequence_tokens, vision, embedding_manager, decoding_manager, next_ego_state=None):
        # import pdb;pdb.set_trace()
        generation_sequence_tokens.update_input(sequence_tokens)
        # step 1: add bos token
        generation_sequence_tokens.append_bos_token()

        # step 2: add traffic light tokens
        # if generation_sequence_tokens.pred_log_prob is None:
        iterate_set_module_merge_state(self.transformer, True)
        iterate_set_module_merge_state(decoding_manager, True)
        iterate_set_module_merge_state(embedding_manager, True)

        sequence_tokens = generation_sequence_tokens.get_forward_sequence_tokens()
        x = self(sequence_tokens, vision, embedding_manager, decoding_manager, generate_mode=True)
        generation_sequence_tokens.update_trajectory(*decoding_manager.decoding_trajectory(x, sequence_tokens, return_last=True))
        generation_sequence_tokens.append_tl_end_token()
        

        # step 3: add normal agent tokens
        # substep 3.1: precondition with trajectory tokens
        generation_sequence_tokens.precondition_normal_tokens(next_ego_state=next_ego_state)


        # substep 3.2: generate normal tokens
        sequence_tokens = generation_sequence_tokens.get_forward_sequence_tokens()
        x = self(sequence_tokens, vision, embedding_manager, decoding_manager, generate_mode=True)
        generation_sequence_tokens.update_agents(*decoding_manager.decoding_agents(x, sequence_tokens, return_last_frame=True), closeloop_training=False, no_ego=False)

        # step 4: add newborn agent tokens
        # substep 4.1: append newborn begin token
        generation_sequence_tokens.append_newborn_begin_token()
        generation_sequence_tokens.output_sequence_tokens.sort()
        iterate_set_module_merge_state(self.transformer, True)
        iterate_set_module_merge_state(decoding_manager, True)
        iterate_set_module_merge_state(embedding_manager, True)
        return generation_sequence_tokens


def iterate_set_module_merge_state(module, state):
    for child in module.modules():
        if hasattr(child, 'merged'):
            child.merged = state
            # print(child)
