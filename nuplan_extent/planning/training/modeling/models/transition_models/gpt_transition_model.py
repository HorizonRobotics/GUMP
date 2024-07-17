import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
from einops import rearrange
from functools import partial
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType


from nuplan_extent.planning.training.modeling.models.transition_models.base_transition_model import BaseTransitionModel
from nuplan_extent.planning.training.modeling.models.modules.transition_manager.embedding_manager import EmbeddingManager
from nuplan_extent.planning.training.modeling.models.modules.transition_manager.decoding_manager import DecodingManager
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.generation_token_sequence import GenerationTokenSequence

class GPTTransitionModel(BaseTransitionModel):
    def __init__(self,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_embd: int = 768,
                 block_size: int = 1024,
                 bias: bool = True,
                 dropout: float = 0.1,
                 frame_dropout_rate: float = 0.2,
                 init_from: str = 'gpt2',
                 temperature: float = 1.1,
                 top_k: int = 40,
                 map_range=(-56, -56, 56, 56),
                 num_past_steps: int = 4,
                 intermedia_layer_index_for_trajectory=[5, 11, 17, 23],
                 rl_training: bool = False):
        meta_vocab_size = VocabularyStateType.PAD_TOKEN.vocal_size
        super().__init__(n_layer=n_layer,
                         n_head=n_head,
                         n_embd=n_embd,
                         block_size=block_size,
                         meta_vocab_size=meta_vocab_size,
                         dropout=dropout,
                         bias=bias,
                         init_from=init_from,
                         temperature=temperature,
                         top_k=top_k)
        self.frame_dropout_rate = frame_dropout_rate
        self.embedding_manager = EmbeddingManager(
            map_range=map_range,
            n_embd=n_embd
        )
        self.decoding_manager = DecodingManager(
            n_embd=n_embd,
            num_rnn_layers=2,
            embedding_manager=self.embedding_manager,
            topk=top_k,
            temperature=temperature,
            num_agent_attributes=VocabularyStateType.PAD_TOKEN.num_agent_attributes,
            num_max_traffic_light=64,
            intermedia_layer_index_for_trajectory=intermedia_layer_index_for_trajectory
        )
        self.num_past_steps = num_past_steps
        self.temperature = temperature
        self.top_k = top_k
        self.rl_training = rl_training
        
        
    def forward(self, input):
        sequence_tokens = input['sequence_tokens']
        sequence_tokens.update_block_size(self.block_size)
 
        # vision preprocessing
        vision = self.embedding_manager.get_vision_embeddings(input)
        # get token embeddings, forward to gpt model

        transitional_output = self.transition_model.forward_train(
            sequence_tokens,
            vision,
            self.embedding_manager,
            self.decoding_manager
        )

        return {
            'transitional_output': transitional_output
        }
        
    def imagine(self, input, num_imagined_frames=1, max_input_token_size=1400, start_local_index=0, end_local_index=2, next_ego_state=None):
        sequence_tokens = input['sequence_tokens']
        sequence_tokens.update_block_size(self.block_size)
        sequence_tokens.set_training_state(self.training)
        sequence_tokens.assign_position()
        vision = self.embedding_manager.get_vision_embeddings(input)
        # only keep the frames from past to the current frame [0, 1, 2, 3]
        sampled_sequence_tokens = sequence_tokens.sample_frame_segment(start_local_index=start_local_index, end_local_index=end_local_index, max_input_token_size=max_input_token_size)
        all_seqeuence_tokens = sampled_sequence_tokens
        
        generation_sequence_tokens = GenerationTokenSequence(sampled_sequence_tokens)
        for i in range(num_imagined_frames):
            sampled_sequence_tokens.assign_position()
            if sampled_sequence_tokens.get_max_token_size() > 1600:
                break
            generation_sequence_tokens = self.transition_model.generate(
                sampled_sequence_tokens,
                generation_sequence_tokens,
                vision,
                self.embedding_manager,
                self.decoding_manager,
                next_ego_state=next_ego_state[:, i] if next_ego_state is not None else None,
            )
            # import pdb;pdb.set_trace()
            imagine_seqeuence_tokens = generation_sequence_tokens.output_sequence_tokens
            imagine_seqeuence_tokens.mark_as_imagined()

            all_seqeuence_tokens = all_seqeuence_tokens + imagine_seqeuence_tokens
            sampled_sequence_tokens = sampled_sequence_tokens + imagine_seqeuence_tokens
            sampled_sequence_tokens = sampled_sequence_tokens.sample_frame_segment(start_local_index=1, end_local_index=4, max_input_token_size=max_input_token_size) # remove first frame
        # import pdb;pdb.set_trace()
        return {
            'all_seqeuence_tokens': all_seqeuence_tokens
        }

