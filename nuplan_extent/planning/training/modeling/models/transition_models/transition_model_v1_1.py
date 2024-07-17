import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
from einops import rearrange
from functools import partial
from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.visual_gpt_v2 import GPTV2
from nuplan_extent.planning.training.modeling.models.modules.generative_model.llama3_model import Llama3Model, ModelArgs
# from nuplan_extent.planning.training.modeling.models.modules.generative_model.mamba_model import MambaModel, MambaConfig

from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.base_model import GPTConfig
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type_v1_1 import VocabularyStateType, PositionalStateType
from nuplan_extent.planning.training.modeling.models.transition_models.base_transition_model import BaseTransitionModel
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils
import numpy as np
np.set_printoptions(precision=2, suppress=True)

class TransitionModelV1_1(nn.Module):
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
                 top_k: int = 40):
        super().__init__()
        meta_vocab_size = VocabularyStateType.PAD_TOKEN.vocal_size
        temperature = float(os.environ.get('TEMPERATURE', temperature))
        top_k = int(os.environ.get('TOPK', top_k))

        self.use_sliding_window = True # True to enable sliding window decoding
        self.enable_qkv_cache = False  # True to enable qkv cache for llama3
        ## Init model
   
        if init_from == "llama3":
            self.model_selection = 'llama3'
            model_args: ModelArgs = ModelArgs(
                dim = n_embd,
                n_layers = n_layer,
                n_heads = n_head,
                n_kv_heads = None,
                vocab_size = meta_vocab_size,
                multiple_of = 256,  # make SwiGLU hidden layer size multiple of large power of 2
                ffn_dim_multiplier = None,
                norm_eps = 1e-5,
                rope_theta = 500000,
                max_batch_size = 80,
                max_seq_len = 3000,
            )
            model = Llama3Model(model_args)
            self.transition_model = model
        elif init_from.startswith('gpt2'):
            self.model_selection = 'gpt2'
            model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                            bias=bias, vocab_size=meta_vocab_size, dropout=dropout)  # start with model_args from command line
            if init_from == 'scratch':
                # init a new model from scratch
                print("Initializing a new model from scratch")
                # determine the vocab size we'll use for from-scratch training
                model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
                gptconf = GPTConfig(**model_args)
                model = GPTV2(gptconf)
            elif init_from.startswith('gpt2'):
                print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
                # initialize from OpenAI GPT-2 weights
                override_args = dict(
                    dropout=dropout,
                    vocab_size=meta_vocab_size,
                    block_size=block_size)
                model = GPTV2.from_pretrained(init_from, override_args)
                # read off the created config params, so we can store them into
                # checkpoint correctly
                for k in ['n_layer', 'n_head', 'n_embd', 'bias']:
                    model_args[k] = getattr(model.config, k)
            self.transition_model = model
            self.transition_model.temperature = temperature
            self.transition_model.top_k = top_k
        else:
            raise ValueError(f"Invalid init_from: {init_from}")

        self.block_size = block_size
        self.n_embd = n_embd
        self.temperature = temperature
        self.top_k = top_k

    def forward_train(
            self,
            image_features,
            tokenized_arrays,
            embedder,
            token_decoder,
            latent_features=None,
            last_frame_only=False):
        """
        Forward pass for training.

        Args:
            image_features (Tensor): The input image features.
            tokenized_arrays (Tensor): The tokenized input arrays.
            embedder (Embedder): The embedder object used for embedding.
            token_decoder (TokenDecoder): The token decoder object used for decoding.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The predicted agent logits, predicted agent tokens, and target tokenized state.
        """
        image_features = embedder.get_vision_embeddings(image_features)
        token_embeddings = embedder.get_token_embeddings(
            tokenized_arrays, latent_features, image_features.dtype, image_features.device, dynamic_forward=True)
        output_features = self.transition_model.forward(
            token_embeddings, image_features)
        pred_agent_logits, pred_agent_tokens, target_tokenized_state, hidden = token_decoder.decoding_agents(
            embedder, output_features, tokenized_arrays, last_frame_only=last_frame_only)
        pred_control_logits, target_ctrl_tokens = token_decoder.decoding_controls(
            embedder, output_features, tokenized_arrays)
        
        return pred_agent_logits, pred_agent_tokens, target_tokenized_state, pred_control_logits, target_ctrl_tokens, hidden

    def forward_inference(
            self,
            image_features,
            tokenized_arrays,
            embedder,
            token_decoder,
            render,
            latent_features=None,
            num_imagine_frames=16,
            num_conditioned_frames=4):
        """
        Forward pass for inference.

        Args:
            image_features (Tensor): The input image features.
            tokenized_arrays (List[List[int]]): The tokenized arrays.
            embedder (Embedder): The embedder object for token and vision embeddings.
            token_decoder (TokenDecoder): The token decoder object for decoding agents.
            render (Callable): The rendering function for detokenizing and rendering.
            num_imagine_frames (int, optional): The number of frames to imagine. Defaults to 8.
            num_conditioned_frames (int, optional): The number of conditioned frames. Defaults to 4.

        Returns:
            List[List[int]]: The updated tokenized arrays after imagining frames.
        """
        image_features = embedder.get_vision_embeddings(image_features)

        hist_tokenized_arrays = gutils.extract_history_data(deepcopy(tokenized_arrays), num_conditioned_frames)

        # cached forward, for kv cache initialization
        token_embeddings, valid_mask = embedder.get_token_embeddings(
            hist_tokenized_arrays, image_features.dtype, image_features.device, dynamic_forward=True, return_valid_mask=True)
        # clear kv cache
        self.transition_model.clear_kv_cache()
        output_features = self.transition_model.cached_forward(
            token_embeddings, image_features, valid_mask=valid_mask)

        # use last frame as precondition
        last_tokenized_arrays = gutils.extract_last_frame_data(
            deepcopy(hist_tokenized_arrays), skip_nb_bos=True)
        last_tokenized_arrays = gutils.add_one_to_frame_index(last_tokenized_arrays)
        
        for i in range(num_imagine_frames):
            token_embeddings, valid_mask = embedder.get_token_embeddings(
                last_tokenized_arrays, image_features.dtype, image_features.device, dynamic_forward=True, return_valid_mask=True)
            # 1. something wrong with the kv cat cache with padding
            output_features = self.transition_model.cached_forward(
                token_embeddings, image_features)
            pred_agent_logits, pred_agent_tokens, _, _= token_decoder.decoding_agents(
                embedder, output_features, last_tokenized_arrays, last_frame_only=True)

            # update last frame with predicted agent tokens
            last_tokenized_arrays = render.update_last_frame_data(
                last_tokenized_arrays, pred_agent_tokens.cpu().detach().numpy())

            # detokenize and render
            last_tokenized_arrays = render(last_tokenized_arrays)
            
            # mark as generated
            last_tokenized_arrays = gutils.mark_as_generated(last_tokenized_arrays)
            
            # save imagined frame to history
            hist_tokenized_arrays = gutils.update_history_data(
                hist_tokenized_arrays, last_tokenized_arrays)
            token_embeddings, valid_mask = embedder.get_token_embeddings(
                last_tokenized_arrays, image_features.dtype, image_features.device, dynamic_forward=True, return_valid_mask=True)
            output_features = self.transition_model.cached_forward(
                token_embeddings, image_features, valid_mask=valid_mask)

            last_tokenized_arrays = gutils.add_one_to_frame_index(last_tokenized_arrays)
        return hist_tokenized_arrays


    def forward_inference_without_cache(
            self,
            image_features,
            tokenized_arrays,
            embedder,
            token_decoder,
            render,
            latent_features=None,
            num_imagine_frames=16,
            num_conditioned_frames=4,
            update_initial_prompts=False):
        """
        Forward pass for inference.

        Args:
            image_features (Tensor): The input image features.
            tokenized_arrays (List[List[int]]): The tokenized arrays.
            embedder (Embedder): The embedder object for token and vision embeddings.
            token_decoder (TokenDecoder): The token decoder object for decoding agents.
            render (Callable): The rendering function for detokenizing and rendering.
            num_imagine_frames (int, optional): The number of frames to imagine. Defaults to 8.
            num_conditioned_frames (int, optional): The number of conditioned frames. Defaults to 4.

        Returns:
            List[List[int]]: The updated tokenized arrays after imagining frames.
        """
        image_features = embedder.get_vision_embeddings(image_features)
        
        hist_tokenized_arrays = gutils.extract_history_data(
            deepcopy(tokenized_arrays), num_conditioned_frames)
        
        keep_visual_cache = True
        for i in range(num_imagine_frames):
            # use last frame as precondition
            last_tokenized_arrays = gutils.extract_last_frame_data(deepcopy(hist_tokenized_arrays))
            last_tokenized_arrays = gutils.add_one_to_frame_index(last_tokenized_arrays)
                # save imagined frame to history
            input_tokenized_arrays = gutils.update_history_data(
                deepcopy(hist_tokenized_arrays), last_tokenized_arrays)
                
            if self.use_sliding_window:
                input_tokenized_arrays = gutils.extract_last_frame_data(input_tokenized_arrays, last_num_frame=num_conditioned_frames, skip_nb_bos=False)
                
            # cached forward, for kv cache initialization
            token_embeddings = embedder.get_token_embeddings(
                input_tokenized_arrays, latent_features, image_features.dtype, image_features.device, dynamic_forward=True)
            # out of block size
            if token_embeddings.shape[1] >= self.block_size:
                break
            # clear kv cache
            if self.model_selection == 'llama3':
                output_features = self.transition_model.forward(token_embeddings, image_features, enable_cache=self.enable_qkv_cache, visual_cache=keep_visual_cache)
            else:
                output_features = self.transition_model.forward(token_embeddings, image_features)

            pred_agent_logits, pred_agent_tokens, _, _ = token_decoder.decoding_agents(
                embedder, output_features, input_tokenized_arrays, last_frame_only=True)
            # update last frame with predicted agent tokens
            last_tokenized_arrays = render.update_last_frame_data(
                last_tokenized_arrays, pred_agent_tokens.cpu().detach().numpy())
                
            # detokenize and render
            last_tokenized_arrays = render(last_tokenized_arrays, hist_tokenized_arrays)
            # print(last_tokenized_arrays[0, ~np.isnan(last_tokenized_arrays[0]).any(axis=-1), :5])
                
            last_tokenized_arrays = gutils.mark_as_generated(last_tokenized_arrays)
            hist_tokenized_arrays = gutils.update_history_data(
                hist_tokenized_arrays, last_tokenized_arrays)
            
            keep_visual_cache = False
        # self.transition_model.clear_kv_cache()
        return hist_tokenized_arrays