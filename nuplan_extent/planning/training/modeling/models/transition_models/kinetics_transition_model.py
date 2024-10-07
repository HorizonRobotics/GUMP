import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
from einops import rearrange
from functools import partial
from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.visual_gpt_v2 import GPTV2
from nuplan_extent.planning.training.modeling.models.modules.generative_model.kinetics_attention import Kinetics_GUMP, ModelArgs

from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.base_model import GPTConfig
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType
import numpy as np
np.set_printoptions(precision=2, suppress=True)

class KineticsTransitionModel(nn.Module):
    def __init__(self,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_embd: int = 768,
                 block_size: int = 1024,
                 temperature: float = 1.1,
                 n_spatial_layer: int = 2,
                 n_temporal_layer: int = 2,
                 top_k: int = 40):
        super().__init__()
        meta_vocab_size = VocabularyStateType.PAD_TOKEN.vocal_size
        temperature = float(os.environ.get('TEMPERATURE', temperature))
        top_k = int(os.environ.get('TOPK', top_k))

        model_args: ModelArgs = ModelArgs(
            dim = n_embd,
            n_layers = n_layer,
            n_spatial_layer = n_spatial_layer,
            n_temporal_layer = n_temporal_layer,
            n_heads = n_head,
            n_kv_heads = None,
            vocab_size = meta_vocab_size,
            multiple_of = 256,  # make SwiGLU hidden layer size multiple of large power of 2
            ffn_dim_multiplier = None,
            norm_eps = 1e-5,
            rope_theta = 500000,
            max_batch_size = 80,
            max_seq_len = 512,
        )
        model = Kinetics_GUMP(model_args)
        self.transition_model = model

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
            latent_features,
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
        image_features, pos_embeddings_vision = embedder.get_vision_embeddings(image_features)
        token_embeddings, valid_mask, pos_embeddings_agent = embedder.get_token_embeddings(
            tokenized_arrays, image_features.dtype, image_features.device)

        output_features = self.transition_model.forward_train(
            token_embeddings, image_features, valid_mask=valid_mask, pos_embeddings_vision=pos_embeddings_vision, pos_embeddings_agent=pos_embeddings_agent)

        pred_agent_logits, pred_agent_tokens, target_tokenized_state, hidden = token_decoder.decoding_agents(
            embedder, output_features, tokenized_arrays, last_frame_only=last_frame_only)
        return pred_agent_logits, pred_agent_tokens, target_tokenized_state, None, None, hidden, valid_mask

    def forward_inference_without_cache(
            self,
            image_features,
            tokenized_arrays,
            embedder,
            token_decoder,
            render,
            latent_features,
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
        image_features, pos_embeddings_vision = embedder.get_vision_embeddings(image_features)
        hist_tokenized_arrays = deepcopy(tokenized_arrays[:, :num_conditioned_frames, :, :])

        use_sliding_window = True
        
        for i in range(num_imagine_frames):

            input_tokenized_arrays = deepcopy(hist_tokenized_arrays)
            if use_sliding_window:
                input_tokenized_arrays = deepcopy(hist_tokenized_arrays[:, -num_conditioned_frames:, :, :])

            token_embeddings, valid_mask, pos_embeddings_agent = embedder.get_token_embeddings(
                input_tokenized_arrays, image_features.dtype, image_features.device)

            output_features = self.transition_model.forward_inference(token_embeddings, image_features, valid_mask=valid_mask, pos_embeddings_vision=pos_embeddings_vision, pos_embeddings_agent=pos_embeddings_agent)

            pred_agent_logits, pred_agent_tokens, _, _ = token_decoder.decoding_agents(
                embedder, output_features, hist_tokenized_arrays, last_frame_only=True)

            # update last frame with predicted agent tokens
            # update hist action for next frame embedding
            hist_tokenized_arrays[:, -1:, :, :] = render.update_last_frame_data(
                input_tokenized_arrays[:, [-1], :, :], pred_agent_tokens.cpu().detach().numpy())
                
            last_tokenized_arrays = render.update_last_frame_data(
                deepcopy(input_tokenized_arrays[:, [-1], :, :]), pred_agent_tokens.cpu().detach().numpy())
                
            # detokenize and render
            last_tokenized_arrays = render(last_tokenized_arrays)

            # Update history tokenized array
            hist_tokenized_arrays = np.concatenate((hist_tokenized_arrays, last_tokenized_arrays), axis=1)

        return hist_tokenized_arrays

    def forward_inference_test(
            self,
            image_features,
            tokenized_arrays,
            embedder,
            token_decoder,
            render,
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
        hist_tokenized_arrays = deepcopy(tokenized_arrays[:, :num_conditioned_frames, :, :])

        # [1 1 1 1]

        for i in range(num_imagine_frames):
            # copy gt control tokens
            last_tokenized_arrays = deepcopy(hist_tokenized_arrays[:, [-1]])
            last_tokenized_arrays[:, -1, :, [0, 15]] = tokenized_arrays[:, num_conditioned_frames + i - 1, :, [0,15]]
            # [1 1 1 1] [new]
            # detokenize and render
            last_tokenized_arrays = render(last_tokenized_arrays)
            # [1 1 1 1] [1]
            # import pdb; pdb.set_trace()
            valid_control_mask = tokenized_arrays[:, num_conditioned_frames + i - 1, :, 20]
            last_tokenized_arrays = np.where(valid_control_mask[:, None, :, None], last_tokenized_arrays, tokenized_arrays[:, num_conditioned_frames + i, :, :][:, None])
            
            # Update history tokenized array
            hist_tokenized_arrays = np.concatenate((hist_tokenized_arrays, last_tokenized_arrays), axis=1)
        return hist_tokenized_arrays