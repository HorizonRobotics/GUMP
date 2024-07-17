import logging
import torch
import random
import numpy as np

from torch import nn
from typing import List, Dict

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from nuplan_extent.planning.training.modeling.models.utils.log_utils import render_and_save_features
from einops import rearrange, repeat
np.set_printoptions(precision=2, suppress=True)

logger = logging.getLogger(__name__)


class GUMPV1_1(TorchModuleWrapper):
    def __init__(
        self,
        image_encoder: nn.Module,
        tokenizer: nn.Module,
        embedder: nn.Module,
        transition_model: nn.Module,
        token_decoder: nn.Module,
        render: nn.Module,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        future_trajectory_sampling: TrajectorySampling,
        preloader: nn.Module = None,
        postprocessor: nn.Module = None,
        pretraining_path: str = None,
        num_paralell_scenario: int = 1,
        num_conditioned_frames: int = 4,
        downstream_task: str = 'scenario_extrapolation',
    ):
        """
        World Model V2, for fast training and inference. 
        support end to end learning.
        """
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self.preloader = preloader
        self.image_encoder = image_encoder
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.transition_model = transition_model
        self.token_decoder = token_decoder
        self.render = render
        self.postprocessor = postprocessor
        self._num_paralell_scenario = num_paralell_scenario
        self._num_conditioned_frames = num_conditioned_frames
        self._downstream_task = downstream_task

        self.pretraining_path = pretraining_path
        self.load_pretrained_weights()

    def set_vis_features(self,
                         is_vis_features: bool,
                         vis_features_path: str):
        """
        Set params for saving features, for visualization, only when simulation feature video callback is on.
        :param is_vis_features: whether to save features
        :param vis_features_path: path to save features
        """
        self._is_vis_features = is_vis_features
        self._vis_features_path = vis_features_path
        
    def load_pretrained_weights(self):
        logger.info(f'loading pretrained weights: {self.pretraining_path}')
        def match_incompatible_keys(checkpoint_state_dict, src, dst):
            filtered_state_dict = {}
            for name, param in checkpoint_state_dict.items():
                if name.startswith(src):
                    name = name[len(src):]
                filtered_state_dict[name] = param
            return filtered_state_dict
        
        def filter_incompatible_keys(model_state_dict, checkpoint_state_dict, name_prefix=''):
            filtered_state_dict = {}
            for name, param in checkpoint_state_dict.items():
                name = name_prefix + name
                if name in model_state_dict and param.shape == model_state_dict[name].shape:
                    filtered_state_dict[name] = param
                else:
                    logger.info(f"Skipping incompatible key: {name}")
            return filtered_state_dict
                
        
        if self.pretraining_path is not None:
            checkpoint = torch.load(self.pretraining_path, map_location='cpu')['state_dict']
            checkpoint = match_incompatible_keys(checkpoint, src='model.', dst='')
            
            checkpoint_not_found_num = 0
            model_not_found_num = 0
            for k,v in checkpoint.items():
                if k not in self.state_dict() or self.state_dict()[k].shape != v.shape:
                    logger.info(f'{k} not in model')
                    print(f'{k} not in model')
                    checkpoint_not_found_num += 1
            for k,v in self.state_dict().items():
                if k not in checkpoint or checkpoint[k].shape != v.shape:
                    logger.info(f'{k} not in checkpoint')
                    model_not_found_num += 1
            self.load_state_dict(filter_incompatible_keys(self.state_dict(), checkpoint), strict=False)
            logger.info(f'checkpoint not found num: {checkpoint_not_found_num}/{len(checkpoint)}, model not found num: {model_not_found_num}/{len(self.state_dict())}')
        # lora.mark_only_lora_as_trainable(self, bias='lora_only')
    
    
    def forward(self, input_features: FeaturesType, scenario=None) -> TargetsType:
        """
        Predict
        :param input_features: input features containing
        :return: targets: predictions from network
        """
        if self.preloader is not None:
            input_features = self.preloader(input_features)

        if self.training:
            return self.forward_train(input_features)
        else:     
            if self._downstream_task == 'scenario_extrapolation':
                return self.forward_inference(input_features, n_repeat=self._num_paralell_scenario, num_imagine_frames=16)
            elif self._downstream_task == 'planning':
                infer_results = self.forward_inference(input_features, n_repeat=self._num_paralell_scenario, num_imagine_frames=16)
                planning_results = self.postprocessor(infer_results)
                if self._is_vis_features:
                    render_and_save_features(planning_results, self._vis_features_path, bev_range=[-104., -104., 104., 104.])
                return planning_results
            elif self._downstream_task == 'rl_training':
                results = self.forward_train(input_features)
                return results
            elif self._downstream_task == 'e2e':
                infer_results = self.forward_inference(input_features, n_repeat=self._num_paralell_scenario, num_imagine_frames=16)
                return infer_results
            else:
                raise ValueError(f'Unknown downstream task: {self._downstream_task}')

    def forward_train(self, input_features: FeaturesType) -> Dict:
        """
        Forward pass for training
        :param input_features: input features
        :param targets: targets
        """
        # Encode image
        image_features = self.image_encoder(input_features)

        # Tokenize vectors
        tokenized_dict = self.tokenizer.forward_train(input_features['vector'])

        tokenized_arrays = tokenized_dict.get('tokenized_arrays', None)
        latent_features = tokenized_dict.get('latent_features', None)
        # Transition model
        pred_agent_logits, pred_agent_tokens, target_tokenized_state, pred_control_logits, target_ctrl_tokens, hidden = self.transition_model.forward_train(
            image_features=image_features,
            tokenized_arrays=tokenized_arrays,
            embedder=self.embedder,
            token_decoder=self.token_decoder,
            latent_features=latent_features,
            last_frame_only=self._downstream_task == 'rl_training'
        )
        return {
            'pred_agent_logits': pred_agent_logits,
            'pred_agent_tokens': pred_agent_tokens,
            'target_ctrl_tokens': target_ctrl_tokens,
            'pred_control_logits': pred_control_logits,
            'target_tokenized_state': target_tokenized_state,
            'tokenized_arrays': tokenized_arrays,
            'representation': hidden[-1]
        }

    def forward_inference(self, input_features: FeaturesType, n_repeat: int = 1, num_imagine_frames: int = 1) -> Dict:
        """
        Forward pass for inference
        :param input_features: input features
        :return: predictions from network
        """
        # Encode image
        image_features = self.image_encoder(input_features)

        # Tokenize vectors
        tokenized_dict = self.tokenizer.forward_inference(input_features['vector'])

        image_features = repeat(image_features, 'b l c -> (b n) l c', n=n_repeat)

        tokenized_dict = {k: repeat(arr, 'b l c -> (b n) l c', n=n_repeat) for k, arr in tokenized_dict.items()}
        tokenized_arrays = tokenized_dict.get('tokenized_arrays', None)
        latent_features = tokenized_dict.get('latent_features', None)
        gt_tokenized_arrays = tokenized_dict.get('gt_tokenized_arrays', None)

        # Transition model
        predicted_tokenized_arrays = self.transition_model.forward_inference_without_cache(
            image_features=image_features,
            tokenized_arrays=tokenized_arrays,
            embedder=self.embedder,
            token_decoder=self.token_decoder,
            render=self.render,
            latent_features=latent_features,
            num_imagine_frames=num_imagine_frames,
            num_conditioned_frames=self._num_conditioned_frames,
            update_initial_prompts=self._downstream_task == 'e2e'
        )
        tokenized_arrays = rearrange(tokenized_arrays, '(b n) l c -> b n l c', n=n_repeat)[:, 0:1, :, :]
        predicted_tokenized_arrays = rearrange(predicted_tokenized_arrays, '(b n) l c -> b n l c', n=n_repeat)
        if gt_tokenized_arrays is not None:
            gt_tokenized_arrays = rearrange(gt_tokenized_arrays, '(b n) l c -> b n l c', n=n_repeat)[:, 0:1, :, :]
        if self._downstream_task == 'e2e':
            return {
                'tokenized_arrays': tokenized_arrays,
                'predicted_tokenized_arrays': predicted_tokenized_arrays,
                'gt_tokenized_arrays': gt_tokenized_arrays,
            }
        return {
            'raster': input_features["raster"].data if hasattr(input_features["raster"], 'data') else input_features["raster"],
            'tokenized_arrays': tokenized_arrays,
            'predicted_tokenized_arrays': predicted_tokenized_arrays
        }
        



