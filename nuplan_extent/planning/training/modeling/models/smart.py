import logging
import torch
import random
import numpy as np

from torch import nn
from copy import deepcopy
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


class SMART(TorchModuleWrapper):
    def __init__(
        self,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        future_trajectory_sampling: TrajectorySampling,
        map_encoder: nn.Module,
        embedder: nn.Module = None,
        tokenizer: nn.Module = None,
        transition_model: nn.Module = None,
        dynamic_decoder: nn.Module = None,
        dynamic_render: nn.Module = None,
        pretraining_path: str = None,
    ):
        """
                                                                                map_encoder
                                                                                    |
                        dynamic_tokenizer              embedder              transition_model      dynamic_decoder          dynamic_render
        physical space (<=t-1) --> token/action space(t-1) --> latent space (t-1) --> latent space (t) --> token/action space (t) --> physical space (<=t)
                    |                                                                                                              |
                    ---------------------------------------------------------------------------------------------------------------                                     

        embedder: nn.Module, embed the tokenized features
        map_encoder: nn.Module, encode map elements into latent features
        dynamic_tokenizer: nn.Module, tokenize the dynamic elements, which need to be predicted autoregressive
        transition_model: nn.Module, core model of the world transition, transform from t-1 to t in latent space
        dynamic_decoder: nn.Module, decode the dynamic elements from latent space to token space
        dynamic_render: nn.Module, render the dynamic elements from tokenized space to physical space

        """
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self.embedder = embedder
        self.map_encoder = map_encoder
        self.tokenizer = tokenizer
        self.transition_model = transition_model
        self.dynamic_decoder = dynamic_decoder
        self.dynamic_render = dynamic_render

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
    
    
    def forward(self, input_features: FeaturesType, scenario=None) -> TargetsType:
        """
        Predict
        :param input_features: input features containing
        :return: targets: predictions from network
        """
        input_features = self.tokenizer.tokenize_data(input_features)

        if self.training:
            return self.forward_train(input_features)
        else:     
            return self.forward_inference(input_features)

    def forward_train(self, input_features: FeaturesType) -> Dict:
        """
        Forward pass for training
        :param input_features: input features
        :param targets: targets
        """
        # import cProfile
        # prof = cProfile.Profile()
        # prof.enable()

        # Encode map elements
        map_features = self.map_encoder(input_features)

        # Transition model
        transition_dict = self.transition_model(
            data=input_features,
            map_features=map_features,
            tokenizer=self.tokenizer,
            embedder=self.embedder,
            dynamic_decoder=self.dynamic_decoder,
            dynamic_render=self.dynamic_render,
        )
        # prof.disable()
        # prof.print_stats(sort='cumtime')
        # prof.dump_stats('/mnt/nas26/yihan01.hu/tmp/forward_train.prof')
        # import pdb; pdb.set_trace()
        return transition_dict

    def forward_inference(self, input_features: FeaturesType) -> Dict:
        """
        Forward pass for inference
        :param input_features: input features
        :return: predictions from network
        """
        # Encode map elements
        map_features = self.map_encoder(input_features)

        # Transition model
        transition_dict = self.transition_model(
            data=input_features,
            map_features=map_features,
            tokenizer=self.tokenizer,
            embedder=self.embedder,
            dynamic_decoder=self.dynamic_decoder,
            dynamic_render=self.dynamic_render,
        )
        prediction_dict = self.transition_model.inference(
            data=input_features,
            map_features=deepcopy(map_features),
            tokenizer=self.tokenizer,
            embedder=self.embedder,
            dynamic_decoder=self.dynamic_decoder,
            dynamic_render=self.dynamic_render,
            n_repeat=32,
        )
        transition_dict.update(map_features)
        transition_dict.update(prediction_dict)
        return transition_dict
        



