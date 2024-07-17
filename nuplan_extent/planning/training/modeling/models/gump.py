import logging
from typing import List

import torch
import random
from torch import nn
import numpy as np
from copy import deepcopy

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from nuplan_extent.planning.training.preprocessing.features.tensor import Tensor
from nuplan_extent.planning.training.modeling.models.utils.log_utils import render_and_save_features
import loralib as lora

from einops import rearrange

logger = logging.getLogger(__name__)


class GUMP(TorchModuleWrapper):
    def __init__(
        self,
        encoders: List[nn.Module],
        tokenizers: List[nn.Module],
        transition_models: List[nn.Module],
        post_processors: List[nn.Module],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        future_trajectory_sampling: TrajectorySampling,
        checkpoint_path: str = None,
        downstream_task: str = 'sim_agents',
    ):
        """
        World model
        """
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self.encoders = nn.ModuleList(encoders)
        self.tokenizers = nn.ModuleList(tokenizers)
        self.transition_models = nn.ModuleList(transition_models)
        self.post_processors = nn.ModuleList(post_processors)

        # prarams for saving features, for visualization
        # will be alternated at simulation callback
        self._is_vis_features = False
        self._vis_features_path = None
        self.subtask = downstream_task

        self.checkpoint_path = checkpoint_path


    def load_pretrained_weights(self):
        print('loading pretrained weights:', self.checkpoint_path)
        def match_incompatible_keys(checkpoint_state_dict, src, dst):
            filtered_state_dict = {}
            for name, param in checkpoint_state_dict.items():
                if name.startswith(src):
                    name = name[len(src):]
                filtered_state_dict[name] = param
            return filtered_state_dict
        
        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')['state_dict']
            checkpoint = match_incompatible_keys(checkpoint, src='model.', dst='')
            
            checkpoint_not_found_num = 0
            model_not_found_num = 0
            for k,v in checkpoint.items():
                if k not in self.state_dict() or self.state_dict()[k].shape != v.shape:
                    print(f'{k} not in model')
                    checkpoint_not_found_num += 1
            # import pdb; pdb.set_trace()
            for k,v in self.state_dict().items():
                if k not in checkpoint or checkpoint[k].shape != v.shape:
                    print(f'{k} not in checkpoint')
                    model_not_found_num += 1
            self.load_state_dict(checkpoint, strict=False)
            print(f'checkpoint not found num: {checkpoint_not_found_num}/{len(checkpoint)}, model not found num: {model_not_found_num}/{len(self.state_dict())}')
        lora.mark_only_lora_as_trainable(self, bias='lora_only')

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

    def forward(self, input_features: FeaturesType) -> TargetsType:
        """
        Predict
        :param input_features: input features containing
        :return: targets: predictions from network
        """
        encoder_features = {}
        tokenized_features = {}
        transitional_features = {}

        feature_dict = {}
        feature_dict.update(input_features)

        with torch.no_grad():
            for tokenizer in self.tokenizers:
                tokenized_feature = tokenizer(feature_dict)
                tokenized_features.update(tokenized_feature)
                feature_dict.update(tokenized_feature)

        for encoder in self.encoders:
            encoder_feature = encoder(feature_dict)
            encoder_features.update(encoder_feature)
            feature_dict.update(encoder_feature)
        
        feature_dict['sequence_tokens'] = feature_dict['sequence_tokens'].sample_frame_segment(start_local_index=0, end_local_index=99999, max_input_token_size=9999999)
        # still, we need to do training pipeline at inference, to calculate val loss
        feature_dict = self.training_pipeline(feature_dict)
        # inference pipeline
        if not self.training:
            if self.subtask == 'planning':
                feature_dict = self.mcts_expansion(feature_dict)
            elif self.subtask == 'sim_agents':
                # Waymo Sim Agents rollout -> 32 parallel worlds, 8s into future
                all_rollouts = self.sim_agents_batched_rollout(feature_dict)
                feature_dict['sim_agents_rollouts'] = all_rollouts
            else:
                feature_dict.update(self.imagine(feature_dict, num_imagined_frames=16))
            
        for k in list(feature_dict.keys()):
            if isinstance(feature_dict[k], torch.Tensor):
                feature_dict[k] = Tensor(data=feature_dict[k])
            else:
                continue
        return feature_dict
    
    def training_pipeline(self, feature_dict):
        transitional_features = {}
        feature_dict['sequence_tokens'].set_training_state(True)
        for transition_model in self.transition_models:
            transitional_feature = transition_model(feature_dict)
            transitional_features.update(transitional_feature)
            feature_dict.update(transitional_feature)
        feature_dict['sequence_tokens'].set_training_state(self.training)
        return feature_dict
    
    def planning(self, feature_dict, num_imagined_frames=8):
        tmp_feature_dict = feature_dict.copy()

        for transition_model in self.transition_models:
            mcts_features = transition_model.imagine(
                tmp_feature_dict,
                num_imagined_frames=num_imagined_frames,
                next_ego_state=None,
                start_local_index=0,
                end_local_index=3,
                )
            tmp_feature_dict.update(mcts_features)

        # FOR KUN.LI. extract next state here
        all_seqeuence_tokens = tmp_feature_dict['all_seqeuence_tokens']
        # suppose past frames are 0,1; current is 2, so next is 3, we slice next state here
        next_frame_sequence = all_seqeuence_tokens.data[0].sample_frame_segment(start_local_index=4, end_local_index=12)
        next_ego_agents = next_frame_sequence.get_ego_agent_token()  # list of agents, refer to class AgentToken
        trajectory = np.zeros((1, 8, 3))
        for i in range(min(len(next_ego_agents), 8)):
            trajectory[0, i, 0] = next_ego_agents[i].x
            trajectory[0, i, 1] = next_ego_agents[i].y
            trajectory[0, i, 2] = next_ego_agents[i].heading
        tmp_feature_dict['trajectory'] = Trajectory(data=torch.tensor(trajectory))
        if self._is_vis_features:
            render_and_save_features(tmp_feature_dict, self._vis_features_path, bev_range=[-104., -104., 104., 104.])
        # import pdb; pdb.set_trace()
        return tmp_feature_dict
    
    def mcts_expansion(self, feature_dict, num_imagined_frames=1):
        tmp_feature_dict = feature_dict.copy()

        for transition_model in self.transition_models:
            mcts_features = transition_model.imagine(
                tmp_feature_dict,
                num_imagined_frames=num_imagined_frames,
                next_ego_state=feature_dict['next_ego_state'],
                )
            tmp_feature_dict.update(mcts_features)

        # FOR KUN.LI. extract next state here
        all_seqeuence_tokens = tmp_feature_dict['all_seqeuence_tokens']
        # suppose past frames are 0,1; current is 2, so next is 3, we slice next state here
        next_frame_sequence = all_seqeuence_tokens.data[0].sample_frame_segment(start_local_index=3, end_local_index=3)
        next_agents = next_frame_sequence.get_agent_tokens()  # list of agents, refer to class AgentToken
        # import pdb; pdb.set_trace()
        # for iter, post_processor in enumerate(self.post_processors):
        #     post_features = post_processor(tmp_feature_dict)
        #     tmp_feature_dict.update(post_features)
        tmp_feature_dict['next_agents'] = next_agents
        if self._is_vis_features:
            render_and_save_features(tmp_feature_dict, self._vis_features_path, bev_range=[-104., -104., 104., 104.])
        return tmp_feature_dict
    
    def sim_agents_rollout(self, feature_dict):
        """
        use sim_agents_batched_rollout instead! this function is kept for debugging purpose!
        """
        sim_agents_rollouts = []
        for future_i in range(64):
            print('ROLLING FUTURE: ', future_i)
            tmp_feature_dict = feature_dict.copy()
            for transition_model in self.transition_models:
                rollout_features = transition_model.imagine(tmp_feature_dict, num_imagined_frames=16)
                tmp_feature_dict.update(rollout_features)
            
            all_seqeuence_tokens = tmp_feature_dict['all_seqeuence_tokens']
            # suppose past frames are 0,1; current is 2, so next is 3, we slice next state here
            next_frame_sequence = all_seqeuence_tokens.data[0].sample_frame_segment(start_local_index=2, end_local_index=18)
            next_agents = next_frame_sequence.get_agent_tokens(return_all=True)  # list of agents, refer to class AgentToken
            sim_agents_rollouts.append(next_agents)
            del tmp_feature_dict

        return [sim_agents_rollouts]
    
    def sim_agents_batched_rollout(self, feature_dict):
        n_rollouts = 32
        # repeat and batch essential features: [B, ...] => [32*B, ...]
        tmp_feature_dict = feature_dict.copy()
        original_bs = tmp_feature_dict['vision_x'].shape[0]
        tmp_feature_dict['vision_x'] = tmp_feature_dict['vision_x'].repeat_interleave(n_rollouts, dim=0)  # [0-31: batch 1], [32-63: batch 2]....
        tmp_feature_dict['sequence_tokens'].repeat_interleave(n_rollouts)

        # imagine
        for transition_model in self.transition_models:
            rollout_features = transition_model.imagine(tmp_feature_dict, num_imagined_frames=16)
            tmp_feature_dict.update(rollout_features)
        all_seqeuence_tokens = tmp_feature_dict['all_seqeuence_tokens']
        # import pdb; pdb.set_trace()

        # reshape into [[32 rollout for b1], [32 rollout for b2], ...] format. 
        sim_agent_rollouts = [[] for _ in range(original_bs)]
        for batch_index, tok_seq in enumerate(all_seqeuence_tokens.data):
            batch_number = batch_index // n_rollouts
            # suppose past frames are 0,1; current is 2, so next is 3, we slice next state here
            # next_frame_sequence = tok_seq
            next_frame_sequence = tok_seq.sample_frame_segment(start_local_index=2, end_local_index=32)
            next_agents = next_frame_sequence.get_agent_tokens(return_all=True)
            sim_agent_rollouts[batch_number].append(next_agents)

        return sim_agent_rollouts

    def imagine(self, feature_dict, num_imagined_frames=20, start_local_index=0, end_local_index=2, next_ego_state=None):
        """
        imagine future frames for some period of time
        :param next_ego_state: ego state for next frame, numpy array of shape (batchsize, timesteps, 3) (x, y, heading)
        """
        for transition_model in self.transition_models:
            imaging_futures = transition_model.imagine(feature_dict, 
                                                       num_imagined_frames=num_imagined_frames, 
                                                       start_local_index=start_local_index, 
                                                       end_local_index=end_local_index, 
                                                       next_ego_state=next_ego_state)
            feature_dict.update(imaging_futures)
        return feature_dict
