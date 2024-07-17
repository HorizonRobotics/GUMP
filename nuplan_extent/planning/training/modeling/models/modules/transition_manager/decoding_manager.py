import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

import loralib as lora
from einops import rearrange, repeat
from functools import partial
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type import VocabularyStateType, PositionalStateType

from nuplan_extent.planning.training.modeling.models.transition_models.base_transition_model import BaseTransitionModel
from nuplan_extent.planning.training.modeling.models.utils import get_sine_embedding_2d, process_trajectory
from nuplan_extent.planning.training.modeling.models.modules.GRU import CustomGRU, GRULayer

CustomGRU = nn.GRU



class DecodingManager(nn.Module):
    def __init__(self,
                 n_embd,
                 num_rnn_layers,
                 embedding_manager,
                 topk,
                 temperature,
                 num_agent_attributes,
                 num_max_traffic_light,
                 intermedia_layer_index_for_trajectory=[5, 11, 17, 23]
                 ):
        super().__init__()
        self.n_embd=n_embd
        self.num_rnn_layers=num_rnn_layers
        self.embedding_manager = embedding_manager
        self.topk = topk
        self.temperature = temperature
        self.num_agent_attributes = num_agent_attributes
        self.num_max_traffic_light = num_max_traffic_light
        self.intermedia_layer_features = {}

        self.visual_ca_ds_ratio = 3
        self.num_trajectory_predict_layers = 4
        self.past_pred_step = 6
        self.future_pred_step = 16
        self.trajectory_dim = 4
        self.num_modes = 6
        self.intermedia_layer_index_for_trajectory = intermedia_layer_index_for_trajectory

        self.init_traffic_light_head()
        self.init_agents_head()
        self.init_trajectory_head()

    def init_traffic_light_head(self):
        self.traffic_rnn_decoder = CustomGRU(
            input_size=self.n_embd,
            hidden_size=self.n_embd,
            num_layers=self.num_rnn_layers)
        self.traffic_head = nn.Sequential(
            nn.Linear(self.n_embd, 4, bias=False)
        )
        self.tl_pos_embedding = nn.Embedding(self.num_max_traffic_light*2, self.n_embd)


    def init_agents_head(self):
        self.agent_rnn_decoder = CustomGRU(
            input_size=self.n_embd,
            hidden_size=self.n_embd,
            num_layers=self.num_rnn_layers)
        self.lm_head = nn.Sequential(
            nn.Linear(self.n_embd, VocabularyStateType.AGENTS.vocal_size, bias=False)
            # lora.Linear(self.n_embd, VocabularyStateType.AGENTS.vocal_size, r=8, bias=False)
        )
        self.agents_pos_embedding = nn.Embedding(self.num_agent_attributes, self.n_embd)

    def init_trajectory_head(self):
        self.regression_feat_head = []
        self.past_trajectory_head = []
        self.future_trajectory_head = []
        self.prob_head = []
        for i in range(self.num_trajectory_predict_layers):
            reg_feat_head = nn.Sequential(
                nn.Linear(self.n_embd*2, self.n_embd*2),
                nn.LayerNorm(self.n_embd*2),
                nn.ReLU(),
                nn.Linear(self.n_embd * 2, self.n_embd * 2),
                nn.LayerNorm(self.n_embd * 2),
                nn.ReLU(),
            )
            past_traj_head = nn.Linear(self.n_embd*2,  self.past_pred_step * self.trajectory_dim)
            future_traj_head = nn.Linear(self.n_embd*2, self.num_modes * self.future_pred_step * self.trajectory_dim)
            prob_head = nn.Linear(self.n_embd*2, self.num_modes)
            self.regression_feat_head.append(reg_feat_head)
            self.past_trajectory_head.append(past_traj_head)
            self.future_trajectory_head.append(future_traj_head)
            self.prob_head.append(prob_head)
        self.regression_feat_head = nn.ModuleList(self.regression_feat_head)
        self.past_trajectory_head = nn.ModuleList(self.past_trajectory_head)
        self.future_trajectory_head = nn.ModuleList(self.future_trajectory_head)
        self.prob_head = nn.ModuleList(self.prob_head)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def update_intermedia_feature(self, x, intermedia_layer_index, generate_mode=False):
        """
        log and update the intermediate features
        """
        if generate_mode:
            if intermedia_layer_index == self.intermedia_layer_index_for_trajectory[-1]:
                self.intermedia_layer_features[intermedia_layer_index] = x
        else: 
            self.intermedia_layer_features[intermedia_layer_index] = x

    def decoding_traffic_light(self, x, sequence_tokens, return_last=False):
        sequence_tokens.aggregate_traffic_light_info()

        valid_batch_idxes, valid_seq_idxes = sequence_tokens.get_bos_idxes(return_last=return_last)
        hidden = x[valid_batch_idxes, valid_seq_idxes]
        hidden = repeat(hidden, 'b c -> l b c', l=self.num_rnn_layers)

        tl_embed = self.embedding_manager.get_traffic_light_embedding(sequence_tokens, return_last=return_last)

        num_traffic_light = sequence_tokens.get_num_traffic_light(return_last=return_last)

        if self.training:
            target_tl_status = sequence_tokens.get_traffic_light_status_list(np.max(num_traffic_light))
            target_tl_status = torch.from_numpy(target_tl_status).to(x.device).long()

        pred_tl_logits = []
        pred_tl_tokens = []
        for i in range(np.max(num_traffic_light)):
            current_tl_embed = tl_embed[:, i]
            current_tl_embed = current_tl_embed[None, ...] # add a time dim, since we are using rnn, time dim is 1 to AutoRegresive
            pos_embed = self.tl_pos_embedding(torch.tensor([[2*i]], device=hidden.device).long())
            input = current_tl_embed + pos_embed
            output, hidden = self.traffic_rnn_decoder(input, hidden)
            logits = self.traffic_head(output)
            pred_tl_logits.append(logits)

            assert output.shape[0] == 1

            if self.training:
                tl_status = target_tl_status[:, i:i+1]
            else:
                tl_status = self.sampling_logits(logits[0])
                pred_tl_tokens.append(tl_status)

            pos_embed = self.tl_pos_embedding(torch.tensor([[2*i+1]], device=hidden.device).long())
            tl_status_embed = self.embedding_manager.traffic_light_status_embedding(tl_status[..., 0])[None, ...]
            input = tl_status_embed + pos_embed
            _, hidden = self.traffic_rnn_decoder(input, hidden)
        return pred_tl_logits, pred_tl_tokens

    def get_sampling_mask(self, attr_index, agent_tokens=None):
        if attr_index == 0:
            sampling_mask = VocabularyStateType.WIDTH.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 1:
            sampling_mask = VocabularyStateType.LENGTH.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 2:
            sampling_mask = VocabularyStateType.X.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 3:
            sampling_mask = VocabularyStateType.Y.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 4:
            sampling_mask = VocabularyStateType.HEADING.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 5:
            sampling_mask = VocabularyStateType.X.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 6:
            sampling_mask = VocabularyStateType.Y.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 7:
            sampling_mask = VocabularyStateType.HEADING.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 8:
            sampling_mask = VocabularyStateType.X.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 9:
            sampling_mask = VocabularyStateType.Y.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 10:
            sampling_mask = VocabularyStateType.HEADING.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 11:
            sampling_mask = VocabularyStateType.X.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 12:
            sampling_mask = VocabularyStateType.Y.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        elif attr_index == 13:
            sampling_mask = VocabularyStateType.HEADING.get_sampling_mask()
            sampling_mask = np.tile(sampling_mask[None], (len(agent_tokens), 1))
        else:
            raise NotImplementedError
        return sampling_mask


    def decoding_agents(self, x, sequence_tokens, return_last_frame=False, return_last_agent=False, return_ego_only=False):
        assert not (return_last_frame and return_last_agent) , "return_last_frame and return_last_agent can not be True at the same time"
        valid_batch_idxes, valid_seq_idxes = sequence_tokens.get_agent_control_idxes(return_last_frame=return_last_frame, return_last_agent=return_last_agent, return_ego=return_ego_only)
        sequence_agent_tokens, batch_indexes = sequence_tokens.get_agent_tokens(return_last=return_last_frame, return_ego=return_ego_only)  

        if self.training:
            target_tokenized_state = sequence_tokens.get_tokenized_agent_state(mask_invalid=True, return_ego_only=return_ego_only)
        hidden = x[valid_batch_idxes, valid_seq_idxes]
        hidden = repeat(hidden, 'b c -> l b c', l=self.num_rnn_layers)

        input = torch.zeros_like(hidden[0:1])
        pred_agent_logits, pred_agent_tokens = [], []

        for i in range(self.num_agent_attributes):
            pos_embed = self.agents_pos_embedding(torch.tensor([[i]], device=hidden.device).long())
            input = input + pos_embed
            output, hidden = self.agent_rnn_decoder(input, hidden.contiguous())
            logits = self.lm_head(output)
            pred_agent_logits.append(logits)

            assert output.shape[0] == 1
            sampling_mask = self.get_sampling_mask(attr_index=i, agent_tokens=sequence_agent_tokens)
            sampling_mask = torch.from_numpy(sampling_mask).to(x.device)

            if self.training:
                agent_tokens = target_tokenized_state[i]
                pred_agent_tokens_cur = self.sampling_logits(logits[0], sampling_mask=sampling_mask, attr_index=i)
                pred_agent_tokens.append(pred_agent_tokens_cur)
                # agent_tokens = pred_agent_tokens_cur
                # agent_tokens[agent_tokens<0] = pred_agent_tokens_cur[agent_tokens<0]
            else:
                agent_tokens = self.sampling_logits(logits[0], sampling_mask=sampling_mask, attr_index=i)
                pred_agent_tokens.append(agent_tokens)

            # TODO: mask out invalid tokens
            token_embed = self.embedding_manager.token_embedding(agent_tokens[..., 0])[None, ...]
            input = token_embed
        return pred_agent_logits, pred_agent_tokens

    def decoding_normal_tokens(self, x, sequence_tokens):
        valid_batch_idxes, valid_seq_idxes = sequence_tokens.get_normal_decoding_idxes()
        hidden = x[valid_batch_idxes, valid_seq_idxes]
        logits = self.lm_head(hidden)
        assert len(logits.shape) == 2
        normal_tokens = self.sampling_logits(logits)
        return logits, normal_tokens

    def decoding_trajectory(self, x, sequence_tokens, nsamples=512, return_last=False):

        agent_tokens, batch_indexes = sequence_tokens.get_agent_tokens(return_last=return_last)
        if not return_last:
            random_indices = torch.randint(0, len(batch_indexes), (nsamples,))
            agent_tokens = [agent_tokens[idx] for idx in random_indices]
            batch_indexes = torch.tensor(batch_indexes)[random_indices].to(x.device).long()
        agent_id_idxes = torch.tensor([agent_token.index_in_sequence for agent_token in agent_tokens]).to(x.device).long()
        agent_state_idxes = torch.tensor([agent_token.index_in_sequence+1 for agent_token in agent_tokens]).to(x.device).long()

        lvl_pred_past_trajectory, lvl_pred_future_trajectory = [], []
        lvl_pred_past_sine_trajectory, lvl_pred_future_sine_trajectory = [], []
        lvl_pred_log_prob, lvl_layer_index = [], []

        for layer_number, intermedia_layer_feature in self.intermedia_layer_features.items():
            if layer_number not in self.intermedia_layer_index_for_trajectory:
                continue
            if return_last and layer_number != self.intermedia_layer_index_for_trajectory[-1]:
                continue
            layer_index = self.intermedia_layer_index_for_trajectory.index(layer_number)
            agent_feature_id = intermedia_layer_feature[batch_indexes, agent_id_idxes]
            agent_feature_state = intermedia_layer_feature[batch_indexes, agent_state_idxes]
            reg_feature = torch.cat([agent_feature_id, agent_feature_state], dim=-1)
            reg_feat = self.regression_feat_head[layer_index](reg_feature)

            past_sine_heading_traj = self.past_trajectory_head[layer_index](reg_feat)
            future_sine_heading_traj = self.future_trajectory_head[layer_index](reg_feat)
            log_prob = self.log_softmax(self.prob_head[layer_index](reg_feat))

            past_traj = process_trajectory(past_sine_heading_traj, shape=(-1, self.past_pred_step, self.trajectory_dim))
            future_traj = process_trajectory(future_sine_heading_traj, shape=(-1, self.num_modes,self.future_pred_step, self.trajectory_dim))
            lvl_pred_past_trajectory.append(past_traj)
            lvl_pred_future_trajectory.append(future_traj)
            lvl_pred_past_sine_trajectory.append(past_sine_heading_traj.reshape(-1, self.past_pred_step, self.trajectory_dim))
            lvl_pred_future_sine_trajectory.append(future_sine_heading_traj.reshape(-1, self.num_modes, self.future_pred_step, self.trajectory_dim))
            lvl_pred_log_prob.append(log_prob)
            lvl_layer_index.append(layer_index)

        return lvl_pred_past_trajectory, lvl_pred_future_trajectory, lvl_pred_past_sine_trajectory, lvl_pred_future_sine_trajectory, lvl_pred_log_prob, lvl_layer_index, agent_tokens, batch_indexes

    def sampling_logits(self, logits, sampling_mask=None, attr_index=None):
        # apply softmax to convert to probabilities
        logits = torch.nan_to_num(logits, nan=-10)
        # if attr_index == 2:
        #     temperature = 1.3
        # else:
        #     temperature = self.temperature
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
    # def sampling_logits(self, logits, sampling_mask=None, attr_index=None):
    #     # apply softmax to convert to probabilities
    #     logits = torch.nan_to_num(logits, nan=-10)
    #     temperature = self.temperature
    #     logits = logits / temperature

    #     # optionally crop the logits to only the top k options or apply top-p sampling
    #     if sampling_mask is not None:
    #         logits[~sampling_mask] = -float('Inf')
        
    #     # Apply top-p sampling
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    #     sorted_indices_to_remove = cumulative_probs > 0.95
    #     # Shift the indices to the right to keep the first one that cumsum to <= p
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0

    #     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    #     logits[indices_to_remove] = -float('Inf')

    #     # Apply top-k if necessary
    #     if self.topk is not None:
    #         v, _ = torch.topk(logits, min(self.topk, logits.size(-1)))
    #         logits[logits < v[:, [-1]]] = -float('Inf')

    #     # apply softmax to convert logits to (normalized) probabilities
    #     probs = F.softmax(logits, dim=-1)
    #     # sample from the distribution
    #     tokens = torch.multinomial(probs, num_samples=1)
    #     return tokens