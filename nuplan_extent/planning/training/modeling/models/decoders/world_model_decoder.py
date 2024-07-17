from typing import Dict

from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.models.utils import (
    convert_predictions_to_trajectory)
from torch import nn
import torch
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.batch_token_sequence import BatchTokenSequence


class WorldModelDecoder(nn.Module):
    """Simple decoder for trajectory regression using EncodingNetwork."""

    def __init__(
            self,
    ):
        """
        :param in_features_channel: input channels of selected input features
        :param num_output_features: output dimension of decoder (num_steps * trajctory_dim)
        :param summarize_input_gradient: whether to summarize gradient of input
            features (after pooling).
        :param kwargs_to_encoding_network: kwargs to alf.networks.EncodingNetwork.
            It can include all the arguments of `alf.networks.EncodingNetwork` except
            input_tensor_spec, last_layer_size and last_activation.
        """
        super().__init__()

    def forward(self, neck_output: Dict) -> Dict:
        all_sequence_tokens = neck_output['output'].get_forward_sequence_tokens()
        log_action_distribution = neck_output['output'].get_ego_action(all_sequence_tokens)
        representation = neck_output['output'].ego_embed
        return {
            'log_action_distribution': log_action_distribution,
            'all_sequence_tokens_tensor': torch.tensor(all_sequence_tokens.numpy(), dtype=torch.float32),
            'vision_x': neck_output['vision_x'],
            'representation': representation,
        }