from typing import Dict

import torch
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.models.utils import (
    convert_predictions_to_multimode_trajectory,
    convert_predictions_to_trajectory)
from nuplan_extent.planning.training.preprocessing.features.multimode_trajectory import \
    MultiModeTrajectory
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor
from torch import nn


class MLPDecoder(nn.Module):
    """
    simple MLP decoder for trajectory regression.
    """

    def __init__(
        self,
        in_layer_index: int = -1,
        in_features_channel: int = 512,
        embed_dim: int = 128,
        num_output_features: int = 48,
        num_modes: int = 1,
        summarize_input_gradient: bool = False,
    ):
        """
        :param in_layer_index: input feature layer index from upstreams
        :param in_features_channel: input channels of selected input features
        :param embed_dim: embedding dimension of GRU decoders
        :param num_output_features: output dimension of decoder (num_steps * trajctory_dim)
        :param num_modes: number of modes
        :param summarize_input_gradient: whether to summarize gradient of input
            features (after pooling).
        """
        super().__init__()
        self.in_layer_index = in_layer_index
        self.num_modes = num_modes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.num_modes <= 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features_channel,
                          out_features=embed_dim), nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=embed_dim,
                          out_features=num_output_features))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=in_features_channel,
                          out_features=embed_dim), nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True))
            self.regression_mlp = nn.Sequential(
                nn.Linear(in_features=embed_dim, out_features=embed_dim),
                nn.LayerNorm(embed_dim), nn.ReLU(inplace=True),
                nn.Linear(in_features=embed_dim,
                          out_features=num_output_features * num_modes))
            self.prob_mlp = nn.Sequential(
                nn.Linear(in_features=embed_dim, out_features=embed_dim),
                nn.LayerNorm(embed_dim), nn.ReLU(inplace=True),
                nn.Linear(in_features=embed_dim, out_features=num_modes))
            self.log_softmax = torch.nn.LogSoftmax(dim=1)

        if summarize_input_gradient:
            import alf.layers as layers
            self._summarize_gradient = layers.SummarizeGradient(
                "mlp_decoder_input")
        else:
            self._summarize_gradient = nn.Identity()

    def forward(self, neck_output: Dict) -> Dict:
        neck_features = neck_output['neck_output']
        x = self.global_pool(neck_features[self.in_layer_index])  # B, C, 1, 1
        x = self._summarize_gradient(x)
        x = x.squeeze(-1).squeeze(-1)
        if self.num_modes <= 1:
            pred_trajectory = self.mlp(x)
            return {
                "trajectory":
                    Trajectory(
                        data=convert_predictions_to_trajectory(pred_trajectory)
                    )
            }
        else:
            embed_x = self.mlp(x)

            pred_trajectory = self.regression_mlp(
                embed_x)  # B, num_modes * num_steps * 3
            pred_log_prob = self.log_softmax(
                self.prob_mlp(embed_x))  # B, num_modes
            if self.training or "command" not in neck_output:
                return {
                    "multimode_trajectory":
                        MultiModeTrajectory(
                            data=convert_predictions_to_multimode_trajectory(
                                pred_trajectory, mode=self.num_modes)),
                    "pred_log_prob":
                        Tensor(data=pred_log_prob)
                }
            else:
                command = neck_output["command"]
                multi_traj = convert_predictions_to_multimode_trajectory(
                    pred_trajectory, mode=self.num_modes)
                return {
                    "trajectory":
                        Trajectory(data=multi_traj.
                                   data[:, command.data[0], :, :].squeeze(1)),
                }
