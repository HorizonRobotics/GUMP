from typing import Dict

from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.models.utils import (
    convert_predictions_to_trajectory)
from torch import nn


class MLPDecoderV2(nn.Module):
    """Simple decoder for trajectory regression using EncodingNetwork."""

    def __init__(
            self,
            in_features_channel: int = 512,
            num_output_features: int = 48,
            summarize_input_gradient: bool = False,
            **kwargs_to_encoding_network,
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
        import alf
        from alf.networks import EncodingNetwork
        self._decoder = EncodingNetwork(
            input_tensor_spec=alf.TensorSpec((in_features_channel, )),
            last_layer_size=num_output_features,
            last_activation=alf.math.identity,
            **kwargs_to_encoding_network)

        if summarize_input_gradient:
            import alf.layers as layers
            self._summarize_gradient = layers.SummarizeGradient(
                "mlp_decoder_input")
        else:
            self._summarize_gradient = nn.Identity()

    def forward(self, neck_output: Dict) -> Dict:
        x = neck_output['representation']
        x = self._summarize_gradient(x)
        pred_trajectory = self._decoder(x)[0]
        return {
            "trajectory":
                Trajectory(
                    data=convert_predictions_to_trajectory(pred_trajectory))
        }
