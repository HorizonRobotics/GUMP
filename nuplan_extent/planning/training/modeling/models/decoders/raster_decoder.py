import timm
import torch
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.preprocessing.features.camera_e2e import \
    CameraE2E
from torch import nn

from ..utils import convert_predictions_to_trajectory


class RasterDecoder(nn.Module):
    def __init__(
            self,
            raster_model_name: str,
            raster_num_input_channels: int,
            num_output_features: int,
            pretrained: bool,
            with_command: bool = False,
    ):
        super().__init__()

        self.model = timm.create_model(
            raster_model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=raster_num_input_channels,
        )

        self.with_command = with_command

        input_features = self.model.num_features
        if self.with_command:
            input_features += 9

        mlp = torch.nn.Linear(
            in_features=input_features, out_features=num_output_features)

        if hasattr(self.model, 'classifier'):
            self.model.classifier = mlp
        elif hasattr(self.model, 'fc'):
            self.model.fc = mlp
        else:
            raise NameError(
                'Expected output layer named "classifier" or "fc" in model')

    def forward(self, encoder_features: torch.Tensor, inputs: CameraE2E,
                *args) -> torch.Tensor:

        features = self.model.forward_features(encoder_features)
        features = self.model.forward_head(features, pre_logits=True)

        # add nav, command and speed
        if self.with_command:
            states = inputs.command
            states = states.to(features.dtype)
            features = torch.cat((features, states), dim=-1)

        predictions = self.model.fc(features)

        return {
            "trajectory":
                Trajectory(
                    data=convert_predictions_to_trajectory(predictions)),
        }
