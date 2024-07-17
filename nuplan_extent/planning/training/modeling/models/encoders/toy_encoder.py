import torch.nn as nn


class ToyEncoder(nn.Module):
    """
    A toy encoder for E2E model for quick testing.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, image, **kwargs):
        # [B, T, M, 3, H, W]
        N, T, M = image.shape[:3]
        x = image.view(N, T * M * 3, *image.shape[4:])
        x = self.conv(x)
        return x
