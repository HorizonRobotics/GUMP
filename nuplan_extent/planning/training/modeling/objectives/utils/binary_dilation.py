from typing import Tuple

import torch


def binary_dilation_torch(input: torch.Tensor,
                          kernel_shape: Tuple[int]) -> torch.Tensor:
    """
    Perform binary dilation on a tensor using a square structuring element.
    :param input: A binary tensor of shape (batch_size, num_channels, height, width).
    :param kernel_shape: The size of the square structuring element.
    :return: A binary tensor of shape (batch_size, num_channels, height, width) representing the dilated input.
    """
    batch_size, num_channels, height, width = input.shape
    struct_height, struct_width = kernel_shape

    # Calculate the padding required to preserve the shape of the input
    padding = (struct_height // 2, struct_width // 2)

    # Convert the structuring element to a kernel for convolution
    kernel = torch.nn.Conv2d(
        1, 1, (struct_height, struct_width), padding=padding,
        bias=False).to(input.device)
    kernel.weight.data.fill_(1)

    # Set requires_grad to False to allow in-place modification
    kernel.weight.requires_grad = False

    # Perform convolution on the input
    output = torch.zeros_like(input)
    for i in range(num_channels):
        # Apply convolution on each channel of the input tensor
        output[:, i:i + 1] = kernel(input[:, i:i + 1].float()).ge(1).float()

    return output
