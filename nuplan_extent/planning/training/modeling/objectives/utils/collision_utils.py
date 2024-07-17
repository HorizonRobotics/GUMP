from typing import List

import numpy as np

import torch


def collision_checking_torch(input: torch.Tensor,
                             kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform binary dilation on a tensor using a square structuring element.
    :param input: A binary tensor of shape (batch_size, num_channels, height, width).
    :param kernel: A binary tensor of shape (num_channels, 1, struct_height, struct_width).
    :return: A binary tensor of shape (batch_size, num_channels, height, width) representing the dilated input.
    """
    channel, _, struct_height, struct_width = kernel.shape

    # Calculate the padding required to preserve the shape of the input
    padding = (struct_height // 2, struct_width // 2)
    # Convert the structuring element to a kernel for convolution
    conv_layer = torch.nn.Conv2d(
        channel,
        channel, (struct_height, struct_width),
        padding=padding,
        groups=channel,
        bias=False).to(input.device).to(input.dtype)
    conv_layer.weight.data = kernel

    # Set requires_grad to False to allow in-place modification
    conv_layer.weight.requires_grad = False

    # Perform convolution on the input
    output = torch.zeros_like(input)
    # Apply convolution on each channel of the input tensor
    output = conv_layer(input)

    return output


def draw_rotated_rect(batch_rect: torch.Tensor, H: int,
                      W: int) -> torch.Tensor:
    """
    Fills a rotated rectangle on a 2D raster map using PyTorch.
    the rectangle is defined by the x, y, height, width, and heading of the rectangle.
    return a tensor of shape (batch_size, 1, H, W) containing the raster map with the rectangle filled.
    :param batch_rect: A tensor of shape (batch_size, 6) containing the x, y, height_front, height_rear, width, and heading of the rectangle.
    :param H: The height of the raster map.
    :param W: The width of the raster map.
    :return: A tensor of shape (batch_size, 1, H, W) containing the raster map with the rectangle filled.
    """

    # Get the batch size of the input rectangle tensor
    batch_size = batch_rect.shape[0]
    device = batch_rect.device
    dtype = batch_rect.dtype

    # Create a grid of pixel coordinates
    grid = torch.ones((batch_size, 1, H, W), dtype=dtype).to(device)

    # Extract the x, y, height, width, and heading of the rectangle
    x, y, height_front, height_rear, width, theta = batch_rect.T

    # Calculate the rotation angle in radians
    # theta = torch.deg2rad(heading)

    # Create a 3x3 scaling matrix
    scaling = torch.stack(
        [torch.eye(3, dtype=dtype).to(device) for i in range(batch_size)],
        dim=0)
    scaling[:, 0, 0] = width
    scaling[:, 1, 1] = (height_front + height_rear)
    scaling[:, 1, 2] = height_rear - height_front
    # Create a 3x3 rotation matrix
    rotation = torch.stack(
        [torch.eye(3, dtype=dtype).to(device) for i in range(batch_size)],
        dim=0)
    rotation[:, 0, 0] = torch.cos(theta)
    rotation[:, 0, 1] = torch.sin(theta)
    rotation[:, 1, 0] = -torch.sin(theta)
    rotation[:, 1, 1] = torch.cos(theta)

    # Create a 3x3 translation matrix
    translation = torch.stack(
        [torch.eye(3, dtype=dtype).to(device) for i in range(batch_size)],
        dim=0)
    translation[:, 0, 2] = x
    translation[:, 1, 2] = y

    # Create a 3x3 composite matrix by multiplying the rotation and
    # translation matrices
    transform = torch.matmul(translation, torch.matmul(rotation, scaling))

    # Get the inverse of the transformation matrix
    inv_transform = torch.inverse(transform.float())

    # Warp the grid of pixel coordinates using the inverse transformation
    # matrix
    warped_grid = torch.nn.functional.affine_grid(
        inv_transform[:, :2], torch.Size((batch_size, 1, H, W)))

    # Sample the input grid of pixel values using the warped grid of pixel
    # coordinates
    output = torch.nn.functional.grid_sample(grid, warped_grid)

    return output


def ego_occupancy_kernel_render(
        trajectory: torch.Tensor,  # B, 16, 3
        bev_range: List[float],
        bev_meshgrid: List[float],
        ego_width: float,
        ego_front_length: float,
        ego_rear_length: float) -> torch.Tensor:
    """
    Render ego occupancy kernel on a bird's eye view map.
    i.e draw a rectangle on the bird's eye view map representing the ego vehicle.
    :param trajectory: A tensor of shape (batch_size, num_steps, 3) containing the x, y, and heading of the ego vehicle.
    :param bev_range: A list of length 4 containing the x, y, width, and height of the bird's eye view map.
    :param bev_meshgrid: A list of length 2 containing the x and y meshgrid of the bird's eye view map.
    :param ego_width: The width of the ego vehicle.
    :param ego_front_length: The length of the front to rear axis of the ego vehicle.
    :param ego_rear_length: The length of the rear to rear axis of the ego vehicle.
    :return: A tensor of shape (batch_size, num_steps, 1, H, W) containing the bird's eye view map with the ego vehicle
    """
    batch_size = trajectory.shape[0]
    num_steps = trajectory.shape[1]
    target_width = int((bev_range[3] - bev_range[1]) / bev_meshgrid[1])
    target_height = int((bev_range[2] - bev_range[0]) / bev_meshgrid[0])

    batch_rect = torch.zeros((batch_size, num_steps, 6),
                             dtype=trajectory.dtype).to(trajectory.device)
    batch_rect[:, :, 0] = 0  # real world y is car left, so y is width
    batch_rect[:, :, 1] = 0  # real world x is car front, so x is height
    batch_rect[:, :, 2] = ego_front_length / (bev_range[2] - bev_range[0])
    batch_rect[:, :, 3] = ego_rear_length / (bev_range[2] - bev_range[0])
    batch_rect[:, :, 4] = ego_width / (bev_range[3] - bev_range[1])
    batch_rect[:, :, 5] = trajectory[:, :, 2]  # theta

    batch_rect = batch_rect.view(-1, 6)
    output = draw_rotated_rect(batch_rect, target_height, target_width)
    output = output.view(batch_size, num_steps, target_height, target_width)
    return output


def ego_to_corners_batch(
        bbox: List[torch.Tensor],
        ego_width: float = 2.297,
        ego_front_length: float = 4.049,
        ego_rear_length: float = 1.127,
) -> torch.Tensor:
    """Converts ego bounding boxes to corner coordinates.

    :param bbox (List[torch.Tensor]): A list of tensors of shape (2,) representing the bounding boxes.
        Each tensor in the list has four elements: xy and theta. xy is located at the rear axle of ego car.
    :return corners (torch.Tensor): A tensor of shape (b, t, 4, 2) representing the corner coordinates of
        the bounding boxes after rotation. The first dimension of the tensor is batch size, and the second
        dimension is length of predicted trajectory.
    """
    # Define constants
    half_pi = np.pi / 2.0

    # Extract xy and theta
    xy, theta = bbox

    # Define corners of ego car in local coordinate system
    corners = torch.tensor([[-ego_rear_length, ego_width / 2],
                            [ego_front_length, ego_width / 2],
                            [ego_front_length, -ego_width / 2],
                            [-ego_rear_length,
                             -ego_width / 2]]).to(xy.device)  # 4,2

    # Reshape corners tensor
    corners = torch.unsqueeze(
        torch.unsqueeze(corners, dim=0), dim=0)  # 1, 1, 4, 2

    # Define rotation matrix
    rot_mat = torch.stack([
        torch.stack(
            [torch.cos(theta), torch.cos(theta + half_pi)], dim=2),
        torch.stack(
            [torch.sin(theta), torch.sin(theta + half_pi)], dim=2)
    ],
                          dim=2)  # b, t, 2, 2

    # Apply rotation and transformation
    transformed_corners = torch.matmul(rot_mat, corners.permute(
        0, 1, 3, 2)) + torch.unsqueeze(
            xy, dim=3)

    # Return transformed corners tensor
    return transformed_corners.permute(0, 1, 3, 2)


def get_nearest_neighbors(agents_target_trajectory: torch.Tensor,
                          nearest_neighbors: int) -> torch.Tensor:
    """Get the nearest neighbors for ego agent in a given agent target trajectory.

    :param agents_target_trajectory (torch.Tensor): A tensor of shape (time_length, num_agents, num_dimensions) representing the
                                            target trajectory for each agent.
    :return closest_agents (torch.Tensor): A tensor of shape (time_length, num_neighbors, num_dimensions) representing the
                                    nearest neighbors for each agent at each time step.
    :return closest_agents_mask (torch.Tensor): A tensor of shape (num_neighbors) representing a binary mask where the first
                                        num_neighbors elements are set to 1 and the rest are 0.
    """
    # Get the dimensions of the input tensor
    time_length, num_agents, num_dim = agents_target_trajectory.shape

    # Initialize a tensor for the closest agents
    closest_agents = torch.zeros((time_length, nearest_neighbors, num_dim),
                                 dtype=torch.float32,
                                 device=agents_target_trajectory.device)

    # Determine the number of neighbors to find
    num_neighbors = min(nearest_neighbors, num_agents)

    # Sort the agents by distance from each other
    sorted_indices = torch.argsort(
        torch.linalg.vector_norm(agents_target_trajectory[:, :, :2], dim=2),
        dim=1)

    # Reshape the sorted indices tensor and the target trajectory tensor
    reshaped_sorted_indices = sorted_indices.reshape(time_length * num_agents)
    reshaped_agents_target_trajectory = agents_target_trajectory.reshape(
        time_length * num_agents, num_dim)

    # Sort the agents according to the sorted indices
    sorted_agents = reshaped_agents_target_trajectory[
        reshaped_sorted_indices, :]
    sorted_agents = sorted_agents.reshape(time_length, num_agents, num_dim)

    # Copy the closest agents into the closest_agents tensor
    closest_agents[:, :num_neighbors] = sorted_agents[:, :num_neighbors, :]

    # Create a mask to indicate which agents are closest
    closest_agents_mask = torch.zeros((nearest_neighbors),
                                      dtype=torch.float32,
                                      device=agents_target_trajectory.device)
    closest_agents_mask[:num_neighbors] = 1

    return closest_agents, closest_agents_mask


def get_nearest_neighbors_batch(agents_target_trajectory: List[torch.Tensor],
                                device: torch.device,
                                nearest_neighbors: int = 10) -> torch.Tensor:
    """Get the nearest neighbors for a batch of agent target trajectories.

    :param agents_target_trajectory: A list of tensors of shape (time_length, num_agents, num_dimensions)
        representing the target trajectory of agents.
    :param device: The device on which to perform the computation.
    :param nearest_neighbors (int): Number of nearest neighbors to consider.
    :return: A tuple of tensors containing the nearest neighbors and their corresponding masks. The first tensor
        contains the nearest neighbors with shape (batch_size, time_length, num_neighbors, num_dimensions), while
        the second tensor contains the masks with shape (batch_size, num_neighbors).
    """
    batch_size = len(agents_target_trajectory)
    num_time, num_dim = agents_target_trajectory[0].shape[
        0], agents_target_trajectory[0].shape[-1]
    batch_agents_target_close_n = torch.zeros(
        (batch_size, num_time, nearest_neighbors, num_dim),
        dtype=torch.float32,
        device=device,
    )
    batch_agents_target_mask_close_n = torch.zeros(
        (batch_size, nearest_neighbors),
        dtype=torch.float32,
        device=device,
    )
    for idx in range(batch_size):
        agents_target_trajectory_tensor = agents_target_trajectory[idx]
        if agents_target_trajectory_tensor.shape[0] == 0:
            continue
        agents_target_close_n, agents_target_mask_close_n = get_nearest_neighbors(
            agents_target_trajectory_tensor, nearest_neighbors)
        batch_agents_target_close_n[idx] = agents_target_close_n
        batch_agents_target_mask_close_n[idx] = agents_target_mask_close_n

    return batch_agents_target_close_n, batch_agents_target_mask_close_n
