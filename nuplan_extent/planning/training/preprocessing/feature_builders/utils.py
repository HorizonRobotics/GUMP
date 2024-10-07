from typing import List, Optional

import torch
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import torch
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex, EgoInternalIndex
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local


def agent_bbox_to_corners(bboxes: List[torch.tensor]) -> torch.tensor:
    """Converts bounding boxes to corner coordinates.

    Given a list of bounding boxes in the format [xy, l, w, theta], where xy is the
    center point of the box, l and w are the length and width of the box, and theta
    is the angle of the box (in radians) relative to the x-axis, this function returns
    the corner coordinates of the bounding boxes after rotating them by the angle theta.

    :params: bboxes: A list of tensors of shape (4,) representing the bounding boxes. Each
            tensor in the list has four elements: xy, l, w, and theta.

    :returns: corners: A tensor of shape (n, 4, 2) representing the corner coordinates of the bounding
        boxes after rotation. The first dimension of the tensor is the number of bounding
        boxes in the input.

    """
    # Define a variable for half of pi
    half_pi = np.pi / 2.0

    # Unpack the input list into variables
    xy, l, w, theta = bboxes

    # Define the corners of the bounding boxes, shape: (n, 4, 2)
    corners = torch.stack([
        torch.cat([-l / 2, w / 2], dim=1),
        torch.cat([l / 2, w / 2], dim=1),
        torch.cat([l / 2, -w / 2], dim=1),
        torch.cat([-l / 2, -w / 2], dim=1)
    ],
        dim=1)

    # Define the rotation matrix, shape: (n, 2, 2)
    rot_mat = torch.stack([
        torch.cat(
            [torch.cos(theta), torch.cos(theta + half_pi)], dim=1),
        torch.cat(
            [torch.sin(theta), torch.sin(theta + half_pi)], dim=1)
    ],
        dim=1)

    # Rotate the corners and add the center point
    new_corners = torch.matmul(rot_mat, corners.permute(0, 2,
                                                        1)) + torch.unsqueeze(
                                                            xy, dim=2)

    # Return the rotated corners
    return new_corners.permute(0, 2, 1)


def batch_bbox_to_corners(bboxes: torch.tensor) -> torch.tensor:
    """Converts batch of bounding boxes to corner coordinates.

    :params: bboxes: A tensor of shape (T, n, d) or (B, T, n, d) representing the bounding boxes.
    :return corners: A tensor of shape (T, n, 4, 2) or (B, T, n, 4, 2) representing the corner coordinates of the bounding
        boxes after rotation.
    """
    n_dim = bboxes.dim()
    if n_dim == 3:
        # Add a dimension for batch size
        bboxes = bboxes.unsqueeze(0)
    batch_size, time_length, num_agents, dim = bboxes.shape
    transformed_bboxes = bboxes.view(batch_size * time_length * num_agents,
                                     dim)

    bbox_corners = agent_bbox_to_corners([
        transformed_bboxes[:, [AgentFeatureIndex.x(
        ), AgentFeatureIndex.y()]],
        transformed_bboxes[:,
                           AgentFeatureIndex.
                           length():AgentFeatureIndex.length() + 1],
        transformed_bboxes[:,
                           AgentFeatureIndex.
                           width():AgentFeatureIndex.width() + 1],
        transformed_bboxes[:,
                           AgentFeatureIndex.
                           heading():AgentFeatureIndex.heading() + 1]
    ])
    bbox_corners = bbox_corners.view(batch_size, time_length, num_agents, 4, 2)

    if n_dim == 3:
        # Remove the dimension for batch size
        bbox_corners = bbox_corners.squeeze(0)
    return bbox_corners


def pad_agent_with_max(
        agent_history: List[torch.Tensor], max_agents: int) -> torch.tensor:
    """
    pad agent history with max agents, the invalid postion filled with nan, then stack them together
    :param agent_history: list of tensors with shape [(num_agents, d)] * num_frame
    :param max_agents: max number of agents in the batch
    :return: tensor with shape (num_frame, max_agents, d)
    """
    padded_agent_history = []
    for frame_data in agent_history:
        num_agents, d = frame_data.shape

        # If there are less agents than the max, we'll pad the tensor
        if num_agents < max_agents:
            padding = torch.full(
                (max_agents - num_agents, d), float('nan'), dtype=frame_data.dtype)
            frame_data = torch.cat([frame_data, padding], dim=0)
        if num_agents > max_agents:
            # If there are more agents than the max, we'll truncate the tensor
            frame_data = frame_data[:max_agents]
        padded_agent_history.append(frame_data)
    return torch.stack(padded_agent_history, dim=0)


def pack_agents_tensor_withindex(
        padded_agents_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines the local padded agents states and the computed yaw rates into the final output feature tensor.
    :param padded_agents_tensors: The padded agent states for each timestamp.
        Each tensor is of shape <num_agents, len(AgentInternalIndex)> and conforms to the AgentInternalIndex schema.
    :param yaw_rates: The computed yaw rates. The tensor is of shape <num_timestamps, agent>
    :return: The final feature, a tensor of shape [timestamp, num_agents, len(AgentsFeatureIndex)] conforming to the AgentFeatureIndex Schema
    """

    agents_tensor = torch.zeros(
        (len(padded_agents_tensors),
         padded_agents_tensors[0].shape[0],
         AgentInternalIndex.dim())
    )

    for i in range(len(padded_agents_tensors)):
        agents_tensor[i,
                      :,
                      AgentInternalIndex.x()] = padded_agents_tensors[i][:,
                                                                         AgentInternalIndex.x()].squeeze()
        agents_tensor[i,
                      :,
                      AgentInternalIndex.y()] = padded_agents_tensors[i][:,
                                                                         AgentInternalIndex.y()].squeeze()
        agents_tensor[i, :, AgentInternalIndex.heading(
        )] = padded_agents_tensors[i][:, AgentInternalIndex.heading()].squeeze()
        agents_tensor[i,
                      :,
                      AgentInternalIndex.vx()] = padded_agents_tensors[i][:,
                                                                          AgentInternalIndex.vx()].squeeze()
        agents_tensor[i,
                      :,
                      AgentInternalIndex.vy()] = padded_agents_tensors[i][:,
                                                                          AgentInternalIndex.vy()].squeeze()
        agents_tensor[i, :, AgentInternalIndex.track_token(
        )] = padded_agents_tensors[i][:, AgentInternalIndex.track_token()].squeeze()
        agents_tensor[i,
                      :,
                      AgentInternalIndex.width()] = padded_agents_tensors[i][:,
                                                                             AgentInternalIndex.width()].squeeze()
        agents_tensor[i, :, AgentInternalIndex.length(
        )] = padded_agents_tensors[i][:, AgentInternalIndex.length()].squeeze()
    return agents_tensor


def build_generic_ego_features_from_tensor(
        ego_trajectory: torch.Tensor, ego_state_index: int) -> torch.Tensor:
    """
    Build generic agent features from the ego states
    :param ego_trajectory: ego states at past times. Tensors complying with the EgoInternalIndex schema.
    :param reverse: if True, the last element in the list will be considered as the present ego state
    :return: Tensor complying with the GenericEgoFeatureIndex schema.
    """

    anchor_ego_pose = (
        ego_trajectory[
            ego_state_index, [
                EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]
        ]
        .squeeze()
        .double()
    )

    global_ego_poses = ego_trajectory[
        :, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]
    ].double()
    global_ego_velocities = ego_trajectory[
        :, [EgoInternalIndex.vx(), EgoInternalIndex.vy()]
    ].double()
    global_ego_accelerations = ego_trajectory[
        :, [EgoInternalIndex.ax(), EgoInternalIndex.ay()]
    ].double()

    local_ego_poses = global_state_se2_tensor_to_local(
        global_ego_poses, anchor_ego_pose, precision=torch.float64)
    local_ego_velocities = rotate_vectors_to_local(
        global_ego_velocities, anchor_ego_pose[-1], precision=torch.float64
    )
    local_ego_accelerations = rotate_vectors_to_local(
        global_ego_accelerations, anchor_ego_pose[-1], precision=torch.float64
    )

    # Minor optimization. The indices in GenericEgoFeatureIndex and
    # EgoInternalIndex are the same.
    local_ego_trajectory = deepcopy(ego_trajectory)
    local_ego_trajectory[:,
                         EgoInternalIndex.x()] = local_ego_poses[:,
                                                                 0].float()
    local_ego_trajectory[:,
                         EgoInternalIndex.y()] = local_ego_poses[:,
                                                                 1].float()
    local_ego_trajectory[:,
                         EgoInternalIndex.heading()] = local_ego_poses[:,
                                                                       2].float()
    local_ego_trajectory[:, EgoInternalIndex.vx(
    )] = local_ego_velocities[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.vy(
    )] = local_ego_velocities[:, 1].float()
    local_ego_trajectory[:, EgoInternalIndex.ax(
    )] = local_ego_accelerations[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.ay(
    )] = local_ego_accelerations[:, 1].float()

    return local_ego_trajectory

def rotate_vectors_to_local(
    vectors: torch.Tensor, local_heading: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Rotates vectors to the local frame.

    :param vectors: A tensor of shape (..., 2), where the last dimension is [vx, vy].
    :param local_heading: A tensor of shape broadcastable to vectors.shape[:-1].
    :param precision: The precision to use.
    :return: Rotated vectors in the local frame of shape (..., 2).
    """
    if precision is None:
        precision = vectors.dtype

    # Ensure local_heading has the correct precision
    local_heading = local_heading.type(precision)

    # Compute cosine and sine of headings
    cos_h = torch.cos(local_heading)
    sin_h = torch.sin(local_heading)

    # Extract vector components
    vx = vectors[..., 0]
    vy = vectors[..., 1]

    # Perform element-wise rotation
    rotated_x = cos_h * vx + sin_h * vy
    rotated_y = -sin_h * vx + cos_h * vy

    # Stack the rotated components back into a tensor
    rotated_vectors = torch.stack((rotated_x, rotated_y), dim=-1)

    return rotated_vectors



def convert_to_uint8(value: npt.NDArray, scale: Optional[int] = 1):
    """Converts a numpy array to a numpy array of uint8 values."""
    return np.uint8(np.clip(value * scale, 0, 255))


def convert_uint16_to_two_uint8(value: npt.NDArray):
    """Converts a numpy array of uint16 values to two numpy arrays of uint8 values."""
    return np.uint8(value / 256), np.uint8(value % 256)


def convert_two_uint8_to_uint16(value1: npt.NDArray, value2: npt.NDArray):
    """Converts two numpy arrays of uint8 values to a numpy array of uint16 values."""
    return np.uint16(value1) * 256 + np.uint16(value2)

def normalize_angle(angles):
    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi
    return angles
