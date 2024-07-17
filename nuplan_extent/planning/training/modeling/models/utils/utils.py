from typing import Tuple

import numpy as np
from scipy.ndimage import binary_dilation

import torch
import torch.nn as nn
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory


def pack_sequence_dim(x):
    b, s = x.shape[:2]
    return x.view(b * s, *x.shape[2:])


def unpack_sequence_dim(x, b, s):
    return x.view(b, s, *x.shape[1:])


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def convert_predictions_to_trajectory(
        predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


def add_state_dimension(feature: torch.Tensor) -> torch.Tensor:
    """
    Add state dimension to 1d feature tensor
    """
    assert len(feature.shape) == 2
    num_batches, num_features = feature.shape
    return feature.view(num_batches, -1, num_features)


def convert_predictions_to_multimode_trajectory(predictions: torch.Tensor,
                                                mode: int = 1) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, mode, -1, Trajectory.state_size())


def unravel_index(
        indices: torch.LongTensor,
        shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def expand_mask(mask: np.ndarray, width: int) -> np.ndarray:
    """
    Expand a boolean mask using binary dilation with a square kernel of given width.

    Args:
        mask (np.ndarray): boolean array of shape (H, W).
        width (int): width of the square kernel used for binary dilation.

    Returns:
        np.ndarray: the expanded boolean mask of shape (H, W).
    """
    kernel = np.ones((width, width), dtype=np.bool_)
    return binary_dilation(mask, structure=kernel)


def extract_route_bin_from_expert(expert_trajectory: torch.Tensor,
                                  current_speed: torch.Tensor,
                                  delta_speed: float = 1.5,
                                  delta_turning: float = 1.5,
                                  longitudinal_critical_step: int = 4,
                                  latitudinal_critical_step: int = 9):
    """
    classified expert trajectory into multiple bins:
        longitudinal bins:
            - accelerate: when speed(2s) > current_speed - delta_speed
            - deccelerate: when speed(2s) < current_speed + delta_speed
            - maintain current speed (2s) otherwise

        latitudinal bins:
            - turning left: when y(0-4s) < - delta_turning
            - keep forward: when -delta_turning <= y(0-4s) < delta_turning
            - turning right: when y(0-4s) > delta_turning

    :param expert_trajectory: tensor shape (batch, 16, 3) with (x, y, heading), with each step 0.5s spacing
    :param current_speed: [m/s] tensor shape (batch, 1)
    :param delta_speed: float, [m/s] threshold for speed change
    :param delta_turning: float, [m] threshold for turning
    :param longitudinal_critical_step: int, critical step for longitudinal bins
    :param latitudinal_critical_step: int, critical step for latitudinal bins
    :return longitudinal_bin: one hot tensor shape (batch, 3)
    :return latitudinal_bin: one hot tensor shape (batch, 3)
    """
    dtype = expert_trajectory.dtype
    device = expert_trajectory.device

    # Calculate speed in 2 seconds
    speed_future = (expert_trajectory[:, longitudinal_critical_step, :2] -
                    expert_trajectory[:, longitudinal_critical_step - 1, :2]
                    ).norm(dim=-1) / 0.5

    # Classify longitudinal bins
    longitudinal_bin = torch.zeros((expert_trajectory.shape[0], 3),
                                   dtype=dtype,
                                   device=device)
    longitudinal_bin[speed_future < current_speed -
                     delta_speed, 0] = 1  # decelerate
    longitudinal_bin[speed_future > current_speed +
                     delta_speed, 2] = 1  # accelerate
    longitudinal_bin[torch.
                     logical_and(speed_future >= current_speed -
                                 delta_speed, speed_future <= current_speed +
                                 delta_speed), 1] = 1  # maintain speed

    # Calculate y position in 5 seconds
    latitudinal_bin = torch.zeros((expert_trajectory.shape[0], 3),
                                  dtype=dtype,
                                  device=device)
    for batch in range(expert_trajectory.shape[0]):
        future_turning_action = 1  # keep forward
        for i in range(latitudinal_critical_step):
            y_fut = expert_trajectory[batch, i, 1]
            if y_fut < -delta_turning:
                future_turning_action = 0
                break
            elif y_fut > delta_turning:
                future_turning_action = 2
                break
            else:
                future_turning_action = 1
        # Classify latitudinal bins
        latitudinal_bin[batch, future_turning_action] = 1

    return longitudinal_bin, latitudinal_bin

def dequantize_state_value(state_token, device, px, py, ph, pw, pl):
    """
    Dequantize the tokenized states back to their original values.

    :param state_token: Combined token value representing x, y, heading, width, and length states.
    :param device: Torch device (e.g., 'cuda' or 'cpu').
    :return: Dequantized x, y, heading, width, and length values.
    """
    # Parameters for each dimension
    nx, ny, nh, nw, nl = px[1], py[1], ph[1], pw[1], pl[1]
    
    # Convert state_token back to 0th index
    state_token = - state_token -1
    
    # Clamp the state_token within valid range
    state_token = torch.clamp(state_token, 0, nx * ny * nh * nw * nl - 1)
    
    # Extract individual tokens from state token
    length_token = state_token % nl
    width_token = (state_token // nl) % nw
    heading_token = (state_token // (nl * nw)) % nh
    y_token = (state_token // (nh * nl * nw)) % ny
    x_token = state_token // (ny * nh * nl * nw)
    
    # Create bin edges for each dimension
    bins_x = torch.linspace(px[0][0], px[0][1], steps=px[1] + 1, device=device)
    bins_y = torch.linspace(py[0][0], py[0][1], steps=py[1] + 1, device=device)
    bins_h = torch.linspace(ph[0][0], ph[0][1], steps=ph[1] + 1, device=device)
    bins_w = torch.linspace(pw[0][0], pw[0][1], steps=pw[1] + 1, device=device)
    bins_l = torch.linspace(pl[0][0], pl[0][1], steps=pl[1] + 1, device=device)
    
    # Map tokens to bin centers (or edges) to dequantize
    x = (bins_x[x_token] + bins_x[x_token + 1]) / 2
    y = (bins_y[y_token] + bins_y[y_token + 1]) / 2
    heading = (bins_h[heading_token] + bins_h[heading_token + 1]) / 2
    width = (bins_w[width_token] + bins_w[width_token + 1]) / 2
    length = (bins_l[length_token] + bins_l[length_token + 1]) / 2
    
    return x, y, heading, width, length


def dequantize_state_token(state_token):
    """
    Dequantize the tokenized states back to their original tokens.

    :param state_token: Combined token value representing x, y, and heading states.
    :param device: Torch device (e.g., 'cuda' or 'cpu').

    :return: Dequantized x, y, and heading values.
    """
    from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.state_machine.state_type import VocabularyStateType
    nx, ny, nh, nw, nl = VocabularyStateType.X.nx, VocabularyStateType.Y.ny, VocabularyStateType.HEADING.nh, VocabularyStateType.WIDTH.nw, VocabularyStateType.LENGTH.nl
    # Extract individual tokens from state token
    state_token = - state_token - 1  # convert it back to 0th index
    if isinstance(state_token, torch.Tensor):
        state_token = torch.clamp(state_token, 0, nx * ny * nh * nw * nl - 1)
    elif isinstance(state_token, np.ndarray) or isinstance(state_token, int):
        state_token = np.clip(state_token, 0, nx * ny * nh * nw * nl - 1)
    else:
        raise ValueError('state_token must be a torch.Tensor or np.ndarray')

    # Extract individual tokens from state token
    length_token = state_token % nl
    width_token = (state_token // nl) % nw
    heading_token = (state_token // (nl * nw)) % nh
    y_token = (state_token // (nh * nl * nw)) % ny
    x_token = state_token // (ny * nh * nl * nw)

    # Convert tokens to their actual values using the start value from VocabularyStateType
    x_token += VocabularyStateType.X.start
    y_token += VocabularyStateType.Y.start
    heading_token += VocabularyStateType.HEADING.start
    width_token += VocabularyStateType.WIDTH.start
    length_token += VocabularyStateType.LENGTH.start

    return x_token, y_token, heading_token, width_token, length_token

def quantize_state_token(x_token, y_token, heading_token, width_token, length_token):
    """
    Quantize the state values into a single token.

    :param x_token: Token value representing the x state.
    :param y_token: Token value representing the y state.
    :param heading_token: Token value representing the heading state.
    :param width_token: Token value representing the width state.
    :param length_token: Token value representing the length state.
    :return: Quantized state token.
    """
    from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.state_machine.state_type import VocabularyStateType
    nx, ny, nh, nw, nl = (VocabularyStateType.X.nx, VocabularyStateType.Y.ny, 
                          VocabularyStateType.HEADING.nh, VocabularyStateType.WIDTH.nw, 
                          VocabularyStateType.LENGTH.nl)
    
    x_token = x_token - VocabularyStateType.X.start
    y_token = y_token - VocabularyStateType.Y.start
    heading_token = heading_token - VocabularyStateType.HEADING.start
    width_token = width_token - VocabularyStateType.WIDTH.start
    length_token = length_token - VocabularyStateType.LENGTH.start
    
    # Compute the state token
    state_token = (x_token * ny * nh * nw * nl + 
                   y_token * nh * nw * nl + 
                   heading_token * nw * nl + 
                   width_token * nl + 
                   length_token)
    
    state_token = - state_token - 1  # convert to 0th index
    return state_token


def frame_dropout(tokenized_agents, bos_token, pad_token, frame_dropout_rate):
    bos_indices = (tokenized_agents == bos_token).nonzero(
        as_tuple=True)[0][1:]  # do not drop frame0

    # Generate a mask with the same shape as the tensor, with each element
    # having a 0.2 probability of being True
    mask = (
        torch.rand(
            bos_indices.shape[0]) < frame_dropout_rate).to(
        bos_indices.device)
    # import pdb;pdb.set_trace()
    drop_indexes = torch.where(mask)[0]
    for i in range(len(drop_indexes)):
        start_bos = bos_indices[drop_indexes[i]]
        end_bos = bos_indices[drop_indexes[i] + 1] if drop_indexes[i] < len(
            bos_indices) - 1 else len(tokenized_agents)
        # do not drop bos token, we need to know how many frames are dropped
        tokenized_agents[start_bos + 1:end_bos] = pad_token
    return tokenized_agents


def sample_segments(tokenized_agents, block_size, bos_token, pad_token, k):
    device = tokenized_agents.device
    # Find all the indices of BOS_TOKEN
    bos_indices = (tokenized_agents == bos_token).nonzero(as_tuple=True)[0]
    # bos_indices = bos_indices[12:13]

    # Filter out bos_indices that don't satisfy the block size requirement
    valid_start_positions = bos_indices[(
        bos_indices + block_size) <= len(tokenized_agents)]

    # If there are fewer valid start positions than k, sample from all
    # bos_indices
    chosen_start_positions = valid_start_positions if len(
        valid_start_positions) >= k else bos_indices

    # Randomly choose k indices from chosen_start_positions
    sampled_positions = chosen_start_positions[torch.randperm(
        len(chosen_start_positions))[:k]]

    # frame_index
    frame_index = torch.cumsum((tokenized_agents == bos_token), dim=0) - 1

    segments = []
    targets = []
    segment_frame_indexes = []

    for start_pos in sampled_positions:
        end_pos = start_pos + block_size

        # Extract segment
        segment = tokenized_agents[start_pos:min(
            end_pos, len(tokenized_agents))]
        segment_frame_index = frame_index[start_pos:min(
            end_pos, len(tokenized_agents))]

        # Pad the segment if it's not of block_size
        if len(segment) < block_size:
            padding = torch.full(
                (block_size - len(segment),), pad_token, device=device)
            segment = torch.cat((segment, padding), dim=0)
            segment_frame_index = torch.cat(
                (segment_frame_index, padding), dim=0)

        # Create the target which is the segment shifted by one token
        target = segment[1:]

        # Add the next token after the segment or a padding token if we've
        # reached the end
        next_token = tokenized_agents[end_pos] if end_pos < len(
            tokenized_agents) else pad_token
        target = torch.cat(
            (target,
             torch.tensor(
                 [next_token],
                 device=device)),
            dim=0)

        # Save the segment and target
        segments.append(segment)
        targets.append(target)
        segment_frame_indexes.append(segment_frame_index)

    return segments, targets, segment_frame_indexes


def slice_first_k_frame(tokens, bos_token, k, max_length):
    # Find the positions of the bos_token in the tensor.
    bos_positions = (tokens == bos_token).nonzero(as_tuple=True)[0]
    k = min(k, len(bos_positions) - 1)  # -1 to exclude the last bos token

    start = 0
    while start < k:
        # +1 to include the bos token
        sliced = tokens[bos_positions[start]:bos_positions[k] + 1]
        history_tokens = tokens[:bos_positions[k] + 1]
        if sliced.size(0) <= max_length:
            return sliced, history_tokens
        start += 1

    return tokens[:1], tokens[:1]


def slice_last_k_frame(tokens, bos_token, k, max_length):
    # Find the positions of the bos_token in the tensor.
    bos_positions = (tokens == bos_token).nonzero(as_tuple=True)[0]

    while k > 0:
        # If there aren't enough occurrences of bos_token for the current k
        if bos_positions.size(0) < k:
            k -= 1
            continue

        # Find the k-th occurrence of bos_token from the end
        kth_bos_position_from_end = bos_positions[-k].item()

        # Slice the tensor from the kth bos position to the end
        sliced = tokens[kth_bos_position_from_end:]

        # If the slice length is within the max_length, return the slice
        if sliced.size(0) <= max_length:
            return sliced, kth_bos_position_from_end

        # If the slice exceeds the max_length, decrement k and try again
        k -= 1
    raise ValueError(f"Unable to find a valid slice within the max length for any k.")


def shift_down(map_data, shift_pixels):
    # Function to shift the map downward by a certain number of pixels
    if shift_pixels <= 0:
        return map_data

    # Create an array of zeros with the same shape as the input map
    zeros_to_add = np.zeros((shift_pixels, *map_data.shape[1:]), dtype=map_data.dtype)

    # Remove the bottom-most rows equal to shift_pixels
    shifted_map = map_data[shift_pixels:]

    # Add the zeros to the top of the map
    return np.vstack((shifted_map, zeros_to_add))

def angle_to_sin_cos(angles):
    return torch.sin(angles), torch.cos(angles)

def sin_cos_to_angle(sin_component, cos_component):
    return torch.atan2(sin_component, cos_component)

def process_trajectory(trajectory_sine_heading, shape):
    trajectory_sine_heading = trajectory_sine_heading.reshape(shape)
    trajectory = torch.zeros_like(trajectory_sine_heading)
    # x,y, sin, cos
    trajectory[..., :2] = trajectory_sine_heading[..., :2]
    trajectory[..., 2] = sin_cos_to_angle(trajectory_sine_heading[..., 2], trajectory_sine_heading[..., 3])
    return trajectory[..., :3]
    
def encoding_traffic_light(traffic_light_data):
    """
    Encode traffic light data into a single token
    suppose we have int64 traffic token, ignore the negtive, so we only have 63bit, 
    the first 5 bit is for num of traffic light, then each traffic light has 2 bit for status, which stands for red, yellow, green, unknown
    use bit operation to get the token
    like this:
        s|nnnnn|1111|2222|3333|....
    s is the sign bit, 0 for positive, 1 for negtive
    n is the number of traffic light
    :param traffic_light_data: Traffic light data from the map. with shape # num_tl, 5 (idx, x, y, heading, status)
    """

    # Ensure the number of traffic lights does not exceed 31
    num_traffic_lights = min(traffic_light_data.shape[0], 29)
    
    # Initialize the token
    token = num_traffic_lights
    token <<= 58
    # Encode the traffic light status
    for i in range(num_traffic_lights):
        # Extract the status from the traffic_light_data
        status = int(traffic_light_data[i, -1])

        # OR with the status value shifted to the correct position
        token |= status << (2 * i)
    
    return token

def encoding_traffic_light_batch(traffic_light_data_batch):
    """
    Encode batch of traffic light data into a single tensor token.
    
    :param traffic_light_data_batch: A tensor of shape (batch_size, max_num_tl, 5), 
                                     where 5 represents (idx, x, y, heading, status),
                                     max_num_tl is the maximum number of traffic lights in a batch.
    :return: A tensor of shape (batch_size,), each entry is an encoded token
    """
    max_num_tl = 29
    # Ensure the number of traffic lights does not exceed 29
    num_traffic_lights = traffic_light_data_batch.shape[1]

    # Initialize the token
    tokens = num_traffic_lights
    tokens = tokens << (max_num_tl * 2)

    # Encode the traffic light status
    for i in range(num_traffic_lights):
        # Extract the status from the traffic_light_data
        statuses = traffic_light_data_batch[:, i].long()

        # OR with the status value shifted to the correct position
        tokens |= statuses << (2 * i)
    
    return tokens

def decoding_traffic_light(token):
    """
    Decode a traffic light token back into the statuses of individual traffic lights.

    The token is a 64-bit integer encoding the number of traffic lights and their statuses.
    The first 5 bits of the token represent the number of traffic lights (maximum of 31).
    The subsequent bits are grouped in 2-bit segments, each representing the status of a traffic light.
    The status is encoded as follows: 0 for RED, 1 for YELLOW, 2 for GREEN, and 3 for UNKNOWN.

    :param token: A 64-bit integer token encoding the number of traffic lights and their statuses.
    :return: A NumPy array of shape (num_traffic_lights, 1), where each row contains the status of a traffic light.
             The status is represented as an integer, with the same encoding as described above.

    Example:
    --------
    # Example usage
    token = encoding_traffic_light(np.array([[[0, 0, 0, 0, TrafficLightStatus.RED], 
                                              [0, 0, 0, 0, TrafficLightStatus.GREEN], 
                                              [0, 0, 0, 0, TrafficLightStatus.YELLOW]]], dtype=int))
    print(decoding_traffic_light(token))  # Output should be the statuses of the traffic lights
    """
    # Extract the number of traffic lights
    num_traffic_lights = token >> 58  # The first 5 bits represent the number of traffic lights
    token &= (1 << 58) - 1  # Mask out the bits representing the number of traffic lights
    
    # Initialize the list to store the status of each traffic light
    traffic_light_statuses = np.zeros((num_traffic_lights, 1), dtype=np.int64)
    
    
    # Decode the status of each traffic light
    for i in range(num_traffic_lights):
        # Extract the 2-bit status code for the current traffic light
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        status_code = token & 3  # 3 in binary is 11, which masks out all but the last 2 bits
        traffic_light_statuses[i, 0] = status_code
        
        # Shift the token to prepare for the next iteration
        token >>= 2
        
    return traffic_light_statuses


def decoding_traffic_light_batch(tokens):
    """
    Decode batched traffic light tokens back into the statuses of individual traffic lights.
    ...
    :param tokens: A tensor of shape (batch_size,), each entry is an encoded token.
    :return: A tensor of shape (batch_size, max_num_tl), each row contains the status of traffic lights in the batch.
    """
    max_num_tl = 29 # Maximum number of traffic lights
    batch_size = tokens.size(0)
    
    # Extract the number of traffic lights
    num_traffic_lights = (tokens >> (max_num_tl * 2)).long()
    
    # Mask out the bits representing the number of traffic lights
    tokens &= (1 << (max_num_tl * 2)) - 1
    # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
    # Initialize the tensor to store the statuses
    traffic_light_statuses = torch.full((batch_size, max_num_tl), -1, dtype=torch.int64, device=tokens.device)
    # Decode the status of each traffic light
    for i in range(max_num_tl):
        # Extract the 2-bit status code for all batches
        status_codes = tokens & 3
        mask = i < num_traffic_lights  # Mask for valid traffic lights (not padding)
        traffic_light_statuses[mask, i] = status_codes[mask]
        
        # Shift the tokens for the next iteration
        tokens >>= 2
    
    return traffic_light_statuses

def keep_last_true(tensor):
    """
    Keep only the last True value per row in a boolean tensor.
    
    :param tensor: A boolean tensor of shape (N, L)
    :return: A boolean tensor of shape (N, L) with only the last True value kept per row
    """
    # Convert boolean tensor to integer tensor
    int_tensor = tensor.int()
    
    # Reverse the tensor along the L dimension
    reversed_tensor = int_tensor.flip(dims=[1])
    
    # Find the indices of the first True value (which is now 1) in the reversed tensor
    last_true_indices = reversed_tensor.argmax(dim=1)
    
    # Handle rows with no True values (argmax returns 0 in such cases)
    last_true_indices = last_true_indices * reversed_tensor.max(dim=1).values
    
    # Create the result tensor with all False values
    result = torch.zeros_like(tensor, dtype=torch.bool)
    
    # Mark the last True position in each row
    result[torch.arange(tensor.size(0)), -last_true_indices-1] = True
    
    return result
