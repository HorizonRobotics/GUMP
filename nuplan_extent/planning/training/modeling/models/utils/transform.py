import numpy as np


def state_se2_tensor_to_transform_matrix_np(
        input_data: np.ndarray) -> np.ndarray:
    x, y, h = input_data
    cosine = np.cos(h)
    sine = np.sin(h)
    return np.array([[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]])


def state_se2_tensor_to_transform_matrix_batch_np(
        input_data: np.ndarray) -> np.ndarray:
    cosine = np.cos(input_data[:, 2])
    sine = np.sin(input_data[:, 2])
    # Use broadcasting for efficient stacking
    stacked_data = np.stack(
        [cosine, -sine, input_data[:, 0], sine, cosine, input_data[:, 1]], axis=-1)
    # Preallocate the output array and fill it using broadcasting
    output = np.zeros((input_data.shape[0], 3, 3))
    output[:, :2, :3] = stacked_data.reshape(-1, 2, 3)
    output[:, 2, 2] = 1.0
    return output


def transform_matrix_to_state_se2_tensor_batch_np(
        input_data: np.ndarray) -> np.ndarray:
    angles = np.arctan2(input_data[:, 1, 0], input_data[:, 0, 0])
    # Use slicing for in-place modification
    input_data[:, 2, 2] = angles
    return input_data[:, :, 2]


def global_state_se2_tensor_to_local_np(
        global_states: np.ndarray, local_state: np.ndarray) -> np.ndarray:
    local_xform = state_se2_tensor_to_transform_matrix_np(local_state)
    local_xform_inv = np.linalg.inv(local_xform)
    transforms = state_se2_tensor_to_transform_matrix_batch_np(global_states)
    # Use np.einsum for clearer and potentially faster matrix multiplication
    transformed_data = np.einsum('ij,kjl->kil', local_xform_inv, transforms)
    return transform_matrix_to_state_se2_tensor_batch_np(transformed_data)
