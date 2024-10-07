import numpy as np
from enum import Enum
from numba.typed import Dict, List
from numba.core import types
from numba import jit, prange

import time
from enum import Enum
from numba.typed import Dict
from numba.core import types
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.kinetics_state_type import KineticsVocabularyStateType
from nuplan_extent.planning.training.modeling.models.tokenizers.base_tokenizer_utils import (
    safe_add_one,
    normalize_angle,
    average_corner_distance,
    check_collision,
    find_last_frame_index,
    NpSequenceArray,
    ClassType,
    StatusType,
    CasadiStatusType)
from nuplan_extent.planning.training.modeling.models.utils.kinematic_solver import KinematicSolver

from dataclasses import dataclass, field
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
pacifica_params = get_pacifica_parameters()
pacifica_wheel_base = pacifica_params.wheel_base

class TokenType(Enum):
    """
    Enum representing different types of tokens used in sequence modeling
    of vehicle and pedestrian tracking data.

    Attributes:
        DUMMY_TOKEN: Placeholder token for initialization or error handling.
        EGO_TOKEN: Represents the ego vehicle in the sequence.
        AGENT_TOKEN: Represents other agents (vehicles, pedestrians) in the sequence.
        BLANK_TOKEN: Special token for the blank state.
    """
    DUMMY_TOKEN = -1
    EGO_TOKEN = 1
    AGENT_TOKEN = 2
    BLANK_TOKEN = 3

class NpKineticsSequenceArray(NpSequenceArray):
    dim = 21
    lon_speed_dim = 14
    control_dim = 15
    updated_x_dim = 16
    updated_y_dim = 17
    updated_h_dim = 18
    casadi_status_dim = 19
    # track_to_predict_dim = 20
    valid_target_dim = 20
    

# Constants for NpSequenceArray
TOKEN_TYPE_IDX = NpSequenceArray.token_type_dim
FRAME_INDEX_IDX = NpSequenceArray.frame_index_dim
X_DIM = NpSequenceArray.x_dim
Y_DIM = NpSequenceArray.y_dim
HEADING_DIM = NpSequenceArray.heading_dim
VX_DIM = NpSequenceArray.vx_dim
VY_DIM = NpSequenceArray.vy_dim
WIDTH_DIM = NpSequenceArray.width_dim
LENGTH_DIM = NpSequenceArray.length_dim
TRACK_ID_DIM = NpSequenceArray.track_id_dim
TRACK_TOKEN_DIM = NpSequenceArray.track_token_dim
CLASS_TYPE_DIM = NpSequenceArray.class_type_dim
STATUS_DIM = NpSequenceArray.status_dim
RAW_ID_DIM = NpSequenceArray.raw_id_dim
NpSequence_DIM = NpSequenceArray.dim

# Constants for NpKineticsSequenceArray
NpKineticsSequence_DIM = NpKineticsSequenceArray.dim
LON_SPEED_DIM = NpKineticsSequenceArray.lon_speed_dim
CONTROL_DIM = NpKineticsSequenceArray.control_dim
UPDATED_X_DIM = NpKineticsSequenceArray.updated_x_dim
UPDATED_Y_DIM = NpKineticsSequenceArray.updated_y_dim
UPDATED_H_DIM = NpKineticsSequenceArray.updated_h_dim
CASADI_STATUS_DIM = NpKineticsSequenceArray.casadi_status_dim
VALID_TARGET_DIM = NpKineticsSequenceArray.valid_target_dim
# TRACK_TO_PREDICT_DIM = NpKineticsSequenceArray.track_to_predict_dim

BLANK_TOKEN_VALUE = KineticsVocabularyStateType.BLANK.start

CONTROL_START = KineticsVocabularyStateType.CONTROL.start
CONTROL_END = KineticsVocabularyStateType.CONTROL.end

CONTROL_ACC_RANGE, CONTROL_ACC_STEP = KineticsVocabularyStateType.CONTROL.control_acc_range, KineticsVocabularyStateType.CONTROL.control_acc_step
CONTROL_STEERING_RANGE, CONTROL_STEERING_STEP = KineticsVocabularyStateType.CONTROL.control_steering_range, KineticsVocabularyStateType.CONTROL.control_steering_step
GRID_ACC, GRID_STEERING = KineticsVocabularyStateType.CONTROL.ncontrol_acc + 1, KineticsVocabularyStateType.CONTROL.ncontrol_steering + 1

ego_width, ego_length = 2.297, 5.176 # hard code for nuplan
sampling_time = 0.5

AGENT_FEATURE_LEN = 8 # hard code for the dimention of vector data input

@jit(nopython=True, fastmath=True)
def estimate(x0, y0, h0, v0, x1, y1, h1, v1, dt, wheel_base):
    """
    Estimate the dynamics and control
    """
    # if data_type == ClassType.VEHICLE.value:
    control_acc = (v1 - v0) / dt 
    next_target_heading = np.arctan2(y1 - y0, x1 - x0)
    direct_dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
    aim_target_heading = (next_target_heading - h0 + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    # slow speed not accurate
    if abs(direct_dh) < abs(aim_target_heading) and dist > 0.2:
        dh = aim_target_heading
    else:
        dh = direct_dh
    
    h_dot = dh / dt

    if v1 < 0.5:
        control_steering = 0
    else:
        control_steering = np.arctan(h_dot * wheel_base / v1)
    return control_acc, control_steering

@jit(nopython=True, fastmath=True)
def propagate(x0, y0, h0, v0, dt, wheel_base, control_acc, control_steering, data_type):
    """
    Propagate the current dynamics and state using Kinetics Bicycle Model
    """
    v1 = v0 + control_acc * dt
    h1 = h0 + dt * v1 * np.tan(control_steering) / wheel_base
    h1 = (h1 + np.pi) % (2 * np.pi) - np.pi

    x1 = v1 * np.cos(h1) * dt + x0
    y1 = v1 * np.sin(h1) * dt + y0

    return x1, y1, h1, v1


@jit(nopython=True)
def within_valid_range(data, valid_range):
    """
    Check if the given data points (x, y) are within the specified valid range.
    
    Args:
        data (np.ndarray): The data array containing x and y coordinates.
        valid_range (np.ndarray): An array specifying the valid [xmin, xmax, ymin, ymax] range.
    
    Returns:
        bool: True if the data is within the valid range, False otherwise.
    """
    # Checking if the data's coordinates are within the provided range
    if data[X_DIM] > valid_range[0] and data[X_DIM] < valid_range[1] and \
       data[Y_DIM] > valid_range[2] and data[Y_DIM] < valid_range[3]:
        return True
    return False

@jit(nopython=True)
def get_ego_array(frame_index, current_ego_array, track_id):
    """
    Generate an array representing the ego vehicle's state at a specific frame.

    Args:
        frame_index (int): The index of the current frame.
        current_ego_array (np.ndarray): Array containing the ego's state data.
        track_id (int): The track identifier for the ego vehicle.

    Returns:
        np.ndarray: A numpy array representing the ego's state in the sequence.
    """
    raw_id = -1
    if len(current_ego_array) == 7:
        x, y, heading, vx, vy, ax, ay = current_ego_array
        w, l, z, type = ego_width, ego_length, 0, ClassType.VEHICLE.value
    else:
        x, y, heading, vx, vy, ax, ay, w, l, z, type, raw_id = current_ego_array

        type = type - 1 # waymo use 1: vehicle, 2: pedestrian, 3: cyclist
    ego_array = np.zeros((NpSequence_DIM, ))
    ego_array[TOKEN_TYPE_IDX] = TokenType.EGO_TOKEN.value
    ego_array[FRAME_INDEX_IDX] = frame_index
    ego_array[X_DIM] = x
    ego_array[Y_DIM] = y
    ego_array[HEADING_DIM] = normalize_angle(heading)
    ego_array[VX_DIM] = vx
    ego_array[VY_DIM] = vy
    ego_array[WIDTH_DIM] = w
    ego_array[LENGTH_DIM] = l
    ego_array[CLASS_TYPE_DIM] = type
    ego_array[TRACK_TOKEN_DIM] = -1
    ego_array[TRACK_ID_DIM] = track_id
    ego_array[RAW_ID_DIM] = raw_id
    return ego_array


@jit(nopython=True)
def get_agent_array(frame_index, current_agent_array, track_id, class_index):
    """
    Generate an array representing an agent's state in a specific frame.

    Args:
        frame_index (int): The index of the current frame.
        current_agent_array (np.ndarray): Array containing the agent's state data.
        track_id (int): The track identifier for the agent.
        class_index (int): Class type index of the agent (e.g., vehicle, pedestrian).

    Returns:
        np.ndarray: A numpy array representing the agent's state in the sequence.
    """
    raw_id = -1
    if len(current_agent_array) == AGENT_FEATURE_LEN:
        track_token, vx, vy, heading, width, length, x, y = current_agent_array
    else:
        track_token, vx, vy, heading, width, length, x, y, raw_id = current_agent_array
    agent_array = np.zeros((NpSequence_DIM, ))
    agent_array[TOKEN_TYPE_IDX] = TokenType.AGENT_TOKEN.value
    agent_array[FRAME_INDEX_IDX] = frame_index
    agent_array[X_DIM] = x
    agent_array[Y_DIM] = y
    agent_array[HEADING_DIM] = normalize_angle(heading)
    agent_array[VX_DIM] = vx
    agent_array[VY_DIM] = vy
    agent_array[WIDTH_DIM] = width
    agent_array[LENGTH_DIM] = length
    agent_array[CLASS_TYPE_DIM] = class_index
    agent_array[TRACK_ID_DIM] = track_id
    agent_array[TRACK_TOKEN_DIM] = track_token
    agent_array[RAW_ID_DIM] = raw_id
    return agent_array


@jit(nopython=True)
def get_hash(class_index, track_token):
    """
    Generates a hash value based on class index and track token for unique identification.

    Args:
        class_index (int): Class type index (e.g., vehicle, pedestrian).
        track_token (int): Unique track token assigned to an agent.

    Returns:
        int: A 64-bit integer hash value combining class index and track token.
    """
    # Combining the class index and track token to generate a unique hash.
    hash_value = int(class_index) << 32 | int(track_token)
    return hash_value


@jit(nopython=True)
def get_track_id(class_index, current_agent_array, track_id_mapping):
    """
    Determine the track ID for an agent, assigning a new ID if the agent is newly detected.

    Args:
        class_index (int): The class index of the agent.
        current_agent_array (np.ndarray): Current state array of the agent.
        track_id_mapping (Dict): Mapping from hashed values to track IDs.

    Returns:
        tuple: A tuple containing the track ID, newborn status, and updated track ID mapping.
    """
    raw_id = -1
    if len(current_agent_array) == AGENT_FEATURE_LEN:
        track_token, vx, vy, heading, width, length, x, y = current_agent_array
    else:
        track_token, vx, vy, heading, width, length, x, y, raw_id = current_agent_array
    key = get_hash(class_index, track_token)
    track_ids = track_id_mapping.values()
    if key in track_id_mapping:
        is_newborn = False
        track_id = track_id_mapping[key]
    else:
        is_newborn = True
        if len(track_ids) == 0:
            track_id = 1
        else:
            track_id = max(track_ids) + 1
        track_id_mapping[key] = track_id
    return track_id, is_newborn, track_id_mapping

@jit(nopython=True)
def get_blank_tokenized_array(frame_index):
    """
    Create a blank of sequence (BLANK) tokenized array with specified frame index.
    
    Args:
        frame_index (int): The index of the frame for which to create the BOS token.
        
    Returns:
        np.ndarray: A numpy array representing the BLANK token with the frame index.
    """
    blank_array = np.zeros((NpKineticsSequence_DIM,))
    blank_array[TOKEN_TYPE_IDX] = TokenType.BLANK_TOKEN.value
    blank_array[FRAME_INDEX_IDX] = frame_index
    blank_array[CONTROL_DIM] = 0
    blank_array[RAW_ID_DIM] = -1
    return blank_array

@jit(nopython=True)
def get_blank_array(frame_index):
    """
    Create a blank of sequence (BLANK) array with specified frame index.
    
    Args:
        frame_index (int): The index of the frame for which to create the BOS token.
        
    Returns:
        np.ndarray: A numpy array representing the BLANK token with the frame index.
    """
    blank_array = np.zeros((NpSequence_DIM,))
    blank_array[TOKEN_TYPE_IDX] = TokenType.BLANK_TOKEN.value
    blank_array[FRAME_INDEX_IDX] = frame_index
    blank_array[RAW_ID_DIM] = -1
    
    return blank_array

@jit(nopython=True)
def process_single_batch(agents_array, ego_array, max_seq_len=416):
    """
    Process a single batch of agent and ego arrays to create a tokenized sequence.

    Args:
        agents_array (List[np.ndarray]): List of agent arrays for each class type and frame.
        ego_array (np.ndarray): Array containing ego vehicle data across frames.
        max_seq_len (int): Maximum length of the sequence to be processed.

    Returns:
        np.ndarray: Tokenized information array for the batch.
    """
    
    total_num_frames = ego_array.shape[0]
    tokenized_info_array = np.nan * np.ones((total_num_frames, max_seq_len, NpSequence_DIM)) # (T, A, D)
    
    track_id_mapping = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    array_type = types.float64[:]

    # Create a Numba dictionary with int64 keys and float64 array values
    wlh_mapping = Dict.empty(
        key_type=types.int64,
        value_type=array_type,
    )
    # inital_agents_tokens = set()

    for frame_index in range(total_num_frames):
        agent_idx = 0
        # Processing the ego vehicle
        current_ego_array = ego_array[frame_index]
        tokenized_info_array[frame_index, agent_idx] = get_ego_array(frame_index, current_ego_array, track_id=0)
        agent_idx = safe_add_one(agent_idx, max_seq_len)

        for class_index, agents in enumerate(agents_array):
            current_agents_array = agents[frame_index]
            for current_agent_array in current_agents_array:
                if np.isnan(current_agent_array).any():
                    continue
                track_id, is_newborn, track_id_mapping = get_track_id(class_index, current_agent_array, track_id_mapping)
                
                # ignore newborn, skip the new agent
                # if frame_index == 0:
                #     inital_agents_tokens.add(track_id)
                # else:
                #     if track_id not in inital_agents_tokens:
                #         continue
                
                if track_id >= max_seq_len:
                    print("Error: Track ID exceeds the maximum allowable limit.")
                    continue
                agent_idx = int(track_id)
                tokenized_info_array[frame_index, agent_idx] = get_agent_array(frame_index, current_agent_array, track_id=track_id, class_index=class_index)
                fix_wlh = True
                if fix_wlh:
                    if track_id in wlh_mapping:
                        wlh = wlh_mapping[track_id]
                        tokenized_info_array[frame_index, agent_idx, WIDTH_DIM] = wlh[0]
                        tokenized_info_array[frame_index, agent_idx, LENGTH_DIM] = wlh[1]
                    else:
                        width = tokenized_info_array[frame_index, agent_idx, WIDTH_DIM]
                        length = tokenized_info_array[frame_index, agent_idx, LENGTH_DIM]
                        wlh = np.array([width, length], dtype=np.float64)
                        wlh_mapping[track_id] = wlh  
        # Setting the blank token
        for i in range(max_seq_len):
            current_token = tokenized_info_array[frame_index, i]
            if np.isnan(current_token).any():
                tokenized_info_array[frame_index, i] = get_blank_array(frame_index)
    return tokenized_info_array


def tokenize_batched_data(processed_data, tracks_to_predict_raw_ids_batch, max_seq_len=512):
    tokenized_data = np.nan * np.ones((processed_data.shape[0], processed_data.shape[1], max_seq_len, NpKineticsSequence_DIM))
    for bs in range(processed_data.shape[0]):
        tokenized_data[bs] = tokenize_single_batch(processed_data[bs], tracks_to_predict_raw_ids_batch[bs], max_seq_len=max_seq_len)
    return tokenized_data


def tokenize_single_batch(processed_single_batch, tracks_to_predict_raw_ids, max_seq_len=512, receeding_horizon=5, dt=0.5, valid_threshold=0.5, current_frame_index=2):
    """
    Tokenize a batch of data by filtering out entries outside of a valid range and tokenizing the valid entries.
    
    Args:
        processed_single_batch (np.ndarray): Processed batch data to be tokenized.
        max_seq_len (int): Maximum sequence length for the output tokenized data.
        valid_range (np.ndarray): Array specifying the valid [xmin, xmax, ymin, ymax] range.
    
    Returns:
        np.ndarray: The tokenized data array.
    """
    def calculate_token(value1, value_range1, value2, value_range2, start, step1, step2):
        # Clamping the value within the range and calculating the token
        value1 = np.maximum(value_range1[0], np.minimum(value_range1[1], value1))
        value2 = np.maximum(value_range2[0], np.minimum(value_range2[1], value2))

        offset1 = int(round((value1 - value_range1[0]) / step1))
        offset2 = int(round((value2 - value_range2[0]) / step2))
        return start + (offset2 * GRID_ACC + offset1)
    
    total_num_frames = processed_single_batch.shape[0]
    tokenized_data = np.zeros((total_num_frames, max_seq_len, NpKineticsSequence_DIM))
    for agent_index in range(max_seq_len):
        for frame_index in range(total_num_frames):
            data = processed_single_batch[frame_index, agent_index]
            if data[TOKEN_TYPE_IDX] == TokenType.BLANK_TOKEN.value:
                tokenized_data[frame_index:, agent_index, TOKEN_TYPE_IDX] = TokenType.BLANK_TOKEN.value
                tokenized_data[frame_index:, agent_index, FRAME_INDEX_IDX] = np.arange(total_num_frames)[frame_index:]
                tokenized_data[frame_index:, agent_index, RAW_ID_DIM] = -1
                continue
            else:
                tokenized_data[frame_index, agent_index, :data.shape[0]] = data
                tokenized_data[frame_index, agent_index, UPDATED_X_DIM] = tokenized_data[frame_index, agent_index, X_DIM]
                tokenized_data[frame_index, agent_index, UPDATED_Y_DIM] = tokenized_data[frame_index, agent_index, Y_DIM]
                tokenized_data[frame_index, agent_index, UPDATED_H_DIM] = tokenized_data[frame_index, agent_index, HEADING_DIM]
                vx, vy = tokenized_data[frame_index, agent_index, VX_DIM], tokenized_data[frame_index, agent_index, VY_DIM]
                tokenized_data[frame_index, agent_index, LON_SPEED_DIM] = np.sqrt(vx**2 + vy**2)

    for agent_index in range(max_seq_len):
        # # skip the agent if it is not in the initial frame  
        # if tokenized_data[0, agent_index, TOKEN_TYPE_IDX] == TokenType.BLANK_TOKEN.value:
        #     for fi in range(total_num_frames):
        #         tokenized_data[fi, agent_index] = get_blank_tokenized_array(fi)
        #     continue
        raw_id = tokenized_data[0, agent_index, RAW_ID_DIM]
        track_to_predict_mask = False
        if int(raw_id) in tracks_to_predict_raw_ids:
            track_to_predict_mask = True
        tokenized_data[:, agent_index, TRACK_TO_PREDICT_DIM] = track_to_predict_mask
        length = tokenized_data[0, agent_index, LENGTH_DIM]
        length = max(length, 0.01)
        wheel_base = pacifica_wheel_base * length / ego_length
        smoother = KinematicSolver(trajectory_len=receeding_horizon-1, dt=dt, wheel_base=wheel_base)

        for frame_index in range(total_num_frames-receeding_horizon):

            # skip the agent if not enough horizon
            if np.any(tokenized_data[frame_index:frame_index+receeding_horizon, agent_index, TOKEN_TYPE_IDX] == TokenType.BLANK_TOKEN.value):
                for fi in range(frame_index, total_num_frames):
                    tokenized_data[fi, agent_index, CONTROL_DIM] = -1
                break
            
            # extract the horizon
            horizon_data = tokenized_data[frame_index:frame_index+receeding_horizon, agent_index]
            ref_traj = np.zeros((receeding_horizon, 3))
            ref_traj[:, 0] = horizon_data[:, X_DIM]
            ref_traj[:, 1] = horizon_data[:, Y_DIM]
            ref_traj[:, 2] = horizon_data[:, HEADING_DIM]
            x_curr = [horizon_data[0, UPDATED_X_DIM], horizon_data[0, UPDATED_Y_DIM], horizon_data[0, UPDATED_H_DIM], horizon_data[0, LON_SPEED_DIM]]
            smoother.set_reference_trajectory(x_curr=x_curr, reference_trajectory=ref_traj)
            try:
                sol = smoother.solve()
                ego_control = np.vstack([
                    sol.value(smoother.steering), sol.value(smoother.accel)]).T 
                control_acc, control_steering = ego_control[0, 1], ego_control[0, 0]
                tokenized_data[frame_index, agent_index, CASADI_STATUS_DIM] = CasadiStatusType.SUCCESS.value
            except Exception as e:
                # print(e)
                x0, y0, h0, v0 = x_curr
                x1, y1, h1, v1 = horizon_data[1, X_DIM], horizon_data[1, Y_DIM], horizon_data[1, HEADING_DIM], horizon_data[1, LON_SPEED_DIM]
                control_acc, control_steering = estimate(x0, y0, h0, v0, x1, y1, h1, v1, dt, wheel_base)
                tokenized_data[frame_index, agent_index, CASADI_STATUS_DIM] = CasadiStatusType.FAILED.value
                
                
            control_token = calculate_token(control_acc, CONTROL_ACC_RANGE, 
                                            control_steering, CONTROL_STEERING_RANGE, 
                                            CONTROL_START, CONTROL_ACC_STEP, 
                                            CONTROL_STEERING_STEP)
            
            if tokenized_data[frame_index, agent_index, CASADI_STATUS_DIM] == CasadiStatusType.SUCCESS.value:
                tokenized_data[frame_index, agent_index, CONTROL_DIM] = control_token
            else:
                tokenized_data[frame_index, agent_index, CONTROL_DIM] = -1
            

            # detokenize the control token
            control_token = control_token - CONTROL_START
            detokenize_control_acc = CONTROL_ACC_RANGE[0] + (control_token % GRID_ACC) * CONTROL_ACC_STEP
            detokenize_control_steering = CONTROL_STEERING_RANGE[0] + (control_token // GRID_ACC) * CONTROL_STEERING_STEP

            x0, y0, h0, v0 = x_curr
            x1, y1, h1, v1 = propagate(x0, y0, h0, v0, dt, wheel_base, 
                                       detokenize_control_acc, detokenize_control_steering, data[CLASS_TYPE_DIM])
            
            if frame_index+1 < total_num_frames:
                # only update successful prediction
                if tokenized_data[frame_index, agent_index, CASADI_STATUS_DIM] == CasadiStatusType.SUCCESS.value:
                    tokenized_data[frame_index+1, agent_index, UPDATED_X_DIM] = x1
                    tokenized_data[frame_index+1, agent_index, UPDATED_Y_DIM] = y1
                    tokenized_data[frame_index+1, agent_index, UPDATED_H_DIM] = h1
                    tokenized_data[frame_index+1, agent_index, LON_SPEED_DIM] = v1
                    x_targ, y_targ = tokenized_data[frame_index+1, agent_index, X_DIM], tokenized_data[frame_index+1, agent_index, Y_DIM]
                    if np.sqrt((x1 - x_targ)**2 + (y1 - y_targ)**2) < valid_threshold:
                        tokenized_data[frame_index, agent_index, VALID_TARGET_DIM] = 1
                else:
                    # if failed, reset the target to gt, and set the valid target to 0
                    tokenized_data[frame_index, agent_index, VALID_TARGET_DIM] = 0

        valid_target_mask = tokenized_data[:, agent_index, VALID_TARGET_DIM]
        
        x_pred, y_pred = tokenized_data[:, agent_index, UPDATED_X_DIM], tokenized_data[:, agent_index, UPDATED_Y_DIM]
        x_targ, y_targ = tokenized_data[:, agent_index, X_DIM], tokenized_data[:, agent_index, Y_DIM]
        # if not track_to_predict_mask:
        if np.any(valid_target_mask == 1):
            ade = np.mean(np.sqrt((x_pred[valid_target_mask == 1] - x_targ[valid_target_mask == 1])**2 + (y_pred[valid_target_mask == 1] - y_targ[valid_target_mask == 1])**2))
            if ade > valid_threshold:
                tokenized_data[:, agent_index, VALID_TARGET_DIM] = 0
    return tokenized_data

@jit(nopython=True)
def get_tokenized_features(tokenized_arrays):
    """
    Get the tokenized indices and features 
    Args:
        tokenized_arrays (numpy.ndarray): The tokenized arrays.

    Returns:
        tuple: A tuple containing the following numpy arrays:
            - tokenized_embedding_features: tokenized feature
            - state_embedding_feature: additional features
    """
    batch_size, frame_len, agent_len, _ = tokenized_arrays.shape

    state_embedding_features = np.full((batch_size, frame_len, agent_len, 6), 0, dtype=np.float64)  # Assuming 5 features (x, y, heading, w, l)
    tokenized_embedding_features = np.full((batch_size, frame_len, agent_len), 0, dtype=np.int64)
    class_type_features = np.full((batch_size, frame_len, agent_len), 0, dtype=np.int64)
    valid_mask = np.full((batch_size, frame_len, agent_len), 0, dtype=np.int64)

    for bs in range(batch_size):
        for frame_index in range(frame_len):
            for agent_index in range(agent_len):
                data = tokenized_arrays[bs, frame_index, agent_index]
                if data[TOKEN_TYPE_IDX] == TokenType.BLANK_TOKEN.value:
                    continue
                valid_mask[bs, frame_index, agent_index] = 1

                x, y, h, w, l, lon_speed = data[UPDATED_X_DIM], data[UPDATED_Y_DIM], data[UPDATED_H_DIM], data[WIDTH_DIM], data[LENGTH_DIM], data[LON_SPEED_DIM]

                # use the previous control token as embedding token
                if frame_index == 0:
                    control_token = 0
                else:
                    control_token = tokenized_arrays[bs, frame_index-1, agent_index, CONTROL_DIM]

                if control_token < 0:
                    control_token = 0
                if control_token >= CONTROL_END:
                     control_token = 0

                state_embedding_features[bs, frame_index, agent_index] = (x, y, h, w, l, lon_speed)
                tokenized_embedding_features[bs, frame_index, agent_index] = control_token 
                class_type_features[bs, frame_index, agent_index] = data[CLASS_TYPE_DIM]

    return tokenized_embedding_features, state_embedding_features, class_type_features, valid_mask


@jit(nopython=True)
def get_agent_target_features(tokenized_arrays):
    """
    Retrieves the indices of agent target embeddings and state embedding features from tokenized arrays.

    Args:
        tokenized_arrays (numpy.ndarray): The tokenized arrays containing the tokens.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - target_tokenized_features (numpy.ndarray): The target features

    """
    batch_size, frame_len, agent_len, _ = tokenized_arrays.shape
    target_tokenized_features = np.full((batch_size, frame_len, agent_len, 1), -1, dtype=np.int64)

    for bs in range(batch_size):
        for agent_index in range(agent_len):
            for frame_index in range(frame_len):
                is_blank = tokenized_arrays[bs, frame_index, agent_index, TOKEN_TYPE_IDX] == TokenType.BLANK_TOKEN.value
                control_token = tokenized_arrays[bs, frame_index, agent_index, CONTROL_DIM]
                is_valid_target = tokenized_arrays[bs, frame_index, agent_index, VALID_TARGET_DIM] == 1

                if is_blank or control_token <= 0 or not is_valid_target:
                    control_token = -1
                target_tokenized_features[bs, frame_index, agent_index] = (control_token, )
    return target_tokenized_features


@jit(nopython=True)
def update_last_frame_data(last_tokenized_arrays, pred_agent_tokens):
    """
    update last frame data with predicted agent tokens
    :param last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpKineticsSequence_DIM)
    :param pred_agent_tokens: np.ndarray, shape (num_agents, [x, y, heading] tokens)
    :return updated_last_frame_arrays: np.ndarray, shape (batch_size, max_seq_len, NpKineticsSequence_DIM)
    """
    batch_size, frame_len, agent_len, _ = last_tokenized_arrays.shape
    for bs in range(batch_size):
        for frame_index in range(frame_len):
            for agent_index in range(agent_len):
                last_tokenized_arrays[bs, frame_index, agent_index, CONTROL_DIM] = pred_agent_tokens[bs, frame_index, agent_index, 0]
                last_tokenized_arrays[bs, frame_index, agent_index, STATUS_DIM] = StatusType.GENERATED.value

    return last_tokenized_arrays


@jit(nopython=True)
def detokenize_data(tokenized_arrays):
    """
    Detokenize the tokenized arrays to get the original data.
    :param tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpKineticsSequence_DIM)
    :return detokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpKineticsSequence_DIM)
    """
    batch_size, frame_len, agent_len, _ = tokenized_arrays.shape
    for bs in range(batch_size):
        for agent_index in range(agent_len):
            token_type = tokenized_arrays[bs, 0, agent_index, TOKEN_TYPE_IDX]
            if token_type == TokenType.BLANK_TOKEN.value:
                tokenized_arrays[bs, :, agent_index, TOKEN_TYPE_IDX] = TokenType.BLANK_TOKEN.value
                tokenized_arrays[bs, :, agent_index, FRAME_INDEX_IDX] = np.arange(frame_len)
                tokenized_arrays[bs, :, agent_index, RAW_ID_DIM] = -1
                continue
            for frame_index in range(frame_len):
                control_token = tokenized_arrays[bs, frame_index, agent_index, CONTROL_DIM]
                control_token = control_token - CONTROL_START
                control_acc = CONTROL_ACC_RANGE[0] + (control_token % GRID_ACC) * CONTROL_ACC_STEP
                control_steering = CONTROL_STEERING_RANGE[0] + (control_token // GRID_ACC) * CONTROL_STEERING_STEP

                x0, y0, h0, v0 = tokenized_arrays[bs, frame_index, agent_index, UPDATED_X_DIM], tokenized_arrays[bs, frame_index, agent_index, UPDATED_Y_DIM], tokenized_arrays[bs, frame_index, agent_index, UPDATED_H_DIM], tokenized_arrays[bs, frame_index, agent_index, LON_SPEED_DIM]
                
                length = max(tokenized_arrays[bs, frame_index, agent_index, LENGTH_DIM], 0.01)
                wheel_base = pacifica_wheel_base * length / ego_length  
                dt = sampling_time
    
                x1, y1, h1, v1 = propagate(x0, y0, h0, v0, dt, wheel_base, control_acc, control_steering, tokenized_arrays[bs, frame_index, agent_index, CLASS_TYPE_DIM])

                tokenized_arrays[bs, frame_index, agent_index, X_DIM] = x1
                tokenized_arrays[bs, frame_index, agent_index, Y_DIM] = y1
                tokenized_arrays[bs, frame_index, agent_index, HEADING_DIM] = h1
                tokenized_arrays[bs, frame_index, agent_index, UPDATED_X_DIM] = x1
                tokenized_arrays[bs, frame_index, agent_index, UPDATED_Y_DIM] = y1
                tokenized_arrays[bs, frame_index, agent_index, UPDATED_H_DIM] = h1
                tokenized_arrays[bs, frame_index, agent_index, LON_SPEED_DIM] = v1

    return tokenized_arrays


@jit(nopython=True)
def update_last_frame_data_test(last_tokenized_arrays, last_tokenized_arrays_control):
    """
    update last frame data with predicted agent tokens in test mode
    :param last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpKineticsSequence_DIM)
    :param last_tokenized_arrays_control: np.ndarray, shape (num_agents, [x, y, heading] tokens)
    :return updated_last_frame_arrays: np.ndarray, shape (batch_size, max_seq_len, NpKineticsSequence_DIM)
    """
    batch_size, frame_len, agent_len, _ = last_tokenized_arrays.shape
    for bs in range(batch_size):
        for frame_index in range(frame_len):
            for agent_index in range(agent_len):
                last_tokenized_arrays[bs, frame_index, agent_index, FRAME_INDEX_IDX] += 1
                last_tokenized_arrays[bs, frame_index, agent_index, CONTROL_DIM] = last_tokenized_arrays_control[bs, frame_index, agent_index, CONTROL_DIM]
                last_tokenized_arrays[bs, frame_index, agent_index, STATUS_DIM] = StatusType.GENERATED.value

    return last_tokenized_arrays

@jit(nopython=True)
def find_surrounding_agents_in_ground_truth_array(predicted_tokenized_array, current_token):
    surrounding_agents_array = List()
    frame_index = int(current_token[FRAME_INDEX_IDX])

    for token_index in range(len(predicted_tokenized_array[0])):
        other_token = predicted_tokenized_array[frame_index, token_index]
        if other_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value]:
            if other_token[TRACK_ID_DIM] != current_token[TRACK_ID_DIM]:
                surrounding_agents_array.append(other_token)
    return surrounding_agents_array

@jit(nopython=True, parallel=True)
def calculate_smin_ade_batch(predicted_tokenized_array, tokenized_array, is_avg_corner_dist=False, ego_only=False):
    smin_ade = 0
    for i in prange(len(predicted_tokenized_array)):
        smin_ade += calculate_smin_ade(predicted_tokenized_array[i], tokenized_array[i], is_avg_corner_dist, ego_only)
    return smin_ade

@jit(nopython=True, parallel=True)
def calculate_collision_rate_batch(predicted_tokenized_array, ego_only=False):
    collision_rate = 0
    for i in prange(len(predicted_tokenized_array)):
        collision_rate += calculate_collision_rate(predicted_tokenized_array[i], ego_only)
    return collision_rate / len(predicted_tokenized_array)

@jit(nopython=True)
def calculate_smin_ade(predicted_tokenized_array, tokenized_array, is_avg_corner_dist=False, ego_only=False):
    smin_ade = np.inf
    ade_np = np.zeros(len(predicted_tokenized_array))

    _, frame_len, agent_len, _ = predicted_tokenized_array.shape
    for scene_index in range(len(predicted_tokenized_array)):
        scene_ade = 0
        num_agents = 0
        for frame_index in range(frame_len):
            for agent_index in range(agent_len):
                current_token = predicted_tokenized_array[scene_index, frame_index, agent_index]
                if current_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value]:
                    if ego_only and current_token[TOKEN_TYPE_IDX] != TokenType.EGO_TOKEN.value:
                        continue
                    # skip the conditined tokens
                    if current_token[STATUS_DIM] == StatusType.CONDITION.value:
                        continue
                    
                    groud_truth_token = tokenized_array[frame_index, agent_index]
                    if groud_truth_token[TOKEN_TYPE_IDX] not in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value]:
                        continue
                    if is_avg_corner_dist:
                        xa, ya, ha, wa, la = current_token[X_DIM], current_token[Y_DIM], current_token[HEADING_DIM], current_token[WIDTH_DIM], current_token[LENGTH_DIM]
                        xb, yb, hb, wb, lb = groud_truth_token[X_DIM], groud_truth_token[Y_DIM], groud_truth_token[HEADING_DIM], groud_truth_token[WIDTH_DIM], groud_truth_token[LENGTH_DIM]
                        dist = average_corner_distance(
                            np.array([xa, ya, ha, wa, la]),
                            np.array([xb, yb, hb, wb, lb])
                        )
                    else:
                        x_diff = current_token[X_DIM] - groud_truth_token[X_DIM]
                        y_diff = current_token[Y_DIM] - groud_truth_token[Y_DIM]
                        dist = np.sqrt(x_diff**2 + y_diff**2)
                    scene_ade += dist
                    num_agents += 1
        ade = scene_ade / (num_agents + 1e-5)
        ade_np[scene_index] = ade
    smin_ade = np.min(ade_np)
    return smin_ade

@jit(nopython=True)
def calculate_collision_rate(predicted_tokenized_array, ego_only=False, filter_dist=10):
    # predicted_tokenized_array: T, A, K
    frame_len, agent_len, _ = predicted_tokenized_array.shape
    num_agents = 0
    collied_agents = 0
    for frame_index in range(frame_len):
        for agent_index in range(agent_len):
            current_token = predicted_tokenized_array[frame_index, agent_index]
            if current_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value]:
                if ego_only and current_token[TOKEN_TYPE_IDX] != TokenType.EGO_TOKEN.value:
                    continue
                # skip the conditined tokens
                if current_token[STATUS_DIM] == StatusType.CONDITION.value:
                    continue
                surrounding_agents_array = find_surrounding_agents_in_ground_truth_array(predicted_tokenized_array, current_token)
                num_agents += 1
                for other_token in surrounding_agents_array:
                    if other_token[TOKEN_TYPE_IDX] not in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value]:
                        continue
                    xa, ya, ha, wa, la = current_token[X_DIM], current_token[Y_DIM], current_token[HEADING_DIM], current_token[WIDTH_DIM], current_token[LENGTH_DIM]
                    xb, yb, hb, wb, lb = other_token[X_DIM], other_token[Y_DIM], other_token[HEADING_DIM], other_token[WIDTH_DIM], other_token[LENGTH_DIM]
                    # prefailter
                    if (xa - xb) ** 2 + (ya - yb) ** 2 > filter_dist ** 2:
                        continue
                    is_collided = check_collision(
                        np.array([xa, ya, ha, wa, la]),
                        np.array([xb, yb, hb, wb, lb]))
                    if is_collided:
                        collied_agents += 1
    collision_rate = collied_agents / (num_agents + 1e-5)
    if not ego_only:
        # the collision is counted twice for each pair of agents
        collision_rate /= 2
    return collision_rate