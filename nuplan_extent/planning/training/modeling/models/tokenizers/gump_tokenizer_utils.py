import numpy as np
from enum import Enum
from numba.typed import Dict, List
from numba.core import types
from numba import jit, prange
import random


from enum import Enum
from numba.typed import Dict
from numba.core import types
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type_v1_1 import VocabularyStateType, PositionalStateType
from nuplan_extent.planning.training.modeling.models.tokenizers.base_tokenizer_utils import (
    safe_add_one,
    normalize_angle,
    average_corner_distance,
    check_collision,
    find_last_frame_index,
    NpSequenceArray,
    NpTokenizedSequenceArray,
    TokenType,
    ClassType,
    StatusType
)


X_START, X_END, X_RANGE, X_STEP = VocabularyStateType.X.start, VocabularyStateType.X.end, VocabularyStateType.X.x_range, VocabularyStateType.X.x_step
Y_START, Y_END, Y_RANGE, Y_STEP = VocabularyStateType.Y.start, VocabularyStateType.Y.end, VocabularyStateType.Y.y_range, VocabularyStateType.Y.y_step
HEADING_START, HEADING_END, HEADING_RANGE, HEADING_STEP = VocabularyStateType.HEADING.start, VocabularyStateType.HEADING.end, VocabularyStateType.HEADING.heading_range, VocabularyStateType.HEADING.heading_step
WIDTH_START, WIDTH_END, WIDTH_RANGE, WIDTH_STEP = VocabularyStateType.WIDTH.start, VocabularyStateType.WIDTH.end, VocabularyStateType.WIDTH.width_range, VocabularyStateType.WIDTH.width_step
LENGTH_START, LENGTH_END, LENGTH_RANGE, LENGTH_STEP = VocabularyStateType.LENGTH.start, VocabularyStateType.LENGTH.end, VocabularyStateType.LENGTH.length_range, VocabularyStateType.LENGTH.length_step
CLASS_TYPE_START, CLASS_TYPE_END = VocabularyStateType.CLASS_TYPE.start, VocabularyStateType.CLASS_TYPE.end
BOS_TOKEN_VALUE, NEWBORN_BOS_TOKEN_VALUE, PAD_TOKEN_VALUE = VocabularyStateType.BOS_TOKEN.start, VocabularyStateType.NEWBORN_BEGIN_TOKEN.start, VocabularyStateType.PAD_TOKEN.start

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
NpSequence_DIM = NpSequenceArray.dim
STATUS_DIM = NpSequenceArray.status_dim

# Constants for NpTokenizedSequenceArray
NpTokenizedSequence_DIM = NpTokenizedSequenceArray.dim
TOKENIZED_X_DIM = NpTokenizedSequenceArray.tokenized_x_dim
TOKENIZED_Y_DIM = NpTokenizedSequenceArray.tokenized_y_dim
TOKENIZED_HEADING_DIM = NpTokenizedSequenceArray.tokenized_heading_dim
NEXT_TOKENIZED_X_DIM = NpTokenizedSequenceArray.next_tokenized_x_dim
NEXT_TOKENIZED_Y_DIM = NpTokenizedSequenceArray.next_tokenized_y_dim
NEXT_TOKENIZED_HEADING_DIM = NpTokenizedSequenceArray.next_tokenized_heading_dim

BOS_TOKEN = VocabularyStateType.BOS_TOKEN.start
NEWBORN_BOS_TOKEN = VocabularyStateType.NEWBORN_BEGIN_TOKEN.start
PAD_TOKEN = VocabularyStateType.PAD_TOKEN.start

MAX_TRACK_ID = VocabularyStateType.AGENTS.end

# constant for ego width and length
ego_width, ego_length = 2.297, 5.176 # hard code for nuplan


@jit(nopython=True)
def get_bos_array(frame_index):
    """
    Create a beginning of sequence (BOS) array with specified frame index.
    
    Args:
        frame_index (int): The index of the frame for which to create the BOS token.
        
    Returns:
        np.ndarray: A numpy array representing the BOS token with the frame index.
    """
    bos_array = np.zeros((NpSequence_DIM,))
    bos_array[TOKEN_TYPE_IDX] = TokenType.BOS_TOKEN.value
    bos_array[FRAME_INDEX_IDX] = frame_index
    return bos_array

@jit(nopython=True)
def get_newborn_bos_array(frame_index):
    """
    Create a beginning of sequence array for newborn agents with a specified frame index.
    
    Args:
        frame_index (int): The index of the frame for which to create the newborn BOS token.
        
    Returns:
        np.ndarray: A numpy array representing the newborn BOS token with the frame index.
    """
    newborn_bos_array = np.zeros((NpSequence_DIM,))
    newborn_bos_array[TOKEN_TYPE_IDX] = TokenType.NEWBORN_BOS_TOKEN.value
    newborn_bos_array[FRAME_INDEX_IDX] = frame_index
    return newborn_bos_array

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
    x, y, heading, vx, vy, ax, ay = current_ego_array
    ego_array = np.zeros((NpSequence_DIM, ))
    ego_array[TOKEN_TYPE_IDX] = TokenType.EGO_TOKEN.value
    ego_array[FRAME_INDEX_IDX] = frame_index
    ego_array[X_DIM] = x
    ego_array[Y_DIM] = y
    ego_array[HEADING_DIM] = normalize_angle(heading)
    ego_array[VX_DIM] = vx
    ego_array[VY_DIM] = vy
    ego_array[WIDTH_DIM] = ego_width
    ego_array[LENGTH_DIM] = ego_length
    ego_array[CLASS_TYPE_DIM] = ClassType.VEHICLE.value
    ego_array[TRACK_TOKEN_DIM] = -1
    ego_array[TRACK_ID_DIM] = track_id
    return ego_array


@jit(nopython=True)
def get_agent_array(frame_index, current_agent_array, track_id, is_newborn, class_index):
    """
    Generate an array representing an agent's state in a specific frame.

    Args:
        frame_index (int): The index of the current frame.
        current_agent_array (np.ndarray): Array containing the agent's state data.
        track_id (int): The track identifier for the agent.
        is_newborn (bool): Indicator if the agent is newly detected in this frame.
        class_index (int): Class type index of the agent (e.g., vehicle, pedestrian).

    Returns:
        np.ndarray: A numpy array representing the agent's state in the sequence.
    """
    track_token, vx, vy, heading, width, length, x, y = current_agent_array
    agent_array = np.zeros((NpSequence_DIM, ))
    agent_array[TOKEN_TYPE_IDX] = TokenType.NEWBORN_AGENT_TOKEN.value if is_newborn else TokenType.AGENT_TOKEN.value
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
    track_token, vx, vy, heading, width, length, x, y = current_agent_array
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
def process_single_batch(agents_array, ego_array, max_seq_len=10240):
    """
    Process a single batch of agent and ego arrays to create a tokenized sequence.

    Args:
        agents_array (List[np.ndarray]): List of agent arrays for each class type and frame.
        ego_array (np.ndarray): Array containing ego vehicle data across frames.
        max_seq_len (int): Maximum length of the sequence to be processed.

    Returns:
        np.ndarray: Tokenized information array for the batch.
    """
    tokenized_info_array = np.nan * np.ones((max_seq_len, NpSequence_DIM))
    total_num_frames = ego_array.shape[0]
    seq_index = 0
    track_id_mapping = Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )
    for frame_index in range(total_num_frames):
        # Handling beginning of sequence
        tokenized_info_array[seq_index] = get_bos_array(frame_index)
        seq_index = safe_add_one(seq_index, max_seq_len)

        # Processing the ego vehicle
        current_ego_array = ego_array[frame_index]
        tokenized_info_array[seq_index] = get_ego_array(frame_index, current_ego_array, track_id=0)
        seq_index = safe_add_one(seq_index, max_seq_len)

        # Cache for newborn agents to be processed at end of frame
        stashed_newborn_agents_array = []

        # Process each agent for the current frame
        for class_index, agents in enumerate(agents_array):
            current_agents_array = agents[frame_index]
            for current_agent_array in current_agents_array:
                if np.isnan(current_agent_array).any():
                    continue
                track_id, is_newborn, track_id_mapping = get_track_id(class_index, current_agent_array, track_id_mapping)
                if track_id > MAX_TRACK_ID:
                    print("Error: Track ID exceeds the maximum allowable limit.")
                    continue
                # Newborn agents are processed after all existing agents
                if frame_index > 0 and is_newborn:
                    stashed_newborn_agents_array.append(get_agent_array(frame_index, current_agent_array, track_id=track_id, is_newborn=is_newborn, class_index=class_index))
                else:
                    tokenized_info_array[seq_index] = get_agent_array(frame_index, current_agent_array, track_id=track_id, is_newborn=False, class_index=class_index)
                    seq_index = safe_add_one(seq_index, max_seq_len)
        # Handle the beginning of sequence for newborn agents
        tokenized_info_array[seq_index] = get_newborn_bos_array(frame_index)
        seq_index = safe_add_one(seq_index, max_seq_len)

        # Append newborn agents to the sequence
        for stashed_newborn_agent_array in stashed_newborn_agents_array:
            tokenized_info_array[seq_index] = stashed_newborn_agent_array
            seq_index = safe_add_one(seq_index, max_seq_len)
    return tokenized_info_array


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
def tokenize_data(data):
    """
    Convert data points into tokens based on specified ranges and steps.
    
    Args:
        data (np.ndarray): The data array to be tokenized.
    
    Returns:
        np.ndarray: The data array with tokenized values.
    """
    def calculate_token(value, value_range, start, step):
        # Clamping the value within the range and calculating the token
        value = np.maximum(value_range[0], np.minimum(value_range[1], value))
        return start + int(round((value - value_range[0]) / step))
    
    # Tokenizing data only for specific token types
    if int(data[TOKEN_TYPE_IDX]) in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:  
        x, y, heading = data[X_DIM], data[Y_DIM], data[HEADING_DIM]
        x_token = calculate_token(x, X_RANGE, X_START, X_STEP)
        y_token = calculate_token(y, Y_RANGE, Y_START, Y_STEP)
        heading_token = calculate_token(heading, HEADING_RANGE, HEADING_START, HEADING_STEP)
        data[TOKENIZED_X_DIM] = x_token
        data[TOKENIZED_Y_DIM] = y_token
        data[TOKENIZED_HEADING_DIM] = heading_token
    else:
        # Setting tokens to -1 if data type does not match
        data[TOKENIZED_X_DIM] = -1
        data[TOKENIZED_Y_DIM] = -1
        data[TOKENIZED_HEADING_DIM] = -1       
    return data

@jit(nopython=True)
def tokenize_single_batch(processed_single_batch, max_seq_len=4096, valid_range=np.array([-100, 100, -100, 100])):
    """
    Tokenize a batch of data by filtering out entries outside of a valid range and tokenizing the valid entries.
    
    Args:
        processed_single_batch (np.ndarray): Processed batch data to be tokenized.
        max_seq_len (int): Maximum sequence length for the output tokenized data.
        valid_range (np.ndarray): Array specifying the valid [xmin, xmax, ymin, ymax] range.
    
    Returns:
        np.ndarray: The tokenized data array.
    """
    tokenized_data = np.nan * np.ones((max_seq_len, NpTokenizedSequence_DIM))
    seq_idx = 0
    for i in range(len(processed_single_batch)):
        data = processed_single_batch[i]
        if np.isnan(data).any() or not within_valid_range(data, valid_range):
            continue
        tokenized_data[seq_idx, :data.shape[0]] = data
        tokenized_data[seq_idx] = tokenize_data(tokenized_data[seq_idx])
        seq_idx = safe_add_one(seq_idx, max_seq_len)

    # Setting next tokenized values for sequences
    for i in range(len(processed_single_batch)):
        data = tokenized_data[i]
        if np.isnan(data).all():
            continue
        for j in range(i+1, len(processed_single_batch)):
            next_data = tokenized_data[j]
            if next_data[TRACK_ID_DIM] == data[TRACK_ID_DIM] and next_data[FRAME_INDEX_IDX] == data[FRAME_INDEX_IDX] + 1:
                data[NEXT_TOKENIZED_X_DIM] = next_data[TOKENIZED_X_DIM]
                data[NEXT_TOKENIZED_Y_DIM] = next_data[TOKENIZED_Y_DIM]
                data[NEXT_TOKENIZED_HEADING_DIM] = next_data[TOKENIZED_HEADING_DIM]
                break
        tokenized_data[i] = np.nan_to_num(data, -1)
    return tokenized_data

def run_function_multiple_times(num_runs, agents_array, ego_array):
    for _ in range(1):
        process_single_batch(agents_array, ego_array)
    start_time = time.time()
    for _ in range(num_runs):
        tokenized_info_array = process_single_batch(agents_array, ego_array)
    end_time = time.time()
    print("Total time for {} runs: {:.2f} seconds".format(num_runs, end_time - start_time))
    return tokenized_info_array


@jit(nopython=True)
def get_frame_index(tokenized_arrays, block_size=1024, max_frames=1000):
    """
    Get the frame index for the tokenized arrays.

    Args:
        tokenized_arrays (numpy.ndarray): The tokenized arrays.
        block_size (int, optional): The maximum block size. Defaults to 1024.

    Returns:
        list: A list of frame indices.
    """
    batch_size, seq_len, _ = tokenized_arrays.shape
    frame_indices = (max_frames-1) * np.ones((batch_size, block_size,), dtype=np.int64)
    for i in range(tokenized_arrays.shape[0]):
        embed_seq_index = 0
        for j in range(tokenized_arrays.shape[1]):
            current_token = tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                continue
            if embed_seq_index >= block_size:
                break
            token_type = current_token[TOKEN_TYPE_IDX]
            frame_index = current_token[FRAME_INDEX_IDX]
            if token_type in [TokenType.BOS_TOKEN.value, TokenType.NEWBORN_BOS_TOKEN.value, TokenType.PAD_TOKEN.value]:
                frame_indices[i, embed_seq_index] = frame_index
                embed_seq_index = safe_add_one(embed_seq_index, frame_indices.shape[1])
            elif token_type in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                frame_indices[i, embed_seq_index] = frame_index
                embed_seq_index = safe_add_one(embed_seq_index, frame_indices.shape[1])
                if embed_seq_index >= block_size:
                    break
                frame_indices[i, embed_seq_index] = frame_index
                embed_seq_index = safe_add_one(embed_seq_index, frame_indices.shape[1])
            else:
                print('invalid type:', token_type)
    return frame_indices


@jit(nopython=True)
def get_tokenized_inds(tokenized_arrays, block_size=1024):
    """
    Get the tokenized indices and features for control, query, and state embeddings.

    Args:
        tokenized_arrays (numpy.ndarray): The tokenized arrays.
        block_size (int, optional): The maximum block size. Defaults to 1024.

    Returns:
        tuple: A tuple containing the following numpy arrays:
            - ctrl_embedding_inds: The indices for control embeddings.
            - ctrl_embedding_features: The features for control embeddings.
            - query_embedding_inds: The indices for query embeddings.
            - query_embedding_features: The features for query embeddings.
            - state_embedding_inds: The indices for state embeddings.
            - state_embedding_features: The features for state embeddings.
    """
    max_inds = tokenized_arrays.shape[0] * block_size
    ctrl_embedding_inds = np.full((max_inds, 2), -1, dtype=np.int64)
    ctrl_embedding_features = np.full(max_inds, -1, dtype=np.int64)
    query_embedding_inds = np.full((max_inds, 2), -1, dtype=np.int64)
    query_embedding_features = np.full((max_inds, 2), -1, dtype=np.int64)  # Assuming two features (class_index, track_id)
    state_embedding_inds = np.full((max_inds, 2), -1, dtype=np.int64)
    state_embedding_features = np.full((max_inds, 7), -1, dtype=np.float64)  # Assuming 7 features (x, y, heading, w, l, vx, vy)
    tokenized_embedding_features = np.full((max_inds, 3), -1, dtype=np.int64)

    ctrl_index = 0
    query_index = 0
    state_index = 0

    for i in range(tokenized_arrays.shape[0]):
        embed_seq_index = 0
        for j in range(tokenized_arrays.shape[1]):
            current_token = tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                continue
            if embed_seq_index >= block_size:
                break
            token_type = current_token[TOKEN_TYPE_IDX]

            if token_type == TokenType.BOS_TOKEN.value:
                ctrl_embedding_inds[ctrl_index] = (i, embed_seq_index)
                ctrl_embedding_features[ctrl_index] = BOS_TOKEN
                ctrl_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)
            elif token_type == TokenType.NEWBORN_BOS_TOKEN.value:
                ctrl_embedding_inds[ctrl_index] = (i, embed_seq_index)
                ctrl_embedding_features[ctrl_index] = NEWBORN_BOS_TOKEN
                ctrl_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)
            elif token_type == TokenType.PAD_TOKEN.value:
                ctrl_embedding_inds[ctrl_index] = (i, embed_seq_index)
                ctrl_embedding_features[ctrl_index] = PAD_TOKEN
                ctrl_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)
            elif token_type in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                track_id = current_token[TRACK_ID_DIM]
                class_index = current_token[CLASS_TYPE_DIM]
                vx, vy = current_token[VX_DIM], current_token[VY_DIM]
                x, y, heading = current_token[X_DIM], current_token[Y_DIM], current_token[HEADING_DIM]
                w, l = current_token[WIDTH_DIM], current_token[LENGTH_DIM]
                tx, ty, tz = current_token[TOKENIZED_X_DIM], current_token[TOKENIZED_Y_DIM], current_token[TOKENIZED_HEADING_DIM]

                query_embedding_inds[query_index] = (i, embed_seq_index)
                query_embedding_features[query_index] = (class_index + CLASS_TYPE_START, track_id)
                query_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)

                if embed_seq_index >= block_size:
                    break

                state_embedding_inds[state_index] = (i, embed_seq_index)
                state_embedding_features[state_index] = (x, y, heading, w, l, vx, vy)
                tokenized_embedding_features[state_index] = (tx, ty, tz)
                state_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)

    # Trim arrays to the size actually used
    ctrl_embedding_inds = ctrl_embedding_inds[:ctrl_index]
    ctrl_embedding_features = ctrl_embedding_features[:ctrl_index]
    query_embedding_inds = query_embedding_inds[:query_index]
    query_embedding_features = query_embedding_features[:query_index]
    state_embedding_inds = state_embedding_inds[:state_index]
    state_embedding_features = state_embedding_features[:state_index]
    tokenized_embedding_features = tokenized_embedding_features[:state_index]

    return ctrl_embedding_inds, ctrl_embedding_features, query_embedding_inds, query_embedding_features, state_embedding_inds, state_embedding_features, tokenized_embedding_features

@jit(nopython=True)
def get_agent_target_inds(tokenized_arrays, block_size=1024, last_frame_only=False):
    """
    Retrieves the indices of agent target embeddings and state embedding features from tokenized arrays.

    Args:
        tokenized_arrays (numpy.ndarray): The tokenized arrays containing the tokens.
        block_size (int, optional): The maximum number of tokens to consider. Defaults to 1024.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - query_embedding_inds (numpy.ndarray): The indices of agent target embeddings.
            - state_embedding_features (numpy.ndarray): The state embedding features.

    """
    max_inds = tokenized_arrays.shape[0] * block_size
    query_embedding_inds = np.full((max_inds, 2), -1, dtype=np.int64)
    state_embedding_features = np.full((max_inds, 3), -1, dtype=np.float64)  # Assuming features for x, y, and heading

    query_index = 0
    state_index = 0

    for i in range(tokenized_arrays.shape[0]):
        embed_seq_index = 0
        if last_frame_only:
            last_frame_index = find_last_frame_index(tokenized_arrays[i])
        for j in range(tokenized_arrays.shape[1]):
            current_token = tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                continue
            if embed_seq_index >= block_size:
                break
            token_type = current_token[TOKEN_TYPE_IDX]
            if token_type in [TokenType.BOS_TOKEN.value, TokenType.NEWBORN_BOS_TOKEN.value, TokenType.PAD_TOKEN.value]:
                embed_seq_index = safe_add_one(embed_seq_index, block_size)  # Advance the sequence index for control tokens
            elif token_type in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                track_id = current_token[TRACK_ID_DIM]
                class_index = current_token[CLASS_TYPE_DIM]
                token_x, token_y, token_heading = current_token[TOKENIZED_X_DIM], current_token[TOKENIZED_Y_DIM], current_token[TOKENIZED_HEADING_DIM]

                if last_frame_only and current_token[FRAME_INDEX_IDX] != last_frame_index:
                    pass
                else:
                    query_embedding_inds[query_index] = (i, embed_seq_index)
                    query_index += 1

                    state_embedding_features[state_index] = (token_x, token_y, token_heading)
                    state_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)
                if embed_seq_index >= block_size:
                    break
                embed_seq_index = safe_add_one(embed_seq_index, block_size)

            else:
                print('invalid type:', token_type)
    # Trim arrays to the actual size used
    query_embedding_inds = query_embedding_inds[:query_index]
    state_embedding_features = state_embedding_features[:state_index]

    return query_embedding_inds, state_embedding_features

@jit(nopython=True)
def get_ctrl_target_inds(tokenized_arrays, block_size=1024):

    token_type_to_tokenized = {
        0: BOS_TOKEN_VALUE,
        3: NEWBORN_BOS_TOKEN_VALUE,
        5: PAD_TOKEN_VALUE,
    }

    max_inds = tokenized_arrays.shape[0] * block_size
    ctrl_inds = np.full((max_inds, 2), -1, dtype=np.int64)
    target_ctrl_states = np.full(max_inds, -1, dtype=np.int64)
    ctrl_index = 0
    target_ctrl_index = 0

    for i in range(tokenized_arrays.shape[0]):
        embed_seq_index = 0
        for j in range(tokenized_arrays.shape[1]):
            current_token = tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                continue
            if embed_seq_index >= block_size:
                break
            token_type = current_token[TOKEN_TYPE_IDX]    
            if token_type in [TokenType.BOS_TOKEN.value, TokenType.NEWBORN_BOS_TOKEN.value, TokenType.PAD_TOKEN.value]:
                if embed_seq_index > 0:
                    ctrl_inds[ctrl_index] = (i, embed_seq_index - 1)
                    target_ctrl_states[target_ctrl_index] = token_type_to_tokenized[int(token_type)]
                    ctrl_index += 1
                    target_ctrl_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)  # Advance the sequence index for control tokens
            elif token_type in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                if embed_seq_index > 0:
                    ctrl_inds[ctrl_index] = (i, embed_seq_index - 1)
                    target_ctrl_states[target_ctrl_index] = int(current_token[TRACK_ID_DIM])
                    ctrl_index += 1
                    target_ctrl_index += 1
                embed_seq_index = safe_add_one(embed_seq_index, block_size)
                if embed_seq_index >= block_size:
                    break
                embed_seq_index = safe_add_one(embed_seq_index, block_size)
            else:
                print('invalid type:', token_type)
    return ctrl_inds[:ctrl_index], target_ctrl_states[:target_ctrl_index]

@jit(nopython=True)
def extract_history_data(hist_tokenized_arrays, num_conditioned_frames):
    """
    Delete future data from the given hist_tokenized_arrays.

    Args:
        hist_tokenized_arrays (numpy.ndarray): The input array containing tokenized historical data.
        num_conditioned_frames (int): The number of frames to condition on.

    Returns:
        numpy.ndarray: The modified hist_tokenized_arrays with future data deleted.
    """
    for i in range(hist_tokenized_arrays.shape[0]):
        start_frame_index = hist_tokenized_arrays[i, 0, FRAME_INDEX_IDX]
        for j in range(hist_tokenized_arrays.shape[1]):
            current_token = hist_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                continue
            frame_index = current_token[FRAME_INDEX_IDX]
            if (frame_index - start_frame_index) >= num_conditioned_frames:
                hist_tokenized_arrays[i, j, :] = np.nan
    return hist_tokenized_arrays

@jit(nopython=True)
def extract_last_frame_data(hist_tokenized_arrays, last_num_frame=1, skip_nb_bos=True):
    """
    Extracts the last frame data from the given tokenized arrays.

    Args:
        hist_tokenized_arrays (ndarray): The input tokenized arrays.

    Returns:
        ndarray: The last frame arrays with the same shape as the input arrays.

    Notes:
        - Only the data from the last frame is kept.
        - The last frame index is incremented by 1 for each token in the last frame.
    """
    last_frame_arrays = np.nan * np.ones_like(hist_tokenized_arrays)
    for i in range(hist_tokenized_arrays.shape[0]):
        last_frame_index = 0
        for j in range(hist_tokenized_arrays.shape[1]):
            current_token = hist_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            last_frame_index = current_token[FRAME_INDEX_IDX]
        last_frame_index = last_frame_index - last_num_frame + 1
        last_frame_seq_index = 0
        for j in range(hist_tokenized_arrays.shape[1]):
            current_token = hist_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            if current_token[FRAME_INDEX_IDX] >= last_frame_index:

                # skip the newborn bos token, since we do not predict new born agents here
                if skip_nb_bos and current_token[TOKEN_TYPE_IDX] in [TokenType.NEWBORN_BOS_TOKEN.value]:
                    continue
                # update the token type to agent token, since we do not predict new born agents here
                if skip_nb_bos and current_token[TOKEN_TYPE_IDX] in [TokenType.NEWBORN_AGENT_TOKEN.value]:
                    current_token[TOKEN_TYPE_IDX] = TokenType.AGENT_TOKEN.value
                last_frame_arrays[i, last_frame_seq_index] = current_token
                last_frame_seq_index = safe_add_one(last_frame_seq_index, last_frame_arrays.shape[1])
    return last_frame_arrays

@jit(nopython=True)
def extract_first_frame_data(hist_tokenized_arrays, first_num_frame=1):
    """
    Extracts the first num frame data from the given tokenized arrays.

    Args:
        hist_tokenized_arrays (ndarray): The input tokenized arrays.

    Returns:
        ndarray: The first frame arrays with the same shape as the input arrays.

    Notes:
        - Only the data from the last frame is kept.
        - The last frame index is incremented by 1 for each token in the last frame.
    """
    first_frame_arrays = np.nan * np.ones_like(hist_tokenized_arrays)
    for i in range(hist_tokenized_arrays.shape[0]):
        first_frame_seq_index = 0
        first_frame_index = hist_tokenized_arrays[i, 0, FRAME_INDEX_IDX]
        for j in range(hist_tokenized_arrays.shape[1]):
            current_token = hist_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            if current_token[FRAME_INDEX_IDX] < first_frame_index + first_num_frame:
                first_frame_arrays[i, first_frame_seq_index] = current_token
                first_frame_seq_index = safe_add_one(first_frame_seq_index, first_frame_arrays.shape[1])
    return first_frame_arrays


@jit(nopython=True)
def update_last_frame_data(last_tokenized_arrays, pred_agent_tokens):
    """
    update last frame data with predicted agent tokens
    :param last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :param pred_agent_tokens: np.ndarray, shape (num_agents, [x, y, heading] tokens)
    :return updated_last_frame_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    """
    embed_seq_index = 0
    for i in range(len(last_tokenized_arrays)): 
        for j in range(len(last_tokenized_arrays[i])):
            current_token = last_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            if current_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value, TokenType.AGENT_TOKEN.value]:
                last_tokenized_arrays[i, j, TOKENIZED_X_DIM] = pred_agent_tokens[embed_seq_index, 0]
                last_tokenized_arrays[i, j, TOKENIZED_Y_DIM] = pred_agent_tokens[embed_seq_index, 1]
                last_tokenized_arrays[i, j, TOKENIZED_HEADING_DIM] = pred_agent_tokens[embed_seq_index, 2]
                embed_seq_index = safe_add_one(embed_seq_index, pred_agent_tokens.shape[0])
    assert embed_seq_index == len(pred_agent_tokens), f"Number of predicted agent tokens {embed_seq_index} does not match the number of agent tokens {pred_agent_tokens.shape} in the last frame"
    return last_tokenized_arrays

@jit(nopython=True)
def detokenize_data(tokenized_arrays):
    """
    Detokenize the tokenized arrays to get the original data.
    :param tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :return detokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    """
    for i in range(len(tokenized_arrays)):
        for j in range(len(tokenized_arrays[i])):
            current_token = tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            if current_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                x = X_RANGE[0] + (current_token[TOKENIZED_X_DIM] - X_START) * X_STEP
                y = Y_RANGE[0] + (current_token[TOKENIZED_Y_DIM] - Y_START) * Y_STEP
                heading = HEADING_RANGE[0] + (current_token[TOKENIZED_HEADING_DIM] - HEADING_START) * HEADING_STEP
                tokenized_arrays[i, j, X_DIM] = x
                tokenized_arrays[i, j, Y_DIM] = y
                tokenized_arrays[i, j, HEADING_DIM] = heading
    return tokenized_arrays

@jit(nopython=True)
def filter_data(tokenized_arrays, eps=2.0):
    """
    loop over tokenized arrays, remove agents that are out of X_RANGE and Y_RANGE
    """
    filtered_data = np.ones_like(tokenized_arrays) * np.nan
    for i in range(len(tokenized_arrays)):
        filtered_index = 0
        for j in range(len(tokenized_arrays[i])):
            current_token = tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            if current_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                x = current_token[X_DIM]
                y = current_token[Y_DIM]
                if x >= X_RANGE[0] + eps and x <= X_RANGE[1] - eps and y >= Y_RANGE[0] + eps and y <= Y_RANGE[1] - eps:
                    filtered_data[i, filtered_index] = current_token
                    filtered_index += 1
            else:
                filtered_data[i, filtered_index] = current_token
                filtered_index += 1
    return filtered_data


@jit(nopython=True)
def update_history_data(hist_tokenized_arrays, last_tokenized_arrays):
    """
    append last frame data to history data
    :param hist_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :param last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :return updated_hist_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :return updated_last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    """
    for i in range(len(hist_tokenized_arrays)):
        for j in range(len(hist_tokenized_arrays[i])):
            current_token = hist_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
        for k in range(len(last_tokenized_arrays[i])):
            last_token = last_tokenized_arrays[i, k]
            if np.isnan(last_token).any():
                break
            hist_tokenized_arrays[i, j] = last_token
            j += 1
    return hist_tokenized_arrays

@jit(nopython=True)
def add_one_to_frame_index(last_tokenized_arrays):
    """
    add one to frame index in last frame data
    :param last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :return updated_last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    """
    for i in range(len(last_tokenized_arrays)):
        for j in range(len(last_tokenized_arrays[i])):
            current_token = last_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            current_token[FRAME_INDEX_IDX] += 1
    return last_tokenized_arrays
        
@jit(nopython=True)
def count_num_frame(tokenized_arrays):
    """
    Count the number of frames in the tokenized arrays.
    """
    max_frames = - np.inf
    for i in range(tokenized_arrays.shape[0]):
        for j in range(len(tokenized_arrays[i])):
            current_token = tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
        num_frame = tokenized_arrays[i, j-1, FRAME_INDEX_IDX] - tokenized_arrays[i, 0, FRAME_INDEX_IDX] + 1
        max_frames = max(max_frames, num_frame)
    return int(max_frames)

@jit(nopython=True)
def add_one_to_frame_index(last_tokenized_arrays):
    """
    add one to frame index in last frame data
    :param last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :return updated_last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    """
    for i in range(len(last_tokenized_arrays)):
        for j in range(len(last_tokenized_arrays[i])):
            current_token = last_tokenized_arrays[i, j]
            if np.isnan(current_token).any():
                break
            current_token[FRAME_INDEX_IDX] += 1
    return last_tokenized_arrays

@jit(nopython=True)
def mark_as_generated(last_tokenized_arrays):
    """
    mark the last frame data as generated
    :param last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    :return updated_last_tokenized_arrays: np.ndarray, shape (batch_size, max_seq_len, NpTokenizedSequence_DIM)
    """
    for i in range(len(last_tokenized_arrays)):
        for j in range(len(last_tokenized_arrays[i])):
            if np.isnan(last_tokenized_arrays[i, j]).any():
                break
            last_tokenized_arrays[i, j, STATUS_DIM] = StatusType.GENERATED.value
    return last_tokenized_arrays 

@jit(nopython=True)
def find_agent_index_in_ground_truth_array(tokenized_array, current_token):
    # Localize access to current_token attributes
    current_frame_index = current_token[FRAME_INDEX_IDX]
    current_track_id = current_token[TRACK_ID_DIM]
    
    # Convert list to set for faster membership checking
    valid_token_types = {TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value}

    for i in range(len(tokenized_array)):
        # Early exit if NaN is found in the row
        if np.isnan(tokenized_array[i]).any():
            break
        # Check if the current row matches the criteria
        if tokenized_array[i, FRAME_INDEX_IDX] == current_frame_index and tokenized_array[i, TRACK_ID_DIM] == current_track_id and tokenized_array[i, TOKEN_TYPE_IDX] in valid_token_types:
            return i

    return -1

@jit(nopython=True)
def find_surrounding_agents_in_ground_truth_array(predicted_tokenized_array, current_token):
    surrounding_agents_array = List()
    for token_index in range(len(predicted_tokenized_array)):
        other_token = predicted_tokenized_array[token_index]
        if np.isnan(other_token).any():
            break
        if other_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
            if other_token[FRAME_INDEX_IDX] == current_token[FRAME_INDEX_IDX] and other_token[TRACK_ID_DIM] != current_token[TRACK_ID_DIM]:
                surrounding_agents_array.append(other_token)
    return surrounding_agents_array

@jit(nopython=True)
def extract_ego_trajectory(tokenized_array):
    ego_trajectory = List()
    for i in range(len(tokenized_array)):
        current_token = tokenized_array[i]
        if np.isnan(current_token).any():
            break
        if current_token[TOKEN_TYPE_IDX] == TokenType.EGO_TOKEN.value:
            ego_trajectory.append([current_token[X_DIM], current_token[Y_DIM], current_token[HEADING_DIM], current_token[FRAME_INDEX_IDX]])
    return ego_trajectory

@jit(nopython=True)
def extract_agents_trajectory(tokenized_array, max_frames=50):
    agents_info_dict = Dict.empty(key_type=types.int64, value_type=types.float64[:,:])

    frame_index = tokenized_array[0, FRAME_INDEX_IDX]
    frame_index_inlist = 0
    for i in range(len(tokenized_array)):
        current_token = tokenized_array[i]
        cur_frame_index = current_token[FRAME_INDEX_IDX]
        if cur_frame_index > frame_index:
            frame_index = cur_frame_index
            frame_index_inlist += 1
        if np.isnan(current_token).any():
            break
        if current_token[TOKEN_TYPE_IDX] == TokenType.NEWBORN_AGENT_TOKEN.value or current_token[TOKEN_TYPE_IDX] == TokenType.AGENT_TOKEN.value:
            track_id, cls_type, frame_idx = int(current_token[TRACK_ID_DIM]), int(current_token[CLASS_TYPE_DIM]), int(current_token[FRAME_INDEX_IDX])
            if not hash_idx(track_id, cls_type) in agents_info_dict:
                agents_info_dict[hash_idx(track_id, cls_type)] = np.ones((max_frames, 6)) * np.nan
            agents_info_dict[hash_idx(track_id, cls_type)][frame_index_inlist] = np.array([current_token[X_DIM], current_token[Y_DIM], current_token[HEADING_DIM], current_token[WIDTH_DIM], current_token[LENGTH_DIM], current_token[FRAME_INDEX_IDX]])
    return agents_info_dict

@jit(nopython=True)
def hash_idx(track_id, cls_type):
    return track_id*20 + cls_type

@jit(nopython=True)
def dehash_idx(h):
    return h//20, h%20

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
    for scene_index in range(len(predicted_tokenized_array)):
        scene_ade = 0
        num_agents = 0
        for token_index in range(len(predicted_tokenized_array[scene_index])):
            current_token = predicted_tokenized_array[scene_index, token_index]
            if np.isnan(current_token).any():
                break
            if current_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                
                if ego_only and current_token[TOKEN_TYPE_IDX] != TokenType.EGO_TOKEN.value:
                    continue
                # skip the conditined tokens
                if current_token[STATUS_DIM] == StatusType.CONDITION.value:
                    continue
                gt_index = find_agent_index_in_ground_truth_array(tokenized_array, current_token)
                if gt_index == -1:
                    continue
                groud_truth_token = tokenized_array[gt_index]
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
    num_agents = 0
    collied_agents = 0
    for token_index in range(len(predicted_tokenized_array)):
        current_token = predicted_tokenized_array[token_index]
        if np.isnan(current_token).any():
            break
        if current_token[TOKEN_TYPE_IDX] in [TokenType.EGO_TOKEN.value, TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
            if ego_only and current_token[TOKEN_TYPE_IDX] != TokenType.EGO_TOKEN.value:
                continue
            # skip the conditined tokens
            if current_token[STATUS_DIM] == StatusType.CONDITION.value:
                continue
            surrounding_agents_array = find_surrounding_agents_in_ground_truth_array(predicted_tokenized_array, current_token)
            num_agents += 1
            for other_token in surrounding_agents_array:
                if np.isnan(other_token).any():
                    break
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

@jit(nopython=True)
def random_start_sampling(tokenized_data, randidx_min=0, randidx_max=8):
    """
    Randomly sample a starting frame index for each scene in the tokenized data.
    
    Args:
        tokenized_data (np.ndarray): The tokenized data array.
    
    Returns:
        np.ndarray: The array containing the starting frame index for each scene.
    """
    sampled_tokenized_data = np.ones_like(tokenized_data) * np.nan
    for i in range(len(tokenized_data)):
        seq_index = 0
        start_frame_index = np.random.randint(randidx_min, randidx_max)
        for j in range(len(tokenized_data[i])):
            if tokenized_data[i, j, FRAME_INDEX_IDX] >= start_frame_index:
                sampled_tokenized_data[i, seq_index] = tokenized_data[i, j]
                seq_index += 1
    return sampled_tokenized_data

@jit(nopython=True)
def shuffle_agents(tokenized_data):
    shuffled_tokenized_data = np.ones_like(tokenized_data) * np.nan
    # Specify the type of the list elements, assuming current_token is 1D float64 array
    shuffle_buffer = List.empty_list(np.float64[:])

    for i in range(len(tokenized_data)):
        seq_index = 0
        for j in range(len(tokenized_data[i])):
            current_token = tokenized_data[i, j]
            if np.isnan(current_token).any() and len(shuffle_buffer) == 0:
                break
            if current_token[TOKEN_TYPE_IDX] == TokenType.EGO_TOKEN.value:
                shuffled_tokenized_data[i, seq_index] = current_token
                seq_index += 1
            elif current_token[TOKEN_TYPE_IDX] in [TokenType.AGENT_TOKEN.value, TokenType.NEWBORN_AGENT_TOKEN.value]:
                shuffle_buffer.append(current_token)
            elif current_token[TOKEN_TYPE_IDX] in [TokenType.BOS_TOKEN.value, TokenType.NEWBORN_BOS_TOKEN.value, TokenType.PAD_TOKEN.value] or np.isnan(current_token).any():
                indices = np.arange(len(shuffle_buffer))
                np.random.shuffle(indices)
                for k in indices:
                    shuffled_tokenized_data[i, seq_index] = shuffle_buffer[k]
                    seq_index += 1
                shuffle_buffer = List.empty_list(np.float64[:])
                shuffled_tokenized_data[i, seq_index] = current_token
                seq_index += 1
                if np.isnan(current_token).any():
                    break
            else:
                print('invalid type:', current_token[TOKEN_TYPE_IDX])
    return shuffled_tokenized_data


        