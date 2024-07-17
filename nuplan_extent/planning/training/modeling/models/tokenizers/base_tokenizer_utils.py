import numpy as np
from enum import Enum
from numba.typed import Dict, List
from numba.core import types
from numba import jit


from enum import Enum
from numba.typed import Dict
from numba.core import types
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.state_type_v1_1 import VocabularyStateType, PositionalStateType

class TokenType(Enum):
    """
    Enum representing different types of tokens used in sequence modeling
    of vehicle and pedestrian tracking data.

    Attributes:
        DUMMY_TOKEN: Placeholder token for initialization or error handling.
        BOS_TOKEN: Beginning of sequence token.
        EGO_TOKEN: Represents the ego vehicle in the sequence.
        AGENT_TOKEN: Represents other agents (vehicles, pedestrians) in the sequence.
        NEWBORN_BOS_TOKEN: Special token for the beginning of a sequence for newborn agents.
        NEWBORN_AGENT_TOKEN: Represents newly detected agents in the sequence.
        PAD_TOKEN: Padding token used to fill sequence length.
    """
    DUMMY_TOKEN = -1
    BOS_TOKEN = 0
    EGO_TOKEN = 1
    AGENT_TOKEN = 2
    NEWBORN_BOS_TOKEN = 3
    NEWBORN_AGENT_TOKEN = 4
    PAD_TOKEN = 5

class StatusType(Enum):
    """
    Enum representing different status types of agents in traffic scenes.
    
    Attributes:
        CONDITION: Represents conditioned agents in the scene, served as inputs.
        GENERATED: Represents agents generated by the model.
    """
    CONDITION = 0
    GENERATED = 1

class ClassType(Enum):
    """
    Enum representing different class types of agents in traffic scenes.

    Attributes:
        VEHICLE: Represents a vehicle.
        PEDESTRIAN: Represents a pedestrian.
        CYCLIST: Represents a cyclist.
    """
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    STATIC = 3


class NpSequenceArray(np.ndarray):
    """
    A numpy array subclass for handling sequences of tracking data efficiently.

    The array is structured with specific dimensions representing different data fields.

    Attributes:
        dim (int): Number of dimensions in the array.
        token_type (int): Index for token type in the sequence.
        frame_index (int): Index for the frame identifier.
        x_dim (int): Index for the X coordinate.
        y_dim (int): Index for the Y coordinate.
        heading_dim (int): Index for the heading.
        vx_dim (int): Index for the velocity in X.
        vy_dim (int): Index for the velocity in Y.
        track_id_dim (int): Index for the track identifier, unique for objects.
        track_token_dim (int): Index for the track token, unique for each class.
        class_type_dim (int): Index for the class type of the agent.
        status_dim (int): Index for the status of the agent (e.g., condition, generated).
    """
    dim = 13
    token_type_dim = 0 
    frame_index_dim = 1
    x_dim = 2
    y_dim = 3
    heading_dim = 4
    vx_dim = 5
    vy_dim = 6
    width_dim = 7
    length_dim = 8
    track_id_dim = 9 
    track_token_dim = 10
    class_type_dim = 11
    status_dim = 12

    def __new__(cls, input_array):
        """Create a new instance of the array."""
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def token_type(self):
        """Array slice of token types."""
        return self[..., self.token_type_dim]

    @property
    def frame_index(self):
        """Array slice of frame indices."""
        return self[..., self.frame_index_dim]

    @property
    def x(self):
        """Array slice of X coordinates."""
        return self[..., self.x_dim]

    @property
    def y(self):
        """Array slice of Y coordinates."""
        return self[..., self.y_dim]

    @property
    def heading(self):
        """Array slice of headings."""
        return self[..., self.heading_dim]

    @property
    def vx(self):
        """Array slice of velocities in X."""
        return self[..., self.vx_dim]

    @property
    def vy(self):
        """Array slice of velocities in Y."""
        return self[..., self.vy_dim]

    @property
    def width(self):
        """Array slice of headings."""
        return self[..., self.width_dim]
        
    @property
    def length(self):
        """Array slice of headings."""
        return self[..., self.length_dim]
        
    @property
    def track_id(self):
        """Array slice of track IDs."""
        return self[..., self.track_id_dim]

    @property
    def track_token(self):
        """Array slice of track tokens."""
        return self[..., self.track_token_dim]

    @property
    def class_type(self):
        """Array slice of class types."""
        return self[..., self.class_type_dim]

    @property
    def is_ego(self):
        """Boolean array slice for ego vehicles."""
        return self[..., self.token_type_dim] == TokenType.EGO_TOKEN.value

class NpTokenizedSequenceArray(NpSequenceArray):
    dim = 19
    tokenized_x_dim = 13
    tokenized_y_dim = 14
    tokenized_heading_dim = 15
    next_tokenized_x_dim = 16
    next_tokenized_y_dim = 17
    next_tokenized_heading_dim = 18

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

BOS_TOKEN = VocabularyStateType.BOS_TOKEN.start
NEWBORN_BOS_TOKEN = VocabularyStateType.NEWBORN_BEGIN_TOKEN.start
PAD_TOKEN = VocabularyStateType.PAD_TOKEN.start

MAX_TRACK_ID = VocabularyStateType.AGENTS.end

@jit(nopython=True)
def normalize_angle(heading):
    """
    Normalize the angle to be within the range [-pi, pi] using the arctan2 function.

    Args:
        heading (float or np.ndarray): The heading angle in radians.

    Returns:
        float or np.ndarray: Normalized angle between [-pi, pi].
    """
    return np.arctan2(np.sin(heading), np.cos(heading))

@jit(nopython=True)
def safe_add_one(index, max_index):
    """
    Safely increments the index by one if it does not exceed the maximum allowed index.
    
    Args:
        index (int): The current index to be incremented.
        max_index (int): The maximum allowable index.

    Returns:
        int: The incremented index if within bounds, otherwise returns the original index.
    """
    if index < max_index:
        return index + 1
    else:
        print("Error: Index exceeds the maximum allowable limit.")
        raise ValueError("Index exceeds the maximum allowable limit.")
        return index

@jit(nopython=True)
def convert_to_local(cur_state, next_state):
    """
    Convert next_state to the coordinate system where cur_state is the origin and aligned with cur_state's orientation.
    
    Parameters:
        cur_state (np.array): The current state coordinates and orientation (x, y, theta, w, l).
        next_state (np.array): The next state coordinates and orientation (x, y, theta, w, l).
    
    Returns:
        np.array: Transformed next_state relative to cur_state.
    """
    x_c, y_c, theta_c, _, _ = cur_state
    x_n, y_n, theta_n, w, l = next_state
    
    # Calculate relative position
    dx = x_n - x_c
    dy = y_n - y_c
    
    # Apply rotation matrix to align with cur_state's orientation
    cos_theta = np.cos(-theta_c)
    sin_theta = np.sin(-theta_c)
    
    local_x = dx * cos_theta - dy * sin_theta
    local_y = dx * sin_theta + dy * cos_theta
    
    # Compute local orientation
    local_orientation = theta_n - theta_c
    
    # Construct the local state
    return np.array([local_x, local_y, local_orientation, w, l])

@jit(nopython=True)
def convert_to_global(cur_state, local_next_state):
    """
    Convert local_next_state from the coordinate system where cur_state is the origin and aligned with cur_state's orientation
    back to the global coordinate system.
    
    Parameters:
        cur_state (np.array): The current state coordinates and orientation (x, y, theta, w, l).
        local_next_state (np.array): The local next state coordinates and orientation relative to cur_state (x, y, theta, w, l).
    
    Returns:
        np.array: Transformed next_state in global coordinates.
    """
    # Extract position and orientation from cur_state
    x_c, y_c, theta_c, w, l = cur_state
    
    # Extract local position and orientation
    x_l, y_l, theta_l, w_l, l_l = local_next_state
    
    # Apply rotation matrix to convert local coordinates to global coordinates
    rotation_matrix = np.array([[np.cos(theta_c), -np.sin(theta_c)],
                                [np.sin(theta_c),  np.cos(theta_c)]])
    global_coordinates = rotation_matrix @ np.array([x_l, y_l])
    
    # Calculate global position by adding translated coordinates to current position
    global_x = x_c + global_coordinates[0]
    global_y = y_c + global_coordinates[1]
    
    # Compute global orientation
    global_orientation = theta_c + theta_l
    
    # Return new global state
    return np.array([global_x, global_y, global_orientation, w_l, l_l])

@jit(nopython=True, fastmath=True)
def rotate_point(x, y, rad):
    """Rotate a point around the origin by an angle."""
    cos_rad = np.cos(rad)
    sin_rad = np.sin(rad)
    return x * cos_rad - y * sin_rad, x * sin_rad + y * cos_rad

@jit(nopython=True, fastmath=True)
def get_corners(x, y, heading, w, l):
    """Compute the corners of the rectangle based on its state vector."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    half_l = l / 2
    half_w = w / 2
    corners = np.array([
        [half_l * cos_h - half_w * sin_h, half_l * sin_h + half_w * cos_h],
        [-half_l * cos_h - half_w * sin_h, -half_l * sin_h + half_w * cos_h],
        [-half_l * cos_h + half_w * sin_h, -half_l * sin_h - half_w * cos_h],
        [half_l * cos_h + half_w * sin_h, half_l * sin_h - half_w * cos_h]
    ])
    corners += np.array([x, y])
    return corners

@jit(nopython=True, fastmath=True)
def average_corner_distance(sa, sb):
    """Calculate the average distance between corners of two rectangles."""
    corners_a = get_corners(sa[0], sa[1], sa[2], sa[3], sa[4])
    corners_b = get_corners(sb[0], sb[1], sb[2], sb[3], sb[4])
    distances = np.sqrt((corners_a[:, 0] - corners_b[:, 0])**2 + (corners_a[:, 1] - corners_b[:, 1])**2)
    return distances.mean()

@jit(nopython=True)
def project_polygon(axis, corners):
    # Project all corners onto the axis and return the min and max projections
    projections = np.dot(corners, axis)
    return np.min(projections), np.max(projections)

@jit(nopython=True)
def overlap(proj1, proj2):
    # Check if projections on a single axis overlap
    min1, max1 = proj1
    min2, max2 = proj2
    return max1 >= min2 and max2 >= min1

@jit(nopython=True, fastmath=True)
def check_collision(sa, sb):
    """
    Check if two rectangles overlap using the Separating Axis Theorem.

    Args:
        sa (np.array): The state vector of the first rectangle [x, y, heading, width, length].
        sb (np.array): The state vector of the second rectangle [x, y, heading, width, length].

    Returns:
        bool: True if there is an overlap (collision), False otherwise.
    """
    x_a, y_a, heading_a, w_a, l_a = sa
    x_b, y_b, heading_b, w_b, l_b = sb

    corners_a_loop = np.zeros((5, 2))
    corners_b_loop = np.zeros((5, 2))
    corners_a = get_corners(x_a, y_a, heading_a, w_a, l_a)
    corners_b = get_corners(x_b, y_b, heading_b, w_b, l_b)
    
    # Complete the loop for corners
    corners_a_loop[:4] = corners_a
    corners_b_loop[:4] = corners_b
    corners_a_loop[4] = corners_a[0]
    corners_b_loop[4] = corners_b[0]

    # Calculate edges
    edges_a = corners_a_loop[1:] - corners_a_loop[:-1]
    edges_b = corners_b_loop[1:] - corners_b_loop[:-1]

    # Preallocate memory for axes
    num_axes = 4  # As each rectangle has 4 sides and thus 4 potential separating axes
    axes = np.zeros((num_axes * 2, 2))  # There will be 8 axes in total

    # Calculate axes
    for i in range(num_axes):
        axes[i, 0] = -edges_a[i, 1]
        axes[i, 1] = edges_a[i, 0]
        axes[num_axes + i, 0] = -edges_b[i, 1]
        axes[num_axes + i, 1] = edges_b[i, 0]

    # Check for overlap on all axes
    for i in range(num_axes * 2):
        norm_axis = axes[i] / np.linalg.norm(axes[i])
        proj_a = project_polygon(norm_axis, corners_a)
        proj_b = project_polygon(norm_axis, corners_b)
        if not overlap(proj_a, proj_b):
            return False  # Found a separating axis
    return True  # No separating axis found, rectangles must overlap

@jit(nopython=True)
def find_last_frame_index(tokenized_array):
    frame_index = 0
    for i in range(tokenized_array.shape[0]):
        if np.isnan(tokenized_array[i]).any():
            break
        frame_index = tokenized_array[i, FRAME_INDEX_IDX]
    return frame_index

@jit(nopython=True)
def clamp(value, min_value, max_value):
    """
    Manually clamp a value between min_value and max_value.
    
    Args:
        value (float): The value to be clamped.
        min_value (float): The minimum allowable value.
        max_value (float): The maximum allowable value.
    
    Returns:
        float: The clamped value.
    """
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value
        