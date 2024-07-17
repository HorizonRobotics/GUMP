import numpy as np
import numpy.typing as npt
import torch
import logging
from typing import Any, Dict, List, Tuple, Callable, Union
from copy import deepcopy
from easydict import EasyDict as edict

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ConstrainedNonlinearSmoother,
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_augmented_ego_raster,
    rotate_tilt_angle,
)
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import agent_bbox_to_corners

from numba import jit

from nuplan_extent.planning.training.preprocessing.features.tensor import Tensor
from nuplan_extent.planning.training.preprocessing.features.attribute_table import AttributeTable
from nuplan_extent.planning.training.preprocessing.features.token import Token
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import normalize_angle
from nuplan_extent.planning.training.modeling.models.utils import encoding_traffic_light
from nuplan_extent.planning.training.preprocessing.feature_builders.tokenize_utils import (
    get_future_trajectory,
    find_disappeared_newborn_survived,
)

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.token_sequence import TokenSequence
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.batch_token_sequence import BatchTokenSequence
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.traffic_light_token import TrafficLightToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.object_token.agent_token import AgentToken

from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.traffic_light_end_token import TrafficLightEndToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.bos_token import BOSToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.newborn_begin_token import NewBornBeginToken
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.single_token.control_token.pad_token import PADToken

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
Pose = Tuple[float, float, float]  # (x, y, yaw)


class VectorTokenizer:
    """
    This class copies the target trajectory to the feature dictionary.
    Sometimes the target trajectory is used as a input feature during training for the model.
    such as multibin model, we use target trajectory to generate high level command.

    """

    def __init__(self, num_past_steps, dataset, target_width=None, target_height=None, augment_cfg=None, map_range=[-104, -104, 104, 104]):
        self._num_past_steps = num_past_steps
        self.dataset = dataset
        self.target_width = target_width
        self.target_height = target_height
        self.map_range = map_range
        self.augment_cfg = augment_cfg
        self.lora_fintuning = False

    def tokenize(
        self,
        features: FeaturesType,
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        # feature dim : track_token, vx, vy, heading, width, length, x, y
        generic_agents = features['generic_agents']
        selected_classes = ["VEHICLE", "PEDESTRIAN", "BICYCLE"]
        batch_size = len(generic_agents.ego)
        batched_sequence_tokens = BatchTokenSequence()
        is_simulation = 'is_simulation' in features and features['is_simulation']

        if is_simulation:
            traffic_lights = None
            device = features['raster'].data.device
            dtype = features['raster'].data.dtype
            batch_size, channel, _, _ = features['raster'].data.shape
            raster_map = features['raster'].data.cpu().numpy() if 'raster' in features else None
            raster_map = np.transpose(raster_map, (0, 2, 3, 1))
            channel = 12
            features['raster'].data = torch.zeros((batch_size, channel, self.target_height, self.target_width), dtype=dtype, device=device)
        else:
            traffic_lights = features[
                'traffic_lights'].data if 'traffic_lights' in features else None
            raster_map = features['raster'].data[None] if 'raster' in features else None
            if self.lora_fintuning:
                for i in range(batch_size):
                    features['generic_agents'].ego[i] = features['generic_agents'].ego[i]
                    for class_name in selected_classes:
                        features['generic_agents'].agents[class_name][i] = features['generic_agents'].agents[class_name][i]
            device = None
            dtype = None

        # iterate over batch
        for batch_index in range(batch_size):
            ego_array = generic_agents.ego[batch_index]
            agents_array = {class_name: generic_agents.agents[class_name][batch_index] for class_name in selected_classes}
            
            if is_simulation:
                agents_raw_ids = {class_name: generic_agents.agents[class_name+".raw_ids"][batch_index] for class_name in selected_classes}
            else:
                agents_raw_ids = None

            traffic_light_array = None # TODO (yihan01.hu): hard code for now, do not use traffic light
            raster_map_batch = raster_map[batch_index]
            if is_simulation:
                ego_array, agents_array, traffic_light_array, agents_raw_ids = self.convert_to_numpy(
                    ego_array, agents_array, traffic_light_array, agents_raw_ids)
            sequence_tokens, raster_map_batch = self.process_single_batch(
                ego_array, agents_array, traffic_light_array, agents_raw_ids, raster_map=raster_map_batch)
            batched_sequence_tokens.add_batch(sequence_tokens)
            if 'raster' in features:
                if is_simulation:
                    raster_map_batch = np.transpose(raster_map_batch, (2, 0, 1))
                    raster_map_batch = torch.from_numpy(raster_map_batch).to(device).to(dtype)
                    features['raster'].data[batch_index] = raster_map_batch
                else:
                    features['raster'].data = raster_map_batch
        features['sequence_tokens'] = batched_sequence_tokens
        return features

    def convert_to_numpy(self, ego_array, agents_array, traffic_light_array, agents_raw_ids):
        ego_array = ego_array.cpu().numpy()
        agents_array = {
            k: v.cpu().numpy()
            for k, v in agents_array.items() if isinstance(v, torch.Tensor)
        }
        if agents_raw_ids is not None:
            agents_raw_ids = {
                k: v.cpu().numpy()
                for k, v in agents_raw_ids.items() if isinstance(v, torch.Tensor)
            }
        else:
            agents_raw_ids = None
        return ego_array, agents_array, traffic_light_array, agents_raw_ids
    
    def get_tranform_matrix(self):
        if self.augment_cfg is None:
            tranform_matrix = np.eye(3)
        else:
            if self.augment_cfg.get('augment_prob', 0) < np.random.uniform(0, 1):
                tranform_matrix = np.eye(3)
            else:
                random_rotation = self.augment_cfg.get('random_rotation', [0, 0])
                random_translation_x = self.augment_cfg.get('random_translation_x', [0, 0])
                random_translation_y = self.augment_cfg.get('random_translation_y', [0, 0])
                rot_angle = np.random.uniform(random_rotation[0], random_rotation[1])
                translation_x = np.random.uniform(random_translation_x[0], random_translation_x[1])
                translation_y = np.random.uniform(random_translation_y[0], random_translation_y[1])
                tranform_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle), translation_x],
                                            [np.sin(rot_angle), np.cos(rot_angle), translation_y],
                                            [0, 0, 1]])
        return tranform_matrix
    
    def tranform_array(self, ego_array, agents_array, tranform_matrix):
        """
        Transform the array based on the tranform_matrix
        ego_array: [time, feature_dim] x, y, heading, vx, vy, ax, ay, width, length, z, type_idx
        agents_array: Dict of (class_names : nparray [time, num_agent, feature_dim]), track_token, vx, vy, heading, width, length, x, y, z
        transofrm_matrix: [3, 3]
        """
        speed_matrix = np.eye(3)
        speed_matrix[:2, :2] = tranform_matrix[:2, :2]
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        ego_xy_array = np.concatenate([ego_array[:, :2], np.ones((ego_array.shape[0], 1))], axis=1)
        ego_xy_array = np.matmul(ego_xy_array, tranform_matrix.T)
        ego_array[:, :2] = ego_xy_array[:, :2]

        ego_vxy_array = np.concatenate([ego_array[:, 3:5], np.zeros((ego_array.shape[0], 1))], axis=1)
        ego_vxy_array = np.matmul(ego_vxy_array, speed_matrix.T)
        ego_array[:, 3:5] = ego_vxy_array[:, :2]

        ego_heading_array = ego_array[:, 2] + np.arctan2(tranform_matrix[1, 0], tranform_matrix[0, 0])
        ego_array[:, 2] = normalize_angle(ego_heading_array)

        for class_name in agents_array.keys():
            if agents_array[class_name].shape[1] == 0:
                continue
            agents_xy_array = np.concatenate([agents_array[class_name][:, :, 6:8], np.ones((agents_array[class_name].shape[0], agents_array[class_name].shape[1], 1))], axis=2)
            agents_xy_array = np.matmul(agents_xy_array, tranform_matrix.T)
            agents_array[class_name][:, :, 6:8] = agents_xy_array[:, :, :2]

            agents_vxy_array = np.concatenate([agents_array[class_name][:, :, 1:3], np.zeros((agents_array[class_name].shape[0], agents_array[class_name].shape[1], 1))], axis=2)
            agents_vxy_array = np.matmul(agents_vxy_array, speed_matrix.T)
            agents_array[class_name][:, :, 1:3] = agents_vxy_array[:, :, :2]

            agents_heading_array = agents_array[class_name][:, :, 3] + np.arctan2(tranform_matrix[1, 0], tranform_matrix[0, 0])
            agents_array[class_name][:, :, 3] = normalize_angle(agents_heading_array)
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        return ego_array, agents_array
    
    def rescale_raster_if_needed(self, raster, target_width, target_height, dataset):
        """
        raster: [height, width, channel], fp32 numpy array
        tranform_matrix: [3, 3]
        """
        if dataset == 'waymo':
            raster = raster[..., :-1]
            assert raster.shape[-1] == 10
        if target_width is None or raster is None:
            return raster
        # Current dimensions of the raster
        height, width, channels = raster.shape

        # Calculate padding sizes
        pad_width_left = (target_width - width) // 2
        pad_width_right = target_width - width - pad_width_left
        pad_height_top = (target_height - height) // 2
        pad_height_bottom = target_height - height - pad_height_top

        # Create a new array with target size filled with zeros
        padded_raster = np.zeros((target_height, target_width, channels), dtype=raster.dtype)

        # Copy the original raster into the center of the new array
        padded_raster[pad_height_top:height+pad_height_top, pad_width_left:width+pad_width_left, :] = raster

        return padded_raster           
    
    def transform_raster_if_needed(self, raster, tranform_matrix):
        """
        raster: [height, width, channel], fp32 numpy array
        tranform_matrix: [3, 3]
        """
        if raster is None:
            return None
        # if tranform_matrix is eye(3)
        if np.allclose(tranform_matrix, np.eye(3)):
            return raster
        
        meshgrid = self.augment_cfg.get('meshgrid', [0.5, 0.5])
        raster = raster.astype(np.float32)
        dx, dy = tranform_matrix[0, 2]/meshgrid[0], tranform_matrix[1, 2]/meshgrid[1]

        # rotate and translate the raster with bilinear interpolation
        raster = rotate(raster, np.rad2deg(np.arctan2(tranform_matrix[1, 0], tranform_matrix[0, 0])), reshape=False, order=0)
        raster = shift(raster, [dx, dy, 0], order=0)
        return raster.astype(np.float32)
    
    def pad_coordinate_if_needed(self, raster_map):
        target_width = self.target_width
        target_height = self.target_height
        map_range = self.map_range
        if target_width is None or raster_map is None:
            return raster_map
        # create coordinate map, based on the map_range and target_width, target_height
        # Determine the spacing for x and y based on image size
        x_space = (map_range[2] - map_range[0]) / target_width
        y_space = (map_range[3] - map_range[1]) / target_height
        # Create a grid for x and y
        y_map, x_map = np.meshgrid(np.linspace(map_range[1] + y_space / 2, map_range[3] - y_space / 2, target_height),
                                    np.linspace(map_range[0] + x_space / 2, map_range[2] - x_space / 2, target_width))
        # Move the maps to the appropriate device and adjust dimensions
        raster_map = np.concatenate([raster_map, x_map[..., None], y_map[..., None]], axis=-1)

        return raster_map
    
    def nan_to_zero_if_needed(self, raster):
        if raster is None:
            return None
        if isinstance(raster, torch.Tensor):
            return torch.nan_to_num(raster, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            return np.nan_to_num(raster, nan=0.0, posinf=0.0, neginf=0.0)

    def process_single_batch(self, ego_array, agents_array,
                             traffic_light_array, agents_raw_ids, raster_map=None):
        """
        Processes a single batch of features and converts them into a structured token sequence.

        This method takes arrays of ego, agents, and traffic light features, and transforms them
        into a sequence of tokens suitable for use in a tokenized transformer model. The tokens
        represent the state and dynamics of the ego vehicle, various agent classes, and traffic lights
        at different timesteps.

        Args:
            ego_array: A numpy array or a similar structure, representing the ego vehicle's features.
                    Shape is [time (29 timesteps), feature_dim (7 dimensions)].
            agents_array: A dictionary where each key is a class_name (e.g., 'car', 'pedestrian') and
                        the value is an array representing the features of agents of that class.
                        Each array is shaped [time (29 timesteps), max_num_agents, feature_dim (8 dimensions)].
            traffic_light_array: An array representing the features of traffic lights.
                               List of List[Dict], each dict is a traffic light.
                [[Dict] x N] x num_steps, where N is the number of traffic lights in the map,
                    num_steps is the number of timesteps in the observation.

                remapped_tl_vector = {
                    'lane_id': the lane id of the traffic light in the original mapapi
                    'tl_index': reindex the traffic light based on the current observation
                    'lane_coords': array of shape[N, 2], representing the coordinates of the traffic light
                    'traffic_light_status': TrafficLightStatusType
                }
            agents_raw_ids: A dictionary where each key is a class_name (e.g., 'car', 'pedestrian') and
                        the value is a tensor of raw ids of agents of that class.

        Returns:
            TokenSequence: An instance of the class TokenSequence, containing all the generated tokens
                        that represent the input features for a single batch.
        """
        total_num_steps = ego_array.shape[0]

        sequence_tokens = TokenSequence(
            num_frames=total_num_steps,
            current_frame_index=self._num_past_steps)
        
        tranform_matrix = self.get_tranform_matrix()
        ego_array, agents_array = self.tranform_array(ego_array, agents_array, tranform_matrix)
        dataset = 'nuplan' if ego_array.shape[-1] == 7 else 'waymo'
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        raster_map = self.rescale_raster_if_needed(raster_map, self.target_width, self.target_height, dataset)
        raster_map = self.transform_raster_if_needed(raster_map, tranform_matrix)
        raster_map = self.pad_coordinate_if_needed(raster_map)
        raster_map = self.nan_to_zero_if_needed(raster_map)
        
        scenario_agent_set = set()
        # iterate over time
        for t in range(total_num_steps):
            traffic_light_token_sequence = TokenSequence()
            if traffic_light_array is not None:
                for tl_ind in range(len(traffic_light_array[t])):

                    traffic_light_token = TrafficLightToken(
                        **traffic_light_array[t][tl_ind],
                        frame_index=t,
                    )
                    traffic_light_token_sequence.add_token(traffic_light_token)

            agent_token_sequence = TokenSequence()
            
            if not np.isnan(ego_array[t]).any():
                ego_token = AgentToken.from_ego_array(ego_array,
                                                      frame_index=t,
                                                      dataset=dataset,
                                                      raw_id=None)
                agent_token_sequence.add_token(ego_token)
            for cls_idx, class_name in enumerate(agents_array.keys()):
                # [time, num_agent, feature_dim]
                # Get the values
                num_agent = agents_array[class_name].shape[1]
                for agent_idx in range(num_agent):
                    current_agent_array = agents_array[class_name][t,
                                                                   agent_idx]
                    if np.isnan(current_agent_array).any():
                        continue
                    if not AgentToken.within_range(current_agent_array, cls_idx):
                        continue
                    current_agent_index = int(
                        agents_array[class_name][t, agent_idx, 0])
                    current_agents_array = []
                    for tt in range(total_num_steps):
                        agent_ids = agents_array[class_name][tt, :, 0].astype(
                            np.int64)
                        if current_agent_index in agent_ids:
                            current_agents_array.append(
                                agents_array[class_name][
                                    tt, agent_ids == current_agent_index][0])
                        else:
                            current_agents_array.append(
                                np.array([np.nan] *
                                         agents_array[class_name].shape[-1]))
                    current_agents_array = np.stack(current_agents_array,
                                                    axis=0)
                    raw_id = agents_raw_ids[class_name][agent_idx] if agents_raw_ids is not None else None
                    agent_token = AgentToken.from_agent_array(
                        current_agents_array, frame_index=t, type_idx=cls_idx, dataset=dataset, raw_id=raw_id)
                    agent_token_sequence.add_token(agent_token)
                    scenario_agent_set.add(agent_token)
            # construct frame token
            sequence_tokens.add_token(
                BOSToken(frame_index=t,
                         num_traffic_light=len(traffic_light_token_sequence),
                         num_agents=len(agent_token_sequence)))
            sequence_tokens.add_tokens(traffic_light_token_sequence)
            sequence_tokens.add_token(TrafficLightEndToken(frame_index=t))
            sequence_tokens.add_tokens(agent_token_sequence)
        sequence_tokens.reindexing_agent_tokens(scenario_agent_set)
        # sequence_tokens.add_token(PADToken(frame_index=total_num_steps))

        sequence_tokens.reorganize_newborn_tokens()
        sequence_tokens.sort()
        return sequence_tokens, raster_map
