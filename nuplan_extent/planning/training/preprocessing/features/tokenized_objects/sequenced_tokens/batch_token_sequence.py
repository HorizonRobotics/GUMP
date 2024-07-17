from functools import cached_property
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
import numpy as np
from dataclasses import dataclass
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import \
    AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    FeatureDataType, to_tensor)
from nuplan_extent.planning.training.preprocessing.features.tokenized_objects.sequenced_tokens.token_sequence import TokenSequence
@dataclass
class BatchTokenSequence(AbstractModelFeature):
    """
    Dataclass that holds trajectory signals produced from the model or from the dataset for supervision.

    :param data: either a [num_batches, num_states, 3] or [num_states, 3] representing the trajectory
                 where se2_state is [x, y, heading] with units [meters, meters, radians].
    """

    data : FeatureDataType
    
    def __init__(self) -> None:
        self.data = []
    
    def __post_init__(self) -> None:
        pass
    
    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """
        concatenate two batch token sequences
        """
        assert isinstance(other, BatchTokenSequence)
        assert len(self) == len(other)
        combined_batch_token_sequence = BatchTokenSequence()
        for i in range(len(self)):
            combined_batch_token_sequence.add_batch(self.data[i] + other.data[i])
        combined_batch_token_sequence.update_block_size(self.block_size)
        return combined_batch_token_sequence
    
    
    @property
    def num_batches(self):
        return len(self.data)

    @property
    def block_size(self):
        return self.data[0].block_size
    
    def add_batch(self, token_sequence):
        """
        Add a token sequence to the batch
        :param token_sequence: token sequence to be added
        """
        self.data.append(token_sequence)
    
    def merge_batch(self, batch_token_sequence):
        """
        Merge a batch of token sequences to the current batch
        :param batch_token_sequence: batch of token sequences to be merged
        """
        for token_sequence in batch_token_sequence.data:
            self.add_batch(token_sequence)

    def repeat_interleave(self, n_repeats):
        """
        Repeat interleave this batch token sequence, for example when bs=4 and n_repeats=2
        [b1, b2, b3, b4] =>  [b1, b1, b2, b2, b3, b3, b4, b4]
        :param n_repeats: number of times to repeat
        """
        self.data = [tok_seq for tok_seq in self.data for _ in range(n_repeats)]

    def to_device(self, device: torch.device):
        """Implemented. See interface."""
        self.target_device=device
        return self

    def to_feature_tensor(self):
        """Inherited, see superclass."""
        return self

    @classmethod
    def deserialize(cls, data: Dict[str, Any]):
        """Implemented. See interface."""
        cls.data = data["data"]
        return cls

    def unpack(self):
        """Implemented. See interface."""
        return self
    
    @classmethod
    def collate(cls, batch: List[AbstractModelFeature]) :
        """
        Batch features together. This method processes each feature in the batch using
        the 'add_batch' method and returns a new instance of the class representing
        the collated features.

        :param batch: A list of features to be batched.
        :return: An instance of the class representing the batched features.
        """
        # Initialize a new instance for the collated data
        collated_instance = cls()  # Assuming a default constructor is available

        # Process each sample in the batch
        for sample in batch:
            collated_instance.merge_batch(sample)

        return collated_instance
    
    def drop_frames(self, frame_dropout_rate):
        """
        Drop frames from the token sequence based on the specified dropout rate.
        :param frame_dropout_rate: The rate at which frames should be dropped from the sequence.
        """
        for token_sequence in self.data:
            token_sequence.drop_frames(frame_dropout_rate)
    
    def drop_ego(self, ego_dropout_rate=0.5):
        """
        Drop the ego agent token from the token sequence
        """
        for token_sequence in self.data:
            token_sequence.drop_ego(ego_dropout_rate)
            
    def resample(self):
        """
        Resample the token sequence based on the specified block size.
        :param block_size: The size of the block to be resampled.
        """
        resampled_token_sequences = BatchTokenSequence()
        for token_sequence in self.data:
            resampled_token_sequences.add_batch(token_sequence.resample())
        resampled_token_sequences.update_block_size(self.block_size)
        return resampled_token_sequences
    
    def sample_frame_segment(self, start_local_index=0, end_local_index=0, max_input_token_size=999999):
        """
        sampling the token sequence, starting from start_frame_index to end_frame_index, [start_frame_index, end_frame_index]
        if frame_index is None, cut off till the current frame index (last included)
        """
        resampled_token_sequences = BatchTokenSequence()
        for token_sequence in self.data:
            while start_local_index <= end_local_index:
                sampled_token_sequence = token_sequence.sample_frame_segment(start_local_index, end_local_index)
                sampled_token_sequence.assign_position()
                if sampled_token_sequence.total_num_tokens <= max_input_token_size:
                    break
                start_local_index += 1
            resampled_token_sequences.add_batch(sampled_token_sequence)
        resampled_token_sequences.update_block_size(self.block_size)
        return resampled_token_sequences
    
    def get_ego_agent_token(self):
        """
        Get the ego agent token from the token sequence
        """
        ego_agent_tokens = []
        for token_sequence in self.data:
            ego_agent_tokens.extend(token_sequence.get_ego_agent_token())
        return ego_agent_tokens
    
    def get_prev_ego_tokens(self):
        """
        Get the previous ego token from the token sequence
        """
        prev_ego_tokens = []
        for token_sequence in self.data:
            prev_ego_tokens.append(token_sequence.get_prev_ego_token())
        return prev_ego_tokens
    
    def get_agent_token_by_raw_id(self, raw_ids):
        """
        Get the agent token from the token sequence
        """
        agent_tokens = []
        for token_sequence, raw_id in zip(self.data, raw_ids):
            agent_tokens.extend(token_sequence.get_agent_token_by_raw_id(raw_id))
        return agent_tokens
    
    def update_block_size(self, block_size):
        """
        Update the block size of the token sequence
        :param block_size: The size of the block to be resampled.
        """
        for token_sequence in self.data:
            token_sequence.block_size = block_size
    
    def update_device(self, device):
        self.device = device
    
    def update_dtype(self, dtype):
        self.dtype = dtype

    def update_ego_state(self, ego_state):
        for i, token_sequence in enumerate(self.data):
            token_sequence.update_ego_state(ego_state[i])

    def sort(self):
        """
        Sort the token sequence
        """
        for token_sequence in self.data:
            token_sequence.sort()

    def assign_position(self):
        """
        Tokenize the token sequence
        """
        for token_sequence in self.data:
            token_sequence.assign_position()
            
    def get_max_token_size(self):
        token_sizes = []
        for token_sequence in self.data:
            token_sizes.append(token_sequence.total_num_tokens)
        return max(token_sizes)
            
    def get_agent_state(self):
        """
        Get the agent state from the token sequence
        """
        agent_state = []
        for token_sequence in self.data:
            agent_state.extend(token_sequence.get_agent_state())
        if len(agent_state) == 0:
            return None, None, None, None, None, None
        agent_state = torch.from_numpy(np.stack(agent_state)).to(self.device).to(self.dtype)

        return agent_state.T[..., None]
    
    def set_training_state(self, training):
        """
        Set the training state of the token sequence
        """
        for token_sequence in self.data:
            token_sequence.training = training
    
    def get_tokenized_agent_state(self, mask_invalid=False, return_ego_only=False):
        """
        Get the agent state from the token sequence
        """
        agent_state = []
        for token_sequence in self.data:
            agent_state.extend(token_sequence.get_tokenized_agent_state(return_ego_only=return_ego_only))
        agent_state = np.array(agent_state).astype(np.int32)
        
        if mask_invalid:
            agent_state = np.abs(agent_state)
        agent_state = torch.from_numpy(agent_state).to(self.device).long()
        return agent_state.T[..., None]

    def get_agent_idxes(self):
        """
        Get the agent state from the token sequence
        """
        agent_idxes = []
        for batch_idx, token_sequence in enumerate(self.data):
            agent_idxes.extend(token_sequence.get_agent_idxes(batch_idx))
        agent_idxes = torch.from_numpy(np.stack(agent_idxes)).to(self.device).long()
        return agent_idxes.T
    
    def get_agent_control_idxes(self, return_last_frame=False, return_last_agent=False, return_ego=False):
        """
        Get the agent state from the token sequence
        """
        agent_idxes = []
        for batch_idx, token_sequence in enumerate(self.data):
            agent_idxes.extend(token_sequence.get_agent_control_idxes(batch_idx, return_last_frame=return_last_frame, return_last_agent=return_last_agent, return_ego=return_ego))
        agent_idxes = torch.from_numpy(np.stack(agent_idxes)).to(self.device).long()
        return agent_idxes.T

    def get_normal_decoding_idxes(self):
        """
        Get the agent state from the token sequence
        """
        agent_idxes = []
        for batch_idx, token_sequence in enumerate(self.data):
            agent_idxes.extend(token_sequence.get_normal_decoding_idxes(batch_idx))
        agent_idxes = torch.from_numpy(np.stack(agent_idxes)).to(self.device).long()
        return agent_idxes.T
    
    def get_normal_decoding_tokens(self):
        """
        Get the agent state from the token sequence
        """
        normal_decoding_tokens = []
        for token_sequence in self.data:
            normal_decoding_tokens.extend(token_sequence.get_normal_decoding_tokens())
        return normal_decoding_tokens

    def get_control_state(self):
        """
        Get the control state from the token sequence
        """
        control_state = []
        for token_sequence in self.data:
            control_state.extend(token_sequence.get_control_state())
        control_state = torch.from_numpy(np.stack(control_state)).to(self.device).long()
        return control_state

    def get_control_idxes(self):
        """
        Get the control state from the token sequence using multiprocessing
        """
        control_idxes = []
        for batch_idx, token_sequence in enumerate(self.data):
            control_idxes.extend(token_sequence.get_control_idxes(batch_idx))
        control_idxes = torch.from_numpy(np.stack(control_idxes)).long()
        return control_idxes.T

    def get_traffic_light_state(self):
        """
        Get the traffic light state from the token sequence
        """
        traffic_light_state = []
        for token_sequence in self.data:
            traffic_light_state.extend(token_sequence.get_traffic_light_state())
        if len(traffic_light_state) == 0:
            return None, None, None, None, None
        traffic_light_state = torch.from_numpy(np.stack(traffic_light_state)).to(self.device).to(self.dtype)
        return traffic_light_state.T[..., None]

    def get_traffic_light_idxes(self):
        """
        Get the traffic light state from the token sequence
        """
        traffic_light_idxes = []
        for batch_idx, token_sequence in enumerate(self.data):
            traffic_light_idxes.extend(token_sequence.get_traffic_light_idxes(batch_idx))
        traffic_light_idxes = torch.from_numpy(np.stack(traffic_light_idxes)).to(self.device).long()
        return traffic_light_idxes.T
    
    def get_bos_idxes(self, return_last=False):
        """
        Get the bos state from the token sequence
        """
        bos_idxes = []
        for batch_idx, token_sequence in enumerate(self.data):
            bos_idxes.extend(token_sequence.get_bos_idxes(batch_idx, return_last=return_last))
        if len(bos_idxes) == 0:
            return None, None
        bos_idxes = torch.from_numpy(np.stack(bos_idxes)).to(self.device).long()
        return bos_idxes.T


    def get_bos_tokens(self, return_last=False):
        """
        Get the box tokens from the token sequence
        """
        bos_tokens = []
        for token_sequence in self.data:
            bos_tokens.extend(token_sequence.get_bos_tokens(return_last=return_last))
        return bos_tokens
    
    def get_agent_tokens(self, return_last=False, return_all=False, return_ego=False, return_prev=False):
        """
        Get the agent tokens from the token sequence
        """
        agent_tokens = []
        batch_indexes = []
        for batch_index, token_sequence in enumerate(self.data):
            agent_token_seq = token_sequence.get_agent_tokens(return_last=return_last, return_all=return_all, return_ego=return_ego, return_prev=return_prev)
            batch_indexes.extend([batch_index] * len(agent_token_seq))
            agent_tokens.extend(agent_token_seq)
        return agent_tokens, batch_indexes


    def get_num_traffic_light(self, return_last=False):
        """
        Get the number of traffic lights from the token sequence
        """
        if return_last:
            return self.get_last_num_traffic_light()
        num_traffic_light = []
        bos_tokens = self.get_bos_tokens()
        for bos_token in bos_tokens:
            num_traffic_light.append(bos_token.num_traffic_light)
        return np.array(num_traffic_light)
    
    def get_traffic_light_status_list(self, max_num_traffic_light=64):
        """
        Get the number of traffic lights from the token sequence
        """
        traffic_light_status_list = []
        for token_sequence in self.data:
            traffic_light_status_list.extend(token_sequence.get_traffic_light_status_list(max_num_traffic_light))
        return np.array(traffic_light_status_list)
    
    def get_last_num_traffic_light(self):
        """
        Get the number of traffic lights from the token sequence
        """
        num_traffic_light = []
        for batch_index, token_sequence in enumerate(self.data):
            num_traffic_light.append(len(token_sequence.get_traffic_light_info()))
        return np.array(num_traffic_light)
    
    def get_tl_index(self):
        """
        Get the tl index from the token sequence
        """
        tl_index = []
        for token_sequence in self.data:
            tl_index.append(sorted(list(token_sequence.get_tl_index())))
        return tl_index
    
    def aggregate_traffic_light_info(self):
        """
        Aggregate the traffic light info from the token sequence
        """
        for token_sequence in self.data:
            token_sequence.aggregate_traffic_light_info()
            
    def get_traffic_light_info(self):
        """
        Get the traffic light info from the token sequence
        """
        traffic_light_info = []
        for token_sequence in self.data:
            traffic_light_info.append(token_sequence.get_traffic_light_info())
        return traffic_light_info
    
    def mark_as_imagined(self):
        """
        Mark the token sequence as imagined
        """
        for token_sequence in self.data:
            token_sequence.mark_as_imagined()

    def get_imagined_token_seqeunce(self):
        """
        Get the imagined token sequence
        """
        imagined_token_sequences = BatchTokenSequence()
        for token_sequence in self.data:
            imagined_token_sequences.add_batch(token_sequence.get_imagined_token_seqeunce())
        return imagined_token_sequences
    
    def replace_with_imagined_tokens(self, imagined_token_sequences):
        """
        Replace the token sequence with imagined token sequence
        """
        for token_sequence, imagined_token_sequence in zip(self.data, imagined_token_sequences.data):
            token_sequence.replace_with_imagined_tokens(imagined_token_sequence)
            token_sequence.assign_position()

    def get_valid_num_frames_for_training(self):
        """
        Valid the number of frames for training
        """
        valid_num_frames_for_training = []
        for token_sequence in self.data:
            valid_num_frames_for_training.append(token_sequence.get_valid_num_frames_for_training())
        return valid_num_frames_for_training
    
    def cpu(self):
        return self
    
    def numpy(self):
        numpy_array = []
        max_len = 2048
        for token_sequence in self.data:
            numpy_array.append(token_sequence.numpy())
        for i in range(len(numpy_array)):
            if len(numpy_array[i]) < max_len:
                numpy_array[i] = np.concatenate([numpy_array[i], -np.ones((max_len - len(numpy_array[i]), numpy_array[i].shape[1]))], axis=0)
        return np.stack(numpy_array)

    def from_numpy(self, numpy_array):
        self.data = []
        # to cpu if nessary
        if isinstance(numpy_array, torch.Tensor) :
            numpy_array = numpy_array.cpu().numpy()
        for i in range(numpy_array.shape[0]):
            token_sequence = TokenSequence()
            token_sequence.from_numpy(numpy_array[i])
            self.add_batch(token_sequence)
        return self
    
    def detach(self):
        return self
    

