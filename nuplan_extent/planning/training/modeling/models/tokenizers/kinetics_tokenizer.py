from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import copy
import torch.nn as nn
import numpy as np
import pickle
import time
import nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils as kutils

class KineticsTokenizer(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                max_seq_len: int = 416,
                random_start: bool = False,
                shuffle: bool = False):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._random_start = random_start
        self._shuffle = shuffle

    def forward(self, input: Dict) -> Dict:
        """
        Dummy forward
        """
        batch_size = len(input.data)
        tokenized_data = []
        processed_data = []
        tracks_to_predict_raw_ids_batch = []
        for batch_index in range(batch_size):
            agents_array = [v for k,v in input.data[batch_index]['agents'].items()]
            ego_array = input.data[batch_index]['ego']
            tracks_to_predict_raw_ids = input.data[batch_index]['tracks_to_predict_raw_ids']
            tracks_to_predict_raw_ids_batch.append(tracks_to_predict_raw_ids)

            # hard code for WOD data
            if len(ego_array.shape) == 3:
                ego_array = ego_array[0]
            processed_single_batch = kutils.process_single_batch(
                agents_array, 
                ego_array, 
                max_seq_len=self._max_seq_len)
            processed_data.append(processed_single_batch)
        processed_data = np.stack(processed_data, axis=0)
        tokenized_data = kutils.tokenize_batched_data(processed_data, tracks_to_predict_raw_ids_batch, max_seq_len=self._max_seq_len)
        return tokenized_data

    def forward_train(self, input: Dict) -> Dict:
        return self.forward(input)

    def forward_inference(self, input: Dict) -> Dict:
        """
        Dummy forward
        """
        return self.forward(input)
