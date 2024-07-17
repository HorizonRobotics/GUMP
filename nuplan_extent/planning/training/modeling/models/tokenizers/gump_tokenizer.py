from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import copy
import torch.nn as nn
import numpy as np
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils


class GUMPTokenizer(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                max_seq_len: int = 4096,
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
        for batch_index in range(batch_size):
            agents_array = [v for k,v in input.data[batch_index]['agents'].items()]
            ego_array = input.data[batch_index]['ego']
            processed_single_batch = gutils.process_single_batch(
                agents_array, 
                ego_array, 
                max_seq_len=self._max_seq_len)
            tokenized_single_batch = gutils.tokenize_single_batch(
                processed_single_batch,
                max_seq_len=self._max_seq_len)
            tokenized_data.append(tokenized_single_batch)
        tokenized_data = np.stack(tokenized_data, axis=0)
        return tokenized_data

    def forward_train(self, input: Dict) -> Dict:
        tokenized_data = self.forward(input)
        if self._random_start:
            tokenized_data = gutils.random_start_sampling(tokenized_data, randidx_min=0, randidx_max=10)
        if self._shuffle:
            tokenized_data = gutils.shuffle_agents(tokenized_data)
        return {
            'tokenized_arrays': tokenized_data
        }

    def forward_inference(self, input: Dict) -> Dict:
        """
        Dummy forward
        """
        tokenized_data = self.forward(input)
        return {
            'tokenized_arrays': tokenized_data
        }
