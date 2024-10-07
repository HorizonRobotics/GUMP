from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import copy
import torch.nn as nn
import numpy as np
import pickle
import time
import nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils as kutils



def sampling_start_index(tokenized_data):
    # Assume tokenized_data is a tensor of shape [batch_size, seq_length]
    seq_length = tokenized_data.shape[1]
    max_start = seq_length - 5  # Ensure there's room for the subsequent tokens

    # Step 1: Define the range of possible start indices
    indices = torch.arange(0, max_start, dtype=torch.float32)

    # Step 2: Create an exponential decay distribution
    # Adjust the lambda_decay parameter to control the rate of decay
    lambda_decay = 0.05  # Higher values lead to faster decay
    probabilities = torch.exp(-lambda_decay * indices)

    # Normalize the probabilities so they sum to 1
    probabilities /= probabilities.sum()

    # Step 3: Sample a start index based on the exponential decay probabilities
    start_index = torch.multinomial(probabilities, num_samples=1)

    # If you need the start_index as a scalar integer
    start_index = start_index.item()

    return start_index

class KineticsCachedTokenizer(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                max_seq_len: int = 416,
                random_start: bool = False,):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._random_start = random_start
        self.rand_prob = 1.0
        # self._shuffle = shuffle

    def forward(self, input: Dict) -> Dict:
        """
        Dummy forward
        """
        for key in input.keys():
            if "tokenized_data" in key:
                tokenized_data = input[key].data
                if self.training and self._random_start and np.random.rand() < self.rand_prob:
                    start_index = sampling_start_index(tokenized_data)
                    tokenized_data = tokenized_data[:, start_index:]
        return {"tokenized_arrays": tokenized_data}

    def forward_train(self, input: Dict) -> Dict:
        return self.forward(input)

    def forward_inference(self, input: Dict) -> Dict:
        """
        Dummy forward
        """
        return self.forward(input)
