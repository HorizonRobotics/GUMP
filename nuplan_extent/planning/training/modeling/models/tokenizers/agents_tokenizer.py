from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
from nuplan_extent.planning.training.preprocessing.feature_builders.vector_tokenizer import (
    VectorTokenizer
)


class AgentsTokenizer(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                num_past_steps,
                dataset,
                target_width,
                target_height,
                rl_training=False,
                **kwargs):
        super().__init__()
        tokenizer_cls = VectorTokenizer
        self.rl_training = rl_training
        self.tokenizer = tokenizer_cls(
            num_past_steps,
            dataset,
            target_width,
            target_height,
            **kwargs
        )

    def forward(self, input: Dict) -> Dict:
        """
        Dummy forward
        """
        if 'sequence_tokens' not in input:
            input['is_simulation'] = True
            input = self.tokenizer.tokenize(input)
        else:
            input['is_simulation'] = False
        return input
