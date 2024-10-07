from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
import nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils as kutils

class KineticsRender(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                **kwargs):
        super().__init__()

    def forward(self, tokenized_arrays) -> Dict:
        """
        Dummy forward
        """
        detokenized_arrays = kutils.detokenize_data(tokenized_arrays)
        return detokenized_arrays

    def update_last_frame_data(self, last_tokenized_arrays, pred_agent_tokens) -> Dict:
        """
        Dummy forward
        """
        return kutils.update_last_frame_data(last_tokenized_arrays, pred_agent_tokens)
