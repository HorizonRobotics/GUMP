from typing import Any, Dict, List, Tuple, Callable, Union
import torch
import torch.nn as nn
import numpy as np
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils

class GUMPRender(nn.Module):
    """
    DummyPostProcessor
    """

    def __init__(self,
                **kwargs):
        super().__init__()

    def forward(self, tokenized_arrays: Dict, hist_tokenized_arrays = None) -> Dict:
        """
        Dummy forward
        """
        detokenized_arrays = gutils.detokenize_data(tokenized_arrays)
        detokenized_arrays = gutils.filter_data(detokenized_arrays, eps=2.0)
        return detokenized_arrays

    def update_last_frame_data(self, last_tokenized_arrays, pred_agent_tokens) -> Dict:
        """
        Dummy forward
        """
        return gutils.update_last_frame_data(last_tokenized_arrays, pred_agent_tokens)
