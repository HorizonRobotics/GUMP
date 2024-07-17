from typing import Dict, List

import torch
import torch.nn as nn


class GUMPNaivePlanner(nn.Module):
    """
    Naive planner that takes the predicted control logits and selects the control action with the highest probability.
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions: Dict) -> Dict:
        """
        Forward function of GUMPNaivePlanner.

        Args:
            predictions (Dict): A dictionary of the encoder output. Must contain the 'pred_control_logits' tensor of shape (B, 3).

        Returns:
            Dict: A dictionary of the post-processor output.
        """
        import pdb; pdb.set_trace()