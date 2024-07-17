from typing import Dict

import torch.nn as nn


class DummyPostProcessor(nn.Module):
    """
    DummyPostProcessor

    """

    def forward(self, decoder_output: Dict) -> Dict:
        """
        Dummy forward
        """
        return decoder_output
