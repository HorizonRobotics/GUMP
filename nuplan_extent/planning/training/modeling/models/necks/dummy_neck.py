from typing import Dict

import torch.nn as nn


class DummyNeck(nn.Module):
    """
    DummyNeck

    """

    def forward(self, encoder_output: Dict) -> Dict:
        """
        Dummy forward
        """
        if "command" in encoder_output.keys():
            command = encoder_output["command"]
            return {"neck_output": encoder_output["encoder_features"], "command": command}
        else:
            return {"neck_output": encoder_output["encoder_features"]}
