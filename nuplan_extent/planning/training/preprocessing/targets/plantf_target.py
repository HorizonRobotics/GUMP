from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    to_device,
    to_numpy,
    to_tensor,
)
from torch.nn.utils.rnn import pad_sequence


@dataclass
class PlanTFTarget(AbstractModelFeature):
    data: Dict[str, Any]

    @classmethod
    def collate(cls, feature_list: List[PlanTFTarget]) -> PlanTFTarget:
        batch_data = {}
        batch_data["agent"] = {
            k: pad_sequence(
                [f.data["agent"][k] for f in feature_list], batch_first=True
            )
            for k in feature_list[0].data["agent"].keys()
        }

        return PlanTFTarget(data=batch_data)

    def to_feature_tensor(self) -> PlanTFTarget:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)
        return PlanTFTarget(data=new_data)

    def to_numpy(self) -> PlanTFTarget:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        return PlanTFTarget(data=new_data)

    def to_device(self, device: torch.device) -> PlanTFTarget:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return PlanTFTarget(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PlanTFTarget:
        return PlanTFTarget(data=data)

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    def is_valid(self) -> bool:
        return True

    @classmethod
    def normalize(self, data, hist_steps=21) -> PlanTFTarget:
        cur_state = data["current_state"]
        center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["current_state"][:3] = 0
        data["agent"]["position"] = np.matmul(
            data["agent"]["position"] - center_xy, rotate_mat
        )
        data["agent"]["velocity"] = np.matmul(
            data["agent"]["velocity"], rotate_mat
        )
        data["agent"]["heading"] -= center_angle

        target_position = (
            data["agent"]["position"][:, hist_steps:]
            - data["agent"]["position"][:, hist_steps - 1][:, None]
        )
        target_heading = (
            data["agent"]["heading"][:, hist_steps:]
            - data["agent"]["heading"][:, hist_steps - 1][:, None]
        )
        target = np.concatenate(
            [target_position, target_heading[..., None]], -1
        )
        target[~data["agent"]["valid_mask"][:, hist_steps:]] = 0

        final_target = {
            "agent": {
                "target": target,
                "valid_mask": data["agent"]["valid_mask"],
            }
        }

        return PlanTFTarget(data=final_target)
