from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    to_tensor,
    FeatureDataType
)


@dataclass
class DictTensorFeature(AbstractModelFeature):
    data_dict: Dict[str, FeatureDataType]

    @classmethod
    def collate(cls, batch: List[DictTensorFeature]) -> DictTensorFeature:
        batch_data = {}
        keys = batch[0].data_dict.keys()
        for key in keys:
            batch_data[key] = torch.stack(
                [b.data_dict[key] for b in batch], dim=0
            )

        return DictTensorFeature(data_dict=batch_data)

    def to_feature_tensor(self) -> DictTensorFeature:
        new_data = {}
        if 'data_dict' in self.data_dict:
            data_dict = self.data_dict['data_dict']
        else:
            data_dict = self.data_dict
        for k, v in data_dict.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                new_data[k] = to_tensor(v)
            else:
                new_data[k] = v
        return DictTensorFeature(data_dict=new_data)

    def to_device(self, device: torch.device) -> DictTensorFeature:
        new_data = {}
        for k, v in self.data_dict.items():
            new_data[k] = v.to(device=device)
        return DictTensorFeature(data_dict=new_data)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DictTensorFeature:
        return DictTensorFeature(data)

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError
        
        
        

