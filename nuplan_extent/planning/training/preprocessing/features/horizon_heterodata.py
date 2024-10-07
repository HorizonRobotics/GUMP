from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.typing import TensorFrame, torch_frame

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType

class Collater:
    def __init__(
        self,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")
    
@dataclass
class HorizonHeteroData(AbstractModelFeature):
    """
    Dataclass that holds map/environment signals in a raster (HxWxC) or (CxHxW) to be consumed by the model.

    :param ego_layer: raster layer that represents the ego's position and extent
    :param agents_layer: raster layer that represents the position and extent of agents surrounding the ego
    :param roadmap_layer: raster layer that represents map information around the ego
    """

    data: FeatureDataType

    @property
    def num_batches(self) -> Optional[int]:
        """Number of batches in the feature."""
        return len(self.data['scenario_id'])

    def to_feature_tensor(self) -> AbstractModelFeature:
        """Implemented. See interface."""
        return HorizonHeteroData(data=self.data)
    
    def to_device(self, device: torch.device) -> HorizonHeteroData:
        """Implemented. See interface."""
        return HorizonHeteroData(data=self.data.to(device, non_blocking=True))

    def serialize(self) -> Dict[str, Any]:
        return self.data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> HorizonHeteroData:
        """Implemented. See interface."""
        raise NotImplementedError("Not implemented yet.")

    def unpack(self) -> List[HorizonHeteroData]:
        """Implemented. See interface."""
        raise NotImplementedError("Not implemented yet.")
    
    @classmethod
    def collate(cls, batch: List[AbstractModelFeature]) -> AbstractModelFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        serialized = [sample.serialize() for sample in batch]
        return HorizonHeteroData(data=Collater()(serialized))

    @staticmethod
    def from_feature_tensor(tensor: torch.Tensor) -> HorizonHeteroData:
        """Implemented. See interface."""
        HorizonHeteroData(data=tensor)
