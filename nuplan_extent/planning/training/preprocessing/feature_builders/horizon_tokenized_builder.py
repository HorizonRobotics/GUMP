from dataclasses import dataclass
import numpy as np
import torch

from typing import Any, Dict, Type, List, Generator
from nuplan.common.actor_state.ego_state import EgoState
from torch.utils.data.dataloader import default_collate
import nuplan_extent.planning.training.preprocessing.features.vector_builders as vb
from nuplan_extent.planning.scenario_builder.prepared_scenario import NpEgoState, PreparedScenario
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType
from nuplan_extent.planning.training.preprocessing.features.raster_builders import PreparedScenarioFeatureBuilder
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization, )

@dataclass
class HorizonTokenizedData(AbstractModelFeature):
    """Raster features

    Different from HorizonRaster, this class stores the data as a dictionary with
    the raster name as the key and the raster data as the value.
    """
    data: np.ndarray

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        return HorizonTokenizedData(data=data)

    @classmethod
    def collate(cls, batch: List[AbstractModelFeature]) -> AbstractModelFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """        
        return cls.deserialize(np.concatenate([cls.serialize(b) for b in batch], axis=0))

    def to_feature_tensor(self) -> AbstractModelFeature:
        return self

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        return self

    def unpack(self) -> List[AbstractModelFeature]:
        return [self.data[i:i+1] for i in range(len(self.data))]


class HorizonTokenizedDataBuilder(AbstractFeatureBuilder):
    """
    A raster builder designed for constructing model input features, extending functionality over the HorizonRasterFeatureBuilder by supporting
    the preparation and feature extraction from prepared scenarios. Unlike its predecessor, this class outputs features as a structured dictionary
    (HorizonRasterV2), accommodating more efficient feature processing.

    Key Differences:
    - Supports prepare_scenario() and get_features_from_prepared_scenario() for quicker feature extraction from prepared scenarios.
    - Outputs features in a dictionary format for enhanced structure and accessibility.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "tokenized_data_32x64_v3"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return HorizonTokenizedData  # type: ignore

    def get_features_from_scenario(self,
                                   scenario: AbstractScenario,
                                   iteration: int = 0) -> HorizonTokenizedData:
        raise NotImplementedError()

    def get_features_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
    ) -> HorizonTokenizedData:
        # TODO: implement this. Wrap current_input and initialization into a
        # scenario object. And then call get_features_from_scenario.
        raise NotImplementedError()
