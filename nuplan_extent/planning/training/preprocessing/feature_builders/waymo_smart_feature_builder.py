from __future__ import annotations

from typing import List, Generator

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan_extent.planning.training.preprocessing.features.horizon_heterodata import HorizonHeteroData
from third_party.SMART.smart.datasets.preprocess import TokenProcessor
from third_party.SMART.smart.transforms import WaymoTargetBuilder

class WaymoSmartFeatureBuilder(AbstractFeatureBuilder):
    """
    Raster builder responsible for constructing model input features.
    """
    def __init__(
        self,
        num_historical_steps: int = 11, 
        num_future_steps: int = 80,
        token_size: int = 2048
    ) -> None:
        """
        """
        self.token_processor = TokenProcessor(token_size)
        self.transform = WaymoTargetBuilder(num_historical_steps, 
                                            num_future_steps, 
                                            "train")
    
    def get_features_from_scenario(self,
                                   scenario: AbstractScenario,
                                   iteration: int = 0) -> HorizonHeteroData:
        data = scenario.smart_pickle_file
        data = self.token_processor.preprocess(data)
        data = self.transform(data)
        return HorizonHeteroData(data)
    
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "heterodata"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return HorizonHeteroData  # type: ignore

    def get_features_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
    ) -> HorizonHeteroData:
        # TODO: implement this. Wrap current_input and initialization into a
        # scenario object. And then call get_features_from_scenario.
        raise NotImplementedError()
