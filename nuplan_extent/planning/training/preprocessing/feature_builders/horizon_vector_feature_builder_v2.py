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
class HorizonVectorV2(AbstractModelFeature):
    """Raster features

    Different from HorizonRaster, this class stores the data as a dictionary with
    the raster name as the key and the raster data as the value.
    """
    data: Dict[str, FeatureDataType]

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        return HorizonVectorV2(data=data)

    @classmethod
    def collate(cls, batch: List[AbstractModelFeature]) -> AbstractModelFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """        
        serialized = [sample.serialize() for sample in batch]
        return cls.deserialize(serialized)


    def to_feature_tensor(self) -> AbstractModelFeature:
        return self
        # def to_tensor(data: FeatureDataType) -> FeatureDataType:
        #     if isinstance(data, np.ndarray):
        #         return torch.tensor(data)
        #     elif isinstance(data, dict):
        #         return {name: to_tensor(data) for name in data}
        #     elif isinstance(data, list):
        #         return [to_tensor(d) for d in data]
        #     else:
        #         return data
        # return HorizonVectorV2(data=to_tensor(self.data))

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        return self
        # def to_device(data: FeatureDataType) -> FeatureDataType:
        #     if isinstance(data, torch.Tensor):
        #         return data.to(device=device)
        #     elif isinstance(data, dict):
        #         return {name: to_device(data) for name in data}
        #     elif isinstance(data, list):
        #         return [to_device(d) for d in data]
        #     else:
        #         return data
        # return HorizonVectorV2(data=to_device(self.data))

    def unpack(self) -> List[AbstractModelFeature]:
        batch_size = list(self.data.values())[0].shape[0]
        return [
            HorizonVectorV2(
                data={name: self.data[name][i]
                      for name in self.data}) for i in range(batch_size)
        ]


class HorizonVectorFeatureBuilderV2(AbstractFeatureBuilder,
                                    PreparedScenarioFeatureBuilder):
    """
    A raster builder designed for constructing model input features, extending functionality over the HorizonRasterFeatureBuilder by supporting
    the preparation and feature extraction from prepared scenarios. Unlike its predecessor, this class outputs features as a structured dictionary
    (HorizonRasterV2), accommodating more efficient feature processing.

    Key Differences:
    - Supports prepare_scenario() and get_features_from_prepared_scenario() for quicker feature extraction from prepared scenarios.
    - Outputs features in a dictionary format for enhanced structure and accessibility.
    """

    def __init__(self,
                agent_features: List[str],
                radius: float = 150,
                longitudinal_offset: float = 0,
                past_time_horizon: float = 2.0,
                past_num_steps: int = 4,
                future_time_horizon: int = 8.0,
                future_num_steps: int = 16,
                num_max_agents: int = 256) -> None:
        """
        Initializes the feature builder with configurations for constructing model input features, focusing on agent-related features
        and trajectory sampling for both past and future states. This setup is essential for creating detailed rasters that reflect
        dynamic traffic scenarios and agent behaviors.

        Parameters:
        - agent_features: A list of strings identifying the features related to agents that should be included in the raster. These features
        can include various attributes such as agent type, motion state, and interactions.
        - num_max_agents: The maximum number of agents to include in the raster. This parameter ensures that the model's computational
        load is manageable while still capturing a comprehensive view of the traffic scenario (default is set to 256).
        This initialization method sets the foundation for a feature builder that is capable of capturing complex traffic scenarios
        through detailed rasters, facilitating accurate and efficient model training and inference.
        """
        self._radius = radius
        builders = {}
        builders['agents'] = vb.PastCurrentAgentsVectorBuilder(
            radius=radius,
            longitudinal_offset=longitudinal_offset,
            past_time_horizon=past_time_horizon,
            past_num_steps=past_num_steps,
            future_time_horizon=future_time_horizon,
            future_num_steps=future_num_steps,
            agent_features=agent_features,
            num_max_agents=num_max_agents)
        builders['ego'] = vb.PastCurrentEgoVectorBuilder(
            radius=radius,
            longitudinal_offset=longitudinal_offset,
            past_time_horizon=past_time_horizon,
            past_num_steps=past_num_steps,
            future_time_horizon=future_time_horizon,
            future_num_steps=future_num_steps)
        self._builders = builders

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "vector"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return HorizonVectorV2  # type: ignore

    def get_features_from_scenario(self,
                                   scenario: AbstractScenario,
                                   iteration: int = 0) -> HorizonVectorV2:
        iterations = range(iteration, iteration + 1)
        prepared_scenario = PreparedScenario()
        prepared_scenario.prepare_scenario(scenario, iterations)
        for _, builder in self._builders.items():
            # builder.set_cache_parameter(self._radius, 1e-6)
            builder.prepare_scenario(scenario, prepared_scenario, iterations)
        ego_state = prepared_scenario.get_ego_state_at_iteration(iteration)

        vectors = {}
        for name, builder in self._builders.items():
            vectors[name] = builder.get_features_from_prepared_scenario(
                prepared_scenario, iteration, ego_state)

        return HorizonVectorV2(data=vectors)

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> AbstractModelFeature:
        vectors = {}
        for name, builder in self._builders.items():
            vector_data = builder.get_features_from_prepared_scenario(
                scenario, iteration, ego_state)
            vectors[name] = vector_data

        return HorizonVectorV2(data=vectors)

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:

        for _, builder in self._builders.items():
            builder.prepare_scenario(scenario, prepared_scenario, iterations)

    def get_features_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
    ) -> HorizonVectorV2:
        # TODO: implement this. Wrap current_input and initialization into a
        # scenario object. And then call get_features_from_scenario.
        raise NotImplementedError()
