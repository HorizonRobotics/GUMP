from dataclasses import dataclass
import numpy as np
import torch

from typing import Any, Dict, Type, List, Generator
from nuplan.common.actor_state.ego_state import EgoState
from torch.utils.data.dataloader import default_collate
import nuplan_extent.planning.training.preprocessing.features.raster_builders as rb
from nuplan_extent.planning.scenario_builder.prepared_scenario import NpEgoState, PreparedScenario
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType
from nuplan_extent.planning.training.preprocessing.features.raster_builders import PreparedScenarioFeatureBuilder
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization, )

# mapping from raster name to the number of channels of the raster
RASTER_NAME_TO_CHANNELS = {
    "ego": 1,
    "past_current_agents": 1,
    "roadmap": 1,
    "baseline_paths": 1,
    "route": 1,
    "ego_speed": 1,
    "drivable_area": 1,
    "speed_limit": 1,
    "static_agents_raster": 1,
}


@dataclass
class HorizonRasterV2(AbstractModelFeature):
    """Raster features

    Different from HorizonRaster, this class stores the data as a dictionary with
    the raster name as the key and the raster data as the value.
    """
    data: Dict[str, FeatureDataType]

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        return HorizonRasterV2(data=data)

    def to_feature_tensor(self) -> AbstractModelFeature:
        return HorizonRasterV2(data={name: torch.tensor(data) for name, data in self.data.items()})

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        return HorizonRasterV2(data={name: data.to(device) for name, data in self.data.items()})

    def unpack(self) -> List[AbstractModelFeature]:
        batch_size = list(self.data.values())[0].shape[0]
        return [
            HorizonRasterV2(
                data={name: self.data[name][i]
                      for name in self.data}) for i in range(batch_size)
        ]


class HorizonRasterFeatureBuilderV2(AbstractFeatureBuilder,
                                    PreparedScenarioFeatureBuilder):
    """Raster builder responsible for constructing model input features.

    Different from HorizonRasterFeatureBuilder, this added supports for prepare_scenario()
    and get_features_from_prepared_scenario(), which allows faster feature extraction
    for prepared scenarios.

    Also different from HorizonRasterFeatureBuilder, this class returns the computed
    rasters as a dictionary (HorizonRasterV2)
    """

    def __init__(
            self,
            raster_names: List[str],
            map_features: Dict[str, float],
            target_width: int,
            target_height: int,
            target_pixel_size: float,
            ego_width: float,
            ego_front_length: float,
            ego_rear_length: float,
            ego_longitudinal_offset: float,
            baseline_path_thickness: int,
            past_time_horizon: float,
            past_num_poses: int,
            max_speed_normalizer: float = 16.0,
            use_uint8: bool = False,
    ) -> None:
        """
        Initializes the builder.

        :param raster_names: names of rasters to be built. Supported names are:
            - ego: ego vehicle raster
            - past_current_agents: past and current agents raster
            - roadmap: map raster
            - baseline_paths: baseline paths raster
            - route: route raster
            - ego_speed: ego speed raster
            - drivable_area: drivable area raster
            - speed_limit: speed limit raster
        :param map_features: name of map features to be drawn and their color for encoding.
        :param target_width: [pixels] target width of the raster
        :param target_height: [pixels] target height of the raster
        :param target_pixel_size: [m] target pixel size in meters
        :param ego_width: [m] width of the ego vehicle
        :param ego_front_length: [m] distance between the rear axle and the front bumper
        :param ego_rear_length: [m] distance between the rear axle and the rear bumper
        :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
                                        0.0 means place the ego at 1/2 from the bottom of the raster image.
                                        0.25 means place the ego at 1/4 from the bottom of the raster image.
        :param baseline_path_thickness: [pixels] the thickness of baseline paths in the baseline_paths_raster.
        :param past_time_horizon: [s] time horizon of poses of past feature
        :param past_num_poses: number of poses in a trajectory of past feature
        :param feature_time_interval: [s] time interval of each pose
        :param max_speed_normalizer: [m/s] use max speed to normalize current speed
        :param use_uint8: If True, the raster values will be converted to uint8.
            Any values greater than 1.0 will be clipped to 1.0 and are converted
            to 255. If False, the raster values will be float32.
        """
        assert target_width == target_height, "Target width and height must be equal"
        image_size = target_width
        radius = image_size / 2 * target_pixel_size
        self._radius = radius * 4.0
        self._image_size = image_size
        self._use_uint8 = use_uint8
        builders = {}
        if 'ego' in raster_names:
            builders['ego'] = rb.EgoRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset,
                ego_width=ego_width,
                ego_front_length=ego_front_length,
                ego_rear_length=ego_rear_length)
        if "past_current_agents" in raster_names:
            builders[
                "past_current_agents"] = rb.PastCurrentAgentsRasterBuilder(
                    image_size=image_size,
                    radius=radius,
                    longitudinal_offset=ego_longitudinal_offset,
                    past_time_horizon=past_time_horizon,
                    past_num_steps=past_num_poses)
        if "roadmap" in raster_names:
            builders["roadmap"] = rb.RoadmapRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset,
                map_feature_to_color_dict=map_features)
        if "baseline_paths" in raster_names:
            builders["baseline_paths"] = rb.BaselinePathsRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset,
                line_thickness=baseline_path_thickness)
        if "route" in raster_names:
            builders["route"] = rb.RouteRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset)
        if "ego_speed" in raster_names:
            builders["ego_speed"] = rb.EgoSpeedRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset)
        if "drivable_area" in raster_names:
            builders["drivable_area"] = rb.DrivableAreaRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset)
        if "speed_limit" in raster_names:
            builders["speed_limit"] = rb.SpeedLimitRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset,
                none_speed_limit=0.0,
                max_speed_normalizer=max_speed_normalizer)
        if "static_agents_raster" in raster_names:
            builders["static_agents_raster"] = rb.StaticAgentsRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset,
                past_time_horizon=past_time_horizon,
                past_num_steps=past_num_poses)
        if "traffic_light_raster" in raster_names:
            builders["traffic_light_raster"] = rb.TrafficLightRasterBuilder(
                image_size=image_size,
                radius=radius,
                longitudinal_offset=ego_longitudinal_offset)
        for name in raster_names:
            assert name in builders, f"Raster name {name} is not supported"
        self._builders = builders

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "raster"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return HorizonRasterV2  # type: ignore

    def get_features_from_scenario(self,
                                   scenario: AbstractScenario,
                                   iteration: int = 0) -> HorizonRasterV2:
        iterations = range(iteration, iteration + 1)
        prepared_scenario = PreparedScenario()
        prepared_scenario.prepare_scenario(scenario, iterations)
        for _, builder in self._builders.items():
            # builder.set_cache_parameter(self._radius, 1e-6)
            builder.prepare_scenario(scenario, prepared_scenario, iterations)
        ego_state = prepared_scenario.get_ego_state_at_iteration(iteration)

        rasters = {}
        for name, builder in self._builders.items():
            rasters[name] = builder.get_features_from_prepared_scenario(
                prepared_scenario, iteration, ego_state)

        return HorizonRasterV2(data=rasters)

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> AbstractModelFeature:
        rasters = {}
        for name, builder in self._builders.items():
            raster = builder.get_features_from_prepared_scenario(
                scenario, iteration, ego_state)
            if self._use_uint8:
                raster = np.round(np.minimum(raster, 1.0) * 255).astype(
                    np.uint8)
            rasters[name] = raster

        return HorizonRasterV2(data=rasters)

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:

        for _, builder in self._builders.items():
            builder.prepare_scenario(scenario, prepared_scenario, iterations)

    def get_features_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
    ) -> HorizonRasterV2:
        # TODO: implement this. Wrap current_input and initialization into a
        # scenario object. And then call get_features_from_scenario.
        raise NotImplementedError()
