from __future__ import annotations

from typing import List, Generator

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.simulation.observation.observation_type import \
    DetectionsTracks
from nuplan.planning.training.preprocessing.features.raster_utils import \
    get_ego_raster
from nuplan.planning.training.preprocessing.features.trajectory_utils import \
    convert_absolute_to_relative_poses
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_raster_feature_builder import \
    HorizonRasterFeatureBuilder
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_raster_feature_builder_v2 import HorizonRasterV2, HorizonRasterFeatureBuilderV2
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_past_current_agents_raster, 
    get_route_raster,
    get_static_agents_raster,
    get_speed_raster,
    get_drivable_area_raster,
    get_wod_baseline_paths_raster, 
    get_wod_roadmap_raster,
    get_traffic_light_dict_from_generator,
    get_traffic_light_circle_raster,
    get_wod_baseline_z_raster)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan_extent.planning.training.preprocessing.features.raster_builders import PreparedScenarioFeatureBuilder
SENSOR_FRAME_TIME_INTERVAL = 0.05


class WaymoRasterFeatureBuilderV2(AbstractFeatureBuilder,
                                  PreparedScenarioFeatureBuilder):

    """
    Raster builder responsible for constructing model input features.
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
            feature_time_interval: float = 0.5,
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
        self._use_uint8 = use_uint8
        self.raster_name = raster_names
        self.map_features = map_features
        self.target_width = target_width
        self.target_height = target_height
        self.target_pixel_size = target_pixel_size

        self.ego_longitudinal_offset = ego_longitudinal_offset
        self.baseline_path_thickness = baseline_path_thickness
        self.raster_shape = (self.target_width, self.target_height)

        x_size = self.target_width * self.target_pixel_size / 2.0
        y_size = self.target_height * self.target_pixel_size / 2.0
        x_offset = 2.0 * self.ego_longitudinal_offset * x_size
        self.x_range = (-x_size + x_offset, x_size + x_offset)
        self.y_range = (-y_size, y_size)

        self.radius = (self.x_range[1] - self.x_range[0]) / 2
        self.ego_width_pixels = int(ego_width / self.target_pixel_size)
        self.ego_front_length_pixels = int(
            ego_front_length / self.target_pixel_size)
        self.ego_rear_length_pixels = int(
            ego_rear_length / self.target_pixel_size)

        self.past_time_horizon = past_time_horizon
        self.past_num_poses = past_num_poses
        self.feature_time_interval = feature_time_interval

        self.max_speed_normalizer = max_speed_normalizer
    
       
    def get_features_from_scenario(self,
                                   scenario: AbstractScenario,
                                   iteration: int = 10) -> HorizonRasterV2:
        # 由于horizon-nuplan中utils_cache.py中的compute_or_load_feature函数中默认了以第0
        # 个iteration为key，雨WOD数据集不符，所以这里的iteration必须hardcode为10
        iteration = 10
        ego_state = scenario.get_ego_state_at_iteration(iteration)
        map_api = scenario.map_api
        detections = scenario.get_tracked_objects_at_iteration(iteration)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        past_ego_states = list(
            scenario.get_ego_past_trajectory(
                iteration=iteration,
                time_horizon=self.past_time_horizon,
                num_samples=self.past_num_poses,
            ))
        past_detections = list(
            scenario.get_past_tracked_objects(
                iteration=iteration,
                time_horizon=self.past_time_horizon,
                num_samples=self.past_num_poses,
            ))
        trajectory_past_relative_poses = convert_absolute_to_relative_poses(
            ego_state.rear_axle,
            [state.rear_axle for state in past_ego_states])
        traffic_light_by_iter = scenario.get_traffic_light_status_at_iteration(
            iteration)  # [0] # get the generator from the tuple        
        result = self._compute_feature(
            ego_state,
            detections,
            map_api,
            route_roadblock_ids,
            trajectory_past_relative_poses,
            past_detections,
            traffic_light_by_iter
        )

        return result

    def _compute_feature(
            self,
            ego_state: EgoState,
            detections: DetectionsTracks,
            map_api: AbstractMap,
            route_roadblock_ids: List[str],
            past_ego_trajectory,
            past_detections: List[DetectionsTracks],
            traffic_light_by_iter: Generator[TrafficLightStatusData, None,
                                             None],
    ) -> HorizonRasterV2:
        # Add task A for 1s.
        # Construct map, agents and ego layers
        len_steps = len(past_detections) if past_detections is not None else 0
        if len_steps == 0:
            past_detections = []
        roadmap_raster = get_wod_roadmap_raster(
            ego_state,
            map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
        )

        # construct speed limit layer, we overlap the speed limit value on lane
        # and lane connector
        speed_limit_raster = np.zeros(self.raster_shape, dtype=np.float32)

        # construct roadblock layer
        # drivable_area_raster = np.zeros(self.raster_shape, dtype=np.float32)
        drivable_area_raster = get_drivable_area_raster(
            ego_state,
            map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            longitudinal_offset=self.ego_longitudinal_offset,
        )

        agents_raster = np.zeros(self.raster_shape, dtype=np.float32)

        # Agent historical data
        for past_step, past_detections in enumerate(
                past_detections + [detections], start=1):
            agents_raster = get_past_current_agents_raster(
                agents_raster,
                ego_state,
                past_detections,
                self.x_range,
                self.y_range,
                self.raster_shape,
                color_value=past_step / (len_steps + 1),
            )
        agents_raster = np.asarray(agents_raster)
        agents_raster = np.flip(agents_raster, axis=0)
        agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

        # Ego_raster current
        ego_raster = get_ego_raster(
            self.raster_shape,
            self.ego_longitudinal_offset,
            self.ego_width_pixels,
            self.ego_front_length_pixels,
            self.ego_rear_length_pixels,
        )

        # Single channel baseline raster
        baseline_paths_raster = get_wod_baseline_paths_raster(
            ego_state,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            self.baseline_path_thickness,
        )

        baseline_z_raster = get_wod_baseline_z_raster(
            ego_state,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            self.baseline_path_thickness,
        )            

        # Navigation block raster generated with expert route.
        route_raster = np.zeros(self.raster_shape, dtype=np.float32)
        # route_raster = get_route_raster(
        #     ego_state,
        #     route_roadblock_ids,
        #     map_api,
        #     self.x_range,
        #     self.y_range,
        #     self.raster_shape,
        #     self.target_pixel_size,
        #     longitudinal_offset=self.ego_longitudinal_offset,
        # )

        # Speed raster filled with the same value of normalized speed.
        ego_speed_raster = get_speed_raster(
            past_ego_trajectory, self.raster_shape, self.feature_time_interval,
            self.max_speed_normalizer)
        
        static_agents_raster = np.zeros(self.raster_shape, dtype=np.float32)
        static_agents_raster = get_static_agents_raster(
            static_agents_raster,
            ego_state,
            detections,
            self.x_range,
            self.y_range,
            self.raster_shape,
            color_value=1.0,
            longitudinal_offset=self.ego_longitudinal_offset,
        )
        static_agents_raster = np.asarray(static_agents_raster)
        static_agents_raster = np.flip(static_agents_raster, axis=0)
        static_agents_raster = np.ascontiguousarray(
            static_agents_raster, dtype=np.float32)

        # get traffic light raster
        traffic_light_by_iter_dict = get_traffic_light_dict_from_generator(traffic_light_by_iter)
        traffic_light_raster = get_traffic_light_circle_raster(
            ego_state,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,  # resolution
            traffic_light_by_iter_dict,
        )

        # Speed raster is forced to be placed at the last channel.
        collated_layers: npt.NDArray[np.float32] = np.dstack([
            ego_raster, # 0
            agents_raster, # 1
            roadmap_raster, # 2
            baseline_paths_raster, # 3
            route_raster, # 4
            ego_speed_raster, # 5 
            drivable_area_raster, # 6
            speed_limit_raster,# 7
            static_agents_raster,
            traffic_light_raster,
            baseline_z_raster
        ]).astype(np.float32)
      #   result = HorizonRaster(data=collated_layers)

        # To hardcode the GUMPV1.1 Waymo feature extraction (jiaqi)
        rasters = {
                  "ego": ego_raster,
                  "past_current_agents": agents_raster,
                  "roadmap": roadmap_raster,
                  "baseline_paths": baseline_paths_raster,
                  "route": route_raster,
                  "ego_speed": ego_speed_raster,
                  "drivable_area": drivable_area_raster,
                  "speed_limit": speed_limit_raster,
                  "static_agents_raster": static_agents_raster,
                  "traffic_light_raster": traffic_light_raster,
                }

        output = {}
        for name in self.raster_name:
            data = rasters[name]
            if self._use_uint8:
                data = np.round(np.minimum(data, 1.0) * 255).astype(
                    np.uint8)
            output[name] = data

        result = HorizonRasterV2(data=output)
        return result
    
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "raster"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return HorizonRasterV2  # type: ignore
    
    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> AbstractModelFeature:
        pass

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        pass

    def get_features_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
    ) -> HorizonRasterV2:
        # TODO: implement this. Wrap current_input and initialization into a
        # scenario object. And then call get_features_from_scenario.
        raise NotImplementedError()
