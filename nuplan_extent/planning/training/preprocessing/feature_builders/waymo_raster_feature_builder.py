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
from nuplan_extent.planning.training.preprocessing.features.raster import \
    HorizonRaster
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_past_current_agents_raster, get_speed_raster,
    get_wod_baseline_paths_raster, get_wod_roadmap_raster,
    get_traffic_light_dict_from_generator,
    get_traffic_light_circle_raster,
    get_wod_baseline_z_raster)

SENSOR_FRAME_TIME_INTERVAL = 0.05


class WaymoRasterFeatureBuilder(HorizonRasterFeatureBuilder):
    """
    Raster builder responsible for constructing model input features.
    """

    def get_features_from_scenario(self,
                                   scenario: AbstractScenario,
                                   iteration: int = 10) -> HorizonRaster:
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
    ) -> HorizonRaster:
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
        drivable_area_raster = np.zeros(self.raster_shape, dtype=np.float32)

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

        # Speed raster filled with the same value of normalized speed.
        ego_speed_raster = get_speed_raster(
            past_ego_trajectory, self.raster_shape, self.feature_time_interval,
            self.max_speed_normalizer)
        
        static_agents_raster = np.zeros(self.raster_shape, dtype=np.float32)

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
        result = HorizonRaster(data=collated_layers)
        return result
