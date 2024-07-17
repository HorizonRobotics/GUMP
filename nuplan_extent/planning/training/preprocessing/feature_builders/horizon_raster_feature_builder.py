from __future__ import annotations

from typing import Dict, Generator, List, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import \
    DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization, PlannerInput)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder, AbstractModelFeature)
from nuplan.planning.training.preprocessing.features.raster_utils import \
    get_ego_raster
from nuplan.planning.training.preprocessing.features.trajectory_utils import \
    convert_absolute_to_relative_poses
from nuplan_extent.planning.training.preprocessing.features.raster import \
    HorizonRaster
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_route_raster,
    get_past_current_agents_raster,
    get_agents_raster,
    get_speed_raster,
    get_drivable_area_raster,
    get_speed_limit_raster,
    get_static_agents_raster,
    get_baseline_paths_raster,
    get_roadmap_raster,
    get_traffic_light_dict_from_generator,
    get_traffic_light_circle_raster
)
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import (
    convert_to_uint8,
    convert_uint16_to_two_uint8,
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import (
    convert_absolute_to_relative_poses,
)
from nuplan_extent.planning.training.modeling.models.utils import shift_down

SENSOR_FRAME_TIME_INTERVAL = 0.05


class HorizonRasterFeatureBuilder(AbstractFeatureBuilder):
    """
    Raster builder responsible for constructing model input features.
    """

    def __init__(
            self,
            map_features: Dict[str, int],
            input_channel_indexes: List[int],
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
            feature_time_interval: float = 0.5,
            subsample_ratio_override: float = 1.0,
            with_instance_mask: bool = False,
            max_speed_normalizer: float = 16.0,
    ) -> None:
        """
        Initializes the builder.
        :param map_features: name of map features to be drawn and their color for encoding.
        :param input_channel_indexes: indexes of input channels to be used.
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
        :param subsample_ratio_override: ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)
        :param with_instance_mask: whether to include instance-wise information in the raster. (default: False)
        :param max_speed_normalizer: [m/s] use max speed to normalize current speed
        """
        self.map_features = map_features
        self.input_channel_indexes = input_channel_indexes
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

        self.subsample_ratio_override = subsample_ratio_override
        self.with_instance_mask = with_instance_mask
        self.max_speed_normalizer = max_speed_normalizer
        
        self.shift_pixels = int(
            (ego_front_length -  (ego_front_length + ego_rear_length)/2)/ self.target_pixel_size
        )

        self.shift_pixels = int(
            (ego_front_length - (ego_front_length + ego_rear_length) / 2) /
            self.target_pixel_size
        )

        self.shift_pixels = int(
            (ego_front_length - (ego_front_length + ego_rear_length) / 2) /
            self.target_pixel_size
        )

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "raster"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return HorizonRaster  # type: ignore

    def get_features_from_scenario(
        self, scenario: AbstractScenario, iteration: int = 0, ego_iteration: int = 0
    ) -> HorizonRaster:
        """Inherited, see superclass."""
        ego_state = scenario.get_ego_state_at_iteration(ego_iteration)
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
            traffic_light_by_iter,
        )

        return result

    def get_features_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
    ) -> HorizonRaster:
        """Inherited, see superclass."""
        key_frame_interval = int(
            self.feature_time_interval / SENSOR_FRAME_TIME_INTERVAL *
            self.subsample_ratio_override)
        history = current_input.history
        ego_state = history.ego_states[-1]
        observation = history.observations[-1]
        route_roadblock_ids = initialization.route_roadblock_ids
        past_trajectory = convert_absolute_to_relative_poses(
            ego_state.rear_axle,
            [history.ego_states[::-key_frame_interval][1].rear_axle],
        )
        past_detection = history.observations[::-key_frame_interval][::-1][:-1]
        traffic_light_by_iter = current_input.traffic_light_data
        feature = self._compute_feature(
            ego_state,
            observation,
            initialization.map_api,
            route_roadblock_ids,
            past_trajectory,
            past_detection,
            traffic_light_by_iter,
        )

        if isinstance(observation, DetectionsTracks):
            return feature
        else:
            raise TypeError(
                f"Observation was type {observation.detection_type()}. Expected DetectionsTracks"
            )

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

        roadmap_raster = get_roadmap_raster(
            ego_state,
            map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            longitudinal_offset=self.ego_longitudinal_offset,
        )

        # construct speed limit layer, we overlap the speed limit value on lane
        # and lane connector
        speed_limit_raster = get_speed_limit_raster(
            ego_state,
            map_api,
            ["LANE", "LANE_CONNECTOR"],
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            self.max_speed_normalizer
        )

        # # construct roadblock layer
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
        past_detections = []
        for past_step, past_detection in enumerate(
                past_detections + [detections], start=1):
            agents_raster = get_past_current_agents_raster(
                agents_raster,
                ego_state,
                past_detection,
                self.x_range,
                self.y_range,
                self.raster_shape,
                longitudinal_offset=self.ego_longitudinal_offset,
                color_value=1.0,
            )
        agents_raster = np.asarray(agents_raster)
        agents_raster = np.flip(agents_raster, axis=0)
        agents_raster = np.ascontiguousarray(agents_raster, dtype=np.float32)

        static_agents_raster = np.zeros(self.raster_shape, dtype=np.float32)
        get_static_agents_raster(
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

        # Ego_raster current
        ego_raster = get_ego_raster(
            self.raster_shape,
            self.ego_longitudinal_offset,
            self.ego_width_pixels,
            self.ego_front_length_pixels,
            self.ego_rear_length_pixels,
        )

        # Single channel baseline raster
        baseline_paths_raster = get_baseline_paths_raster(
            ego_state,
            map_api,
            ['LANE', 'LANE_CONNECTOR'],
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            self.baseline_path_thickness,
            longitudinal_offset=self.ego_longitudinal_offset,
        )

        # # Navigation block raster generated with expert route.
        route_raster = get_route_raster(
            ego_state,
            route_roadblock_ids,
            map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.target_pixel_size,
            longitudinal_offset=self.ego_longitudinal_offset,
        )

        # Speed raster filled with the same value of normalized speed.
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        ego_speed_raster = get_speed_raster(
            past_ego_trajectory, self.raster_shape, self.feature_time_interval,
            self.max_speed_normalizer)

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

        
        raster_list = [
            ego_raster,
            agents_raster,# rear_axle
            roadmap_raster,# rear_axle # 2
            baseline_paths_raster,# rear_axle # 3
            route_raster,# rear_axle 4
            ego_speed_raster, 
            drivable_area_raster, # rear_axle 6
            speed_limit_raster, # rear_axle 7
            static_agents_raster, # rear_axle 8
            traffic_light_raster, # rear_axle 9
        ]
        collated_layers: npt.NDArray[np.float32] = np.dstack(
            raster_list).astype(np.float32)

        result = HorizonRaster(data=collated_layers)
        return result
