import unittest

import numpy as np

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import \
    get_test_nuplan_scenario
from nuplan.planning.training.preprocessing.features.trajectory_utils import \
    convert_absolute_to_relative_poses
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    filter_tracked_objects, get_agents_raster, get_augmented_ego_raster,
    get_drivable_area_raster, get_fut_agents_raster,
    get_past_current_agents_raster, get_roadmap_raster, get_route_raster,
    get_speed_limit_raster, get_speed_raster, get_static_agents_raster,
    get_target_raster)


class TestRasterUtils(unittest.TestCase):
    """Test raster building utility functions."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        scenario = get_test_nuplan_scenario()

        self.x_range = [-56.0, 56.0]
        self.y_range = [-56.0, 56.0]
        self.raster_shape = (224, 224)
        self.resolution = 0.5
        self.thickness = 2
        self.max_speed_normalizer = 16.0

        self.ego_state = scenario.initial_ego_state
        self.map_api = scenario.map_api
        self.tracked_objects = scenario.initial_tracked_objects
        self.map_features = {
            'LANE': 255,
            'INTERSECTION': 255,
            'STOP_LINE': 128,
            'CROSSWALK': 128
        }
        self.route_roadblock_ids = scenario.get_route_roadblock_ids()
        self.iteration = 0
        self.draw_by_polygon_tl = True
        self.draw_ego_route_separately_tl = True
        self.traffic_light_raster_shape = (
            224, 224, 6) if self.draw_ego_route_separately_tl else (224, 224,
                                                                    3)
        self.traffic_light_by_iter = scenario.get_traffic_light_status_at_iteration(
            self.iteration)

        self.offset_yaw = 0.3
        self.offset_xy = (1.0, 1.0)
        ego_width = 2.297
        ego_front_length = 4.049
        ego_rear_length = 1.127
        self.ego_longitudinal_offset = 0.2
        self.ego_width_pixels = int(ego_width / self.resolution)
        self.ego_front_length_pixels = int(ego_front_length / self.resolution)
        self.ego_rear_length_pixels = int(ego_rear_length / self.resolution)

        # 2 second, 4 step, 0.5s interval
        absolute_past_states = list(
            scenario.get_ego_past_trajectory(
                iteration=0, time_horizon=2, num_samples=4))
        self.past_trajectory = convert_absolute_to_relative_poses(
            self.ego_state.rear_axle,
            [state.rear_axle for state in absolute_past_states])
        absolute_future_states = list(
            scenario.get_ego_future_trajectory(
                iteration=0, time_horizon=2, num_samples=4))
        self.future_trajectory = convert_absolute_to_relative_poses(
            self.ego_state.rear_axle,
            [state.rear_axle for state in absolute_future_states])
        self.past_detections = list(
            scenario.get_past_tracked_objects(
                iteration=0, time_horizon=2, num_samples=4))

        future_tracked_objects_lst = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=2, num_samples=4)
        ]
        self.present_future_observations = [
            self.tracked_objects.tracked_objects
        ] + future_tracked_objects_lst

        self.agents_raster = np.zeros(self.raster_shape, dtype=np.float32)
        self.feature_time_interval = 0.5
        self.future_agents_raster_shape = (224, 224, 4)

    def test_get_roadmap_raster(self) -> None:
        """
        Test get_augmented_ego_raster / get_target_raster / get_speed_raster
        """
        # Check if there are tracks in the scene in the first place
        self.assertGreater(len(self.tracked_objects.tracked_objects), 0)

        augmented_ego_raster = get_augmented_ego_raster(
            self.raster_shape, self.ego_longitudinal_offset,
            self.ego_width_pixels, self.ego_front_length_pixels,
            self.ego_rear_length_pixels, self.offset_yaw, self.offset_xy)

        target_raster = get_target_raster(self.future_trajectory, self.x_range,
                                          self.y_range, self.raster_shape)

        speed_raster = get_speed_raster(
            self.past_trajectory, self.raster_shape,
            self.feature_time_interval, self.max_speed_normalizer)

        past_current_agents_raster = get_past_current_agents_raster(
            self.agents_raster,
            self.ego_state,
            self.past_detections[0],
            self.x_range,
            self.y_range,
            self.raster_shape,
        )

        navigation_block_raster = get_route_raster(
            self.ego_state,
            self.route_roadblock_ids,
            self.map_api,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
        )

        filtered_future_objects_lst = filter_tracked_objects(
            self.present_future_observations, reverse=False)
        future_agents_raster_list = []
        for tracked_objects in filtered_future_objects_lst:
            future_agents_raster = get_fut_agents_raster(
                self.ego_state, tracked_objects, self.x_range, self.y_range,
                self.raster_shape)
            future_agents_raster_list.append(future_agents_raster)

        future_agents_rasters = np.dstack(future_agents_raster_list).astype(
            np.uint8)
        speed_limit_raster = get_speed_limit_raster(
            self.ego_state, self.map_api, ["LANE", "LANE_CONNECTOR"],
            self.x_range, self.y_range, self.raster_shape, self.resolution,
            self.max_speed_normalizer)

        drivable_area_raster = get_drivable_area_raster(
            self.ego_state,
            self.map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
        )

        static_agents_raster = np.zeros(self.raster_shape, dtype=np.float32)
        get_static_agents_raster(
            static_agents_raster,
            self.ego_state,
            self.past_detections[0],
            self.x_range,
            self.y_range,
            self.raster_shape,
            color_value=1.0)

        current_agents_raster, current_instances_raster = get_agents_raster(
            self.ego_state,
            self.tracked_objects.tracked_objects,
            self.x_range,
            self.y_range,
            self.raster_shape,
            with_instance_mask=True)

        roadmap_raster = get_roadmap_raster(
            self.ego_state,
            self.map_api,
            self.map_features,
            self.x_range,
            self.y_range,
            self.raster_shape,
            self.resolution,
            self.ego_longitudinal_offset,
        )

        # Check dimensions
        self.assertEqual(augmented_ego_raster.shape, self.raster_shape)
        self.assertEqual(target_raster.shape, self.raster_shape)
        self.assertEqual(speed_raster.shape, self.raster_shape)
        self.assertEqual(past_current_agents_raster.shape, self.raster_shape)
        self.assertEqual(navigation_block_raster.shape, self.raster_shape)
        self.assertEqual(future_agents_rasters.shape,
                         self.future_agents_raster_shape)
        self.assertEqual(speed_limit_raster.shape, self.raster_shape)
        self.assertEqual(drivable_area_raster.shape, self.raster_shape)
        self.assertEqual(static_agents_raster.shape, self.raster_shape)
        self.assertEqual(current_agents_raster.shape, self.raster_shape)
        self.assertEqual(current_instances_raster.shape, self.raster_shape)
        self.assertEqual(roadmap_raster.shape, self.raster_shape)

        # Check if objects are drawn on to the raster
        self.assertTrue(np.any(augmented_ego_raster))
        self.assertTrue(np.any(target_raster))
        self.assertTrue(np.any(speed_raster))
        self.assertTrue(np.any(past_current_agents_raster))
        self.assertTrue(np.any(navigation_block_raster))
        self.assertTrue(np.any(future_agents_raster))
        self.assertTrue(np.any(current_agents_raster))
        self.assertTrue(np.any(current_instances_raster))
        self.assertTrue(np.any(roadmap_raster))

        # Check if value on raster is normalized to 0 to 1
        self.assertTrue(0 <= np.any(augmented_ego_raster) <= 1)
        self.assertTrue(0 <= np.any(target_raster) <= 1)
        self.assertTrue(0 <= np.any(speed_raster) <= 1)
        self.assertTrue(0 <= np.any(past_current_agents_raster) <= 1)
        self.assertTrue(0 <= np.any(navigation_block_raster) <= 1)
        self.assertTrue(0 <= np.any(future_agents_raster) <=
                        TrackedObjectType.GENERIC_OBJECT.value + 1)
        self.assertTrue(0 <= np.any(speed_limit_raster) <= 1)
        self.assertTrue(0 <= np.any(drivable_area_raster) <= 1)
        self.assertTrue(0 <= np.any(static_agents_raster) <= 1)
        self.assertTrue(0 <= np.any(current_agents_raster) <= 1)
        self.assertTrue(0 <= np.any(current_instances_raster))
        self.assertTrue(0 <= np.any(roadmap_raster) <= 1)


if __name__ == '__main__':
    unittest.main()
