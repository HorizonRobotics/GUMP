import unittest

# For some unknown reason, "import torch" statement somewhere else in the
# codebase can get stuck. Importing here can avoid the problem.
import torch

import numpy as np

from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_raster_feature_builder import HorizonRasterFeatureBuilder
import nuplan_extent.planning.training.preprocessing.features.raster_builders as rb
from nuplan_extent.planning.scenario_builder.prepared_scenario import PreparedScenario


class TestRasterUtils(unittest.TestCase):
    """Test raster building utility functions."""

    def setUp(self) -> None:
        """
        Initializes DB
        """
        scenario = get_test_nuplan_scenario()
        self.old_features = {}
        for longitudinal_offset in [0.0, 0.25]:
            builder = HorizonRasterFeatureBuilder(
                map_features={
                    "LANE": 1.0,
                    "INTERSECTION": 1.0,
                    "STOP_LINE": 0.5,
                    "CROSSWALK": 0.5
                },
                input_channel_indexes=[
                    0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14
                ],
                target_width=224,  # width of raster passed to the model
                target_height=224,  # height of raster passed to the model
                target_pixel_size=0.5,  # [m] pixel size of raster
                ego_width=2.297,  # [m] width of ego vehicle
                ego_front_length=
                4.049,  # [m] rear axle to front bumper distance of ego vehicle
                ego_rear_length=
                1.127,  # [m] rear axle to rear bumper distance of ego vehicle
                ego_longitudinal_offset=
                longitudinal_offset,  # [%] offset percentage to move the ego vehicle inside the raster
                baseline_path_thickness=
                1,  # [pixel] the thickness of baseline paths in the baseline_paths_raster
                past_time_horizon=2.0,  # [s] time horizon of all poses
                past_num_poses=4,  # number of poses in a trajectory
                feature_time_interval=0.5,  # [s] time interval of each pose
                subsample_ratio_override=
                1.0,  # [%] ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)
                max_speed_normalizer=
                16.0,  # [m/s] use max speed to normalize current speed
            )
            self.old_features[
                longitudinal_offset] = builder.get_features_from_scenario(
                    scenario, 0)
        self.scenario = scenario

    def _test_raster_builder(self, raster_builder, raster_index):
        # change the cache setting so that the objects found from map
        # can match those in HorizonRasterFeatureBuilder
        raster_builder.set_cache_parameter(0.0001, 56.0)
        longitudinal_offset = raster_builder._longitudinal_offset

        prepared_scenario = PreparedScenario()
        iterations = range(0, 1)
        prepared_scenario.prepare_scenario(self.scenario, iterations)
        raster_builder.prepare_scenario(self.scenario, prepared_scenario,
                                        iterations)
        ego_state = prepared_scenario.get_ego_state_at_iteration(0)
        raster = raster_builder.get_features_from_prepared_scenario(
            prepared_scenario, 0, ego_state)
        old_raster = self.old_features[longitudinal_offset].data[:, :,
                                                                 raster_index]
        diff = raster - old_raster
        self.assertLessEqual(
            np.abs(diff).sum(),
            max(raster.sum(), old_raster.sum()) * 1e-2)

    @unittest.skip(
        "Skip because it is slightly different. See code of EgoRasterBuilder for details."
    )
    def test_ego_raster_builder(self):
        for longitudinal_offset in self.old_features:
            ego_raster_builder = rb.EgoRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset,
                ego_width=2.297,
                ego_front_length=4.049,
                ego_rear_length=1.127,
            )
            self._test_raster_builder(ego_raster_builder, 0)

    def test_past_current_agents_raster_builder(self):
        for longitudinal_offset in self.old_features:
            agents_raster_builder = rb.PastCurrentAgentsRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset,
                past_time_horizon=2.0,  # [s] time horizon of all poses
                past_num_steps=4,  # number of past steps in a trajectory
            )
            self._test_raster_builder(agents_raster_builder, 1)

    def test_roadmap_raster_builder(self):
        for longitudinal_offset in self.old_features:
            roadmap_raster_builder = rb.RoadmapRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset,
                map_feature_to_color_dict=dict(
                    LANE=1.0, INTERSECTION=1.0, STOP_LINE=0.5, CROSSWALK=0.5))
            self._test_raster_builder(roadmap_raster_builder, 2)

    def test_baseline_paths_raster_builder(self):
        for longitudinal_offset in self.old_features:
            baseline_paths_raster_builder = rb.BaselinePathsRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset,
                line_thickness=1,
            )
            self._test_raster_builder(baseline_paths_raster_builder, 3)

    def test_route_raster_builder(self):
        for longitudinal_offset in self.old_features:
            route_raster_builder = rb.RouteRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset,
                # the original HorizonRasterFeatureBuilder does not correct
                # artifacts of route
                with_route_roadblock_correction=False,
            )
            self._test_raster_builder(route_raster_builder, 4)

    @unittest.skip("Skip because it is slightly different. See code of"
                   "EgoSpeedRasterBuilder for details.")
    def test_ego_speed_raster_builder(self):
        for longitudinal_offset in self.old_features:
            ego_speed_raster_builder = rb.EgoSpeedRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset,
            )
            self._test_raster_builder(ego_speed_raster_builder, 5)

    def test_drivable_area_raster_builder(self):
        for longitudinal_offset in self.old_features:
            drivable_area_raster_builder = rb.DrivableAreaRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset)
            self._test_raster_builder(drivable_area_raster_builder, 6)

    def test_speed_limit_raster_builder(self):
        for longitudinal_offset in self.old_features:
            speed_limit_raster_builder = rb.SpeedLimitRasterBuilder(
                image_size=224,
                radius=56.0,
                longitudinal_offset=longitudinal_offset,
                none_speed_limit=0.0,
            )
            self._test_raster_builder(speed_limit_raster_builder, 7)


if __name__ == '__main__':
    unittest.main()
