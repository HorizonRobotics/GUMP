import numpy as np
import psutil
import os
import time
import unittest
from nuplan_extent.planning.simulation.logsim_environment import LogSimEnvironment, _get_scenario_files, Action
from nuplan_extent.planning.simulation.rewards import CollisionReward, DrivableAreaComplianceReward
from nuplan_extent.planning.simulation.progress_reward import ProgressReward
from nuplan_extent.planning.simulation.comfort_reward import ComfortReward
from nuplan_extent.planning.training.preprocessing.feature_builders.horizon_raster_feature_builder_v2 import HorizonRasterFeatureBuilderV2
from nuplan_extent.planning.training.preprocessing.target_builders.ego_trajectory_target_builder_v2 import EgoTrajectoryTargetBuilderV2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from alf.utils.spec_utils import consistent_with_spec


class LogSimEnvironmentTest(unittest.TestCase):
    def test_logsim_environment(self):
        batch_size = 10
        num_trajectory_steps = 16

        raster_feature_builder = HorizonRasterFeatureBuilderV2(
            raster_names=[
                'ego', 'past_current_agents', 'roadmap', 'baseline_paths',
                'route', 'ego_speed', 'drivable_area', 'speed_limit'
            ],
            map_features={
                "LANE": 1.0,
                "INTERSECTION": 1.0,
                "STOP_LINE": 0.5,
                "CROSSWALK": 0.5
            },
            target_width=224,  # width of raster passed to the model
            target_height=224,  # height of raster passed to the model
            target_pixel_size=0.5,  # [m] pixel size of raster
            ego_width=2.297,  # [m] width of ego vehicle
            ego_front_length=
            4.049,  # [m] rear axle to front bumper distance of ego vehicle
            ego_rear_length=
            1.127,  # [m] rear axle to rear bumper distance of ego vehicle
            ego_longitudinal_offset=
            0.0,  # [%] offset percentage to move the ego vehicle inside the raster
            baseline_path_thickness=
            1,  # [pixel] the thickness of baseline paths in the baseline_paths_raster
            past_time_horizon=2.0,  # [s] time horizon of all poses
            past_num_poses=4,  # number of poses in a trajectory
            max_speed_normalizer=
            16.0,  # [m/s] use max speed to normalize current speed
        )
        ego_trajectory_target_builder = EgoTrajectoryTargetBuilderV2(
            TrajectorySampling(num_poses=16, time_horizon=8))
        prepared_scenario_list_file = "/data/weixu/data/nuplan_mini_cache/metadata/nuplan_mini_cache_metadata_node_0.csv"

        self._proc = psutil.Process(os.getpid())
        mem0 = self._proc.memory_info().rss // 1e6
        env = LogSimEnvironment(
            num_trajectory_steps=num_trajectory_steps,
            trajectory_step_time=0.5,
            prepared_scenario_files=_get_scenario_files(
                prepared_scenario_list_file)[:1000],
            feature_builders=[raster_feature_builder],
            target_builders=[ego_trajectory_target_builder],
            batch_size=batch_size,
            open_loop_ratio=0.0,
            rewarder_ctors=[
                CollisionReward, DrivableAreaComplianceReward, ProgressReward,
                ComfortReward
            ],
            preload_scenarios=True,
            prioritized_sampling=True,
        )

        steps = np.repeat(
            np.arange(1, num_trajectory_steps + 1, dtype=np.float32)[None, :],
            batch_size,
            axis=0)
        action = Action(x=2 * steps, y=0.5 * steps, heading=0.01 * steps)

        spec = env.time_step_spec()._replace(env_info=env.env_info_spec())
        t0 = time.time()
        for i in range(20):
            ts = env.step(action)
            consistent_with_spec(ts, spec, from_dim=1)
        print("time=", time.time() - t0)
        mem1 = self._proc.memory_info().rss // 1e6
        print("mem=", mem1 - mem0)


if __name__ == '__main__':
    unittest.main()
