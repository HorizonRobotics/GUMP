import unittest

from nuplan.planning.simulation.trajectory.trajectory_sampling import \
    TrajectorySampling
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.scenario_builder.nuscenes_db.test.test_nuscenes_scenario import \
    mock_nuscenes_scenario
from nuplan_extent.planning.training.preprocessing.target_builders.e2e_ego_trajectory_target_builder import \
    E2EEgoTrajectoryTargetBuilder


class TestE2EEgoTrajectoryTargetBuilder(unittest.TestCase):
    """Test builder that constructs E2E targets during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8)
        self.target_builder = E2EEgoTrajectoryTargetBuilder(
            future_trajectory_sampling=trajectory_sampling, )

    def test_e2e_target_builder(self):
        """
        Test E2EEgoTrajectoryTargetBuilder.
        """
        scenario = mock_nuscenes_scenario(2, 2, 4, False)
        targets = self.target_builder.get_targets(scenario=scenario, )
        self.assertIsInstance(targets, Trajectory)
        self.assertEqual((2, 3), targets.data.shape)

    def test_e2e_target_builder_with_iteration(self):
        """
        Test E2EEgoTrajectoryTargetBuilder with iteration.
        """
        scenario = mock_nuscenes_scenario(2, 2, 10, True)
        targets = self.target_builder.get_targets(
            scenario=scenario,
            iteration=3,
        )
        self.assertIsInstance(targets, Trajectory)
        self.assertEqual((2, 3), targets.data.shape)


if __name__ == '__main__':
    unittest.main()
