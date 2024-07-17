import unittest

from nuplan_extent.planning.scenario_builder.carla_db.test.test_carla_scenario import \
    mock_carla_scenario
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor
from nuplan_extent.planning.training.preprocessing.target_builders.e2e_occupancy_target_builder import \
    E2EOccupancyTargetBuilder


class TestE2EEgoTrajectoryTargetBuilder(unittest.TestCase):
    """Test builder that constructs E2E targets during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        draw_range = [-50, -50, 50, 50]
        voxel_size = [0.5, 0.5]
        self.target_builder = E2EOccupancyTargetBuilder(
            draw_range=draw_range, voxel_size=voxel_size)

    def test_e2e_target_builder_with_iteration(self):
        """
        Test E2EEgoTrajectoryTargetBuilder with iteration.
        """
        scenario = mock_carla_scenario(2, 2, 10, True)
        targets = self.target_builder.get_targets(
            scenario=scenario,
            iteration=0,
        )
        self.assertIsInstance(targets, Tensor)
        self.assertEqual((1, 200, 200), targets.data.shape)


if __name__ == '__main__':
    unittest.main()
