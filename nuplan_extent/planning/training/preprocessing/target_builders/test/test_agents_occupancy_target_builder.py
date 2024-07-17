import unittest

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import \
    MockAbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import \
    TrajectorySampling
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan_extent.planning.training.preprocessing.target_builders.agents_occupancy_target_builder import \
    AgentOccTargetBuilder


class TestAgentsOccupancyTargetBuilder(unittest.TestCase):
    """Test builder that constructs agents occupancy targets during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """

        trajectory_sampling = TrajectorySampling(num_poses=16, time_horizon=8)
        self.target_builder = AgentOccTargetBuilder(
            target_width=224,
            target_height=224,
            target_pixel_size=0.5,
            ego_longitudinal_offset=0.0,
            future_trajectory_sampling=trajectory_sampling)

        self.with_instance_target_builder = AgentOccTargetBuilder(
            target_width=224,
            target_height=224,
            target_pixel_size=0.5,
            ego_longitudinal_offset=0.0,
            future_trajectory_sampling=trajectory_sampling,
            with_instance_mask=True)

    def test_agents_occupancy_target_builder(self):
        """
        Test AgentOccTargetBuilder.
        """
        scenario = MockAbstractScenario(
            number_of_future_iterations=16, number_of_detections=10)
        target = self.target_builder.get_targets(scenario)

        self.assertEqual(type(target), Raster)
        self.assertEqual((224, 224, 16), target.data.shape)

    def test_agents_with_instance_occupancy_target_builder(self):
        """
        Test AgentOccTargetBuilder.
        """
        scenario = MockAbstractScenario(
            number_of_future_iterations=16, number_of_detections=10)
        target = self.with_instance_target_builder.get_targets(scenario)

        self.assertEqual(type(target), Raster)
        self.assertEqual((224, 224, 32), target.data.shape)


if __name__ == '__main__':
    unittest.main()
