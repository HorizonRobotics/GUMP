import unittest

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import \
    MockAbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import \
    TrajectorySampling
from nuplan.planning.training.preprocessing.features.agents_trajectories import \
    AgentsTrajectories
from nuplan_extent.planning.training.preprocessing.target_builders.agents_trajectory_target_builder import \
    AgentTrajectoryTargetBuilder


class TestAgentsOccupancyTargetBuilder(unittest.TestCase):
    """Test builder that constructs agents occupancy targets during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """

        trajectory_sampling = TrajectorySampling(num_poses=16, time_horizon=8)
        self.target_builder = AgentTrajectoryTargetBuilder(
            future_trajectory_sampling=trajectory_sampling)

    def test_agents_occupancy_target_builder(self):
        """
        Test AgentOccTargetBuilder.
        """
        scenario = MockAbstractScenario(
            number_of_future_iterations=16, number_of_detections=10)
        target = self.target_builder.get_targets(scenario)

        self.assertEqual(type(target), AgentsTrajectories)
        self.assertEqual(type(target.data), list)
        self.assertEqual((16, 10, 8), target.data[0].shape)


if __name__ == '__main__':
    unittest.main()
