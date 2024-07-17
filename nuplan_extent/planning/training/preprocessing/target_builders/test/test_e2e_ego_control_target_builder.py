import unittest

from nuplan_extent.planning.scenario_builder.carla_db.test.test_carla_scenario import \
    mock_carla_scenario
from nuplan_extent.planning.training.preprocessing.features.control import \
    Control
from nuplan_extent.planning.training.preprocessing.target_builders.e2e_ego_control_target_builder import \
    E2EEgoControlTargetBuilder


class TestE2EEgoControlTargetBuilder(unittest.TestCase):
    """Test builder that constructs E2E targets during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.target_builder = E2EEgoControlTargetBuilder()

    def test_e2e_target_builder(self):
        """
        Test E2EEgoTrajectoryTargetBuilder.
        """
        scenario = mock_carla_scenario(2, 2, 4, False)
        targets = self.target_builder.get_targets(scenario=scenario, )

        self.assertIsInstance(targets, Control)
        self.assertEqual((3, 3), targets.data.shape)

    def test_e2e_target_builder_with_iteration(self):
        """
        Test E2EEgoTrajectoryTargetBuilder with iteration.
        """
        scenario = mock_carla_scenario(2, 2, 10, True)
        targets = self.target_builder.get_targets(
            scenario=scenario,
            iteration=3,
        )
        self.assertIsInstance(targets, Control)
        self.assertEqual((3, 3), targets.data.shape)


if __name__ == '__main__':
    unittest.main()
