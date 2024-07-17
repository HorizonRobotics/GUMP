import unittest

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import \
    MockAbstractScenario
from nuplan_extent.planning.training.preprocessing.feature_builders.ego_state_feature_builder import \
    EgoStateFeatureBuilder


class TestEgoStateFeatureBuilder(unittest.TestCase):
    """
    Tests EgoStateFeatureBuilder.
    """

    def setUp(self) -> None:
        self.builder = EgoStateFeatureBuilder()
        self.scenario = MockAbstractScenario()

    def test_ego_state_feature_builder(self):
        ego_state = self.scenario.get_ego_state_at_iteration(0)

        computed_state = self.builder.get_features_from_scenario(
            self.scenario, iteration=0)

        self.assertEqual(ego_state, computed_state)
