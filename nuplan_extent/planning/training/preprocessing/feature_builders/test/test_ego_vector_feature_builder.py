import unittest

from nuplan.planning.scenario_builder.test.mock_abstract_scenario import \
    MockAbstractScenario
from nuplan_extent.planning.training.preprocessing.feature_builders.ego_vector_feature_builder import \
    EgoVectorFeatureBuilder


class TestEgoVectorFeatureBuilder(unittest.TestCase):
    """
    Tests EgoStateFeatureBuilder.
    """

    def setUp(self) -> None:
        self.builder = EgoVectorFeatureBuilder()
        self.scenario = MockAbstractScenario()

    def test_ego_state_feature_builder(self):
        ego_state = self.scenario.get_ego_state_at_iteration(0)

        computed_state = self.builder.get_features_from_scenario(
            self.scenario, iteration=0)

        self.assertEqual(ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                         computed_state.data[3])
        self.assertEqual(
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            computed_state.data[4])
        self.assertEqual(ego_state.tire_steering_angle, computed_state.data[5])
