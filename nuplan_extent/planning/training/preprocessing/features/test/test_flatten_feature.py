import unittest

import numpy as np

from nuplan_extent.planning.training.preprocessing.features.flatten_feature import \
    FlattenFeature


class TestFlattenFeature(unittest.TestCase):
    """Test Camera end to end representation."""

    def setUp(self) -> None:
        self.bs = 2
        self.states = 8
        self.feature_size = 128
        self.feature = np.zeros((self.states, self.feature_size))
        self.batched_feature = np.zeros((self.bs, self.states,
                                         self.feature_size))

    def test_flatten_feature(self) -> None:
        """
        Test the core functionality of features.
        """
        feature = FlattenFeature(data=self.feature)
        self.assertEqual(None, feature.num_batches)
        self.assertEqual(self.states, feature.num_of_iterations)
        self.assertEqual(self.feature_size, feature.state_size)
        self.assertTrue(
            np.array_equal(self.feature[..., 4, :], feature.state_at_index(4)))

    def test_batched_flatten_feature(self) -> None:
        """
        Test the core functionality of batched features.
        """
        batched_feature = FlattenFeature(data=self.batched_feature)
        self.assertEqual(self.bs, batched_feature.num_batches)
        self.assertEqual(self.states, batched_feature.num_of_iterations)
        self.assertEqual(self.feature_size, batched_feature.state_size)
        self.assertTrue(
            np.array_equal(self.batched_feature[..., 4, :],
                           batched_feature.state_at_index(4)))


if __name__ == '__main__':
    unittest.main()
