import unittest

import numpy as np

import torch
from nuplan_extent.planning.training.modeling.models.utils import (
    add_state_dimension, expand_mask, extract_route_bin_from_expert,
    pack_sequence_dim, unpack_sequence_dim, unravel_index)


class TestUtils(unittest.TestCase):
    """
    Tests utils functions in models/utils.
    """

    def test_pack_sequence_dim(self):
        """
        Tests pack_sequence_dim function.
        """
        x = torch.zeros((4, 3, 6, 3, 200, 200))
        x = pack_sequence_dim(x)
        self.assertEqual((12, 6, 3, 200, 200), x.shape)

    def test_unpack_sequence_dim(self):
        """
        Tests unpack_sequence_dim function.
        """
        x = torch.zeros((12, 6, 3, 200, 200))
        x = unpack_sequence_dim(x, 4, 3)
        self.assertEqual((4, 3, 6, 3, 200, 200), x.shape)

    def test_unravel_index(self):
        """
        Tests test_unravel_index function.
        """
        x = torch.LongTensor([400, 500, 200, 1000])
        x = unravel_index(x, (200, 200))
        self.assertTrue(
            torch.all(
                torch.LongTensor([[2, 0], [2, 100], [1, 0], [5, 0]]) == x))

    def test_expand_mask(self):
        # Test case 1: Expand a small mask
        mask = np.array([[False, False, False, False],
                         [False, True, False, False],
                         [False, False, False, False]])
        width = 2
        expected_result = np.array([[True, True, False, False],
                                    [True, True, False, False],
                                    [False, False, False, False]])
        self.assertTrue(
            np.array_equal(expand_mask(mask, width), expected_result))

        # Test case 2: Expand a larger mask
        mask = np.random.choice([True, False], size=(100, 100), p=[0.3, 0.7])
        width = 5
        expanded_mask = expand_mask(mask, width)
        self.assertTrue(expanded_mask.shape == (100, 100))

    def test_add_state_dimension(self):
        feature = torch.randn(3, 6)
        result = add_state_dimension(feature)
        expected_shape = (3, 1, 6)
        self.assertEqual(result.shape, expected_shape)

    def test_extract_route_bin_from_expert(self):
        # Set up mock inputs
        expert_trajectory = torch.zeros(3, 16, 3)
        current_speed = torch.tensor([5.0, 7.0, 6.0])

        # Expected outputs
        expected_longitudinal_bin = torch.tensor([[1., 0., 0.], [1., 0., 0.],
                                                  [1., 0., 0.]])
        expected_latitudinal_bin = torch.tensor([[0., 1., 0.], [0., 1., 0.],
                                                 [0., 1., 0.]])

        # Call the function
        longitudinal_bin, latitudinal_bin = extract_route_bin_from_expert(
            expert_trajectory, current_speed)

        # Assertions
        self.assertTrue(
            torch.equal(longitudinal_bin, expected_longitudinal_bin))
        self.assertTrue(torch.equal(latitudinal_bin, expected_latitudinal_bin))


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
