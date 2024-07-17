import unittest

import numpy as np

import torch
from nuplan_extent.planning.training.modeling.objectives.utils import (
    binary_dilation_torch, collision_checking_torch, draw_rotated_rect,
    ego_occupancy_kernel_render)


class TestUtils(unittest.TestCase):
    """
    Tests utils functions in models/utils.
    """

    def test_binary_dilation_torch(self):
        """
        Tests binary_dilation_torch function.
        """
        # Test case 1: Test binary dilation on an input tensor with all zeros
        input_tensor = torch.zeros((1, 1, 10, 10))
        struct_shape = (3, 3)
        dilated_tensor = binary_dilation_torch(input_tensor, struct_shape)
        self.assertTrue(torch.equal(dilated_tensor, input_tensor))

        # Test case 2: Test binary dilation on an input tensor with all ones
        input_tensor = torch.ones((1, 1, 10, 10))
        dilated_tensor = binary_dilation_torch(input_tensor, struct_shape)
        expected_tensor = torch.ones((1, 1, 10, 10))
        self.assertTrue(torch.equal(dilated_tensor, expected_tensor))

        # Test case 3: Test binary dilation on an input tensor with a single
        # channel and a non-square shape
        input_tensor = torch.zeros((1, 1, 5, 10))
        dilated_tensor = binary_dilation_torch(input_tensor, struct_shape)
        expected_tensor = torch.zeros((1, 1, 5, 10))
        self.assertTrue(torch.equal(dilated_tensor, expected_tensor))

    def test_collision_checking_torch(self):
        input = torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]]).float()
        kernel = torch.ones((1, 1, 3, 3)).float()
        expected_output = torch.tensor([[[[2, 3, 2], [3, 4, 3],
                                          [2, 3, 2]]]]).float()
        output = collision_checking_torch(input, kernel)
        self.assertTrue(torch.equal(output, expected_output))

    def test_draw_rotated_rect(self):
        # Define input batch of rectangles
        batch_rect = torch.tensor([[0.0, 0.0, 0.2, 0.2, 0.2,
                                    np.pi / 4]]).float()

        # Define raster map dimensions
        H, W = 10, 10

        # Compute output using the function under test
        output = draw_rotated_rect(batch_rect, H, W)

        # Define the expected output
        expected_output = torch.tensor(
            [[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0.1967, 1.0000, 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1.0000, 0.1967, 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]]).float()
        # Check that the computed output matches the expected output
        self.assertTrue(torch.allclose(output, expected_output))

    def test_ego_occupancy_kernel_render(self):
        # Define inputs
        batch_size = 1
        num_steps = 1
        trajectory = torch.zeros((batch_size, num_steps, 3))
        trajectory[:, :, 0] = torch.tensor([[0]])
        trajectory[:, :, 1] = torch.tensor([[0]])
        trajectory[:, :, 2] = torch.tensor([[0]])
        bev_range = [0, 0, 2, 2]
        bev_meshgrid = [0.5, 0.5]
        ego_width = 1.8
        ego_front_length = 4.5
        ego_rear_length = 1.5

        # Execute function
        output = ego_occupancy_kernel_render(trajectory, bev_range,
                                             bev_meshgrid, ego_width,
                                             ego_front_length, ego_rear_length)
        except_output = torch.tensor([[[0.8333, 1.0000, 1.0000, 0.8333],
                                       [0.8333, 1.0000, 1.0000, 0.8333],
                                       [0.8333, 1.0000, 1.0000, 0.8333],
                                       [0.8333, 1.0000, 1.0000, 0.8333]]])
        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_steps, 4, 4))
        self.assertTrue(torch.allclose(output, except_output, 1e-4))


if __name__ == '__main__':
    unittest.main()
