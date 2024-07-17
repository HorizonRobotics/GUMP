import unittest

import numpy as np

import torch
from nuplan_extent.common.geometry.bbox import (bboxes_to_corners_torch,
                                                rotation_3d_in_axis)


class TestBboxUtils(unittest.TestCase):
    """
    Tests bbox utils functions for NuPlan
    """

    def setUp(self) -> None:
        """Set up test case."""
        self.bbox_torch = torch.Tensor([[0., 0., 0., 1., 2., 3., np.pi / 2]])
        self.target_bbox_corners_torch = torch.Tensor(
            [[[-1.0000, 0.5000, 0.0000], [-1.0000, 0.5000, 3.0000],
              [1.0000, 0.5000, 3.0000], [1.0000, 0.5000, 0.0000],
              [-1.0000, -0.5000, 0.0000], [-1.0000, -0.5000, 3.0000],
              [1.0000, -0.5000, 3.0000], [1.0000, -0.5000, 0.0000]]])

        self.points = torch.Tensor([[[1., 2., 4.]]])
        self.angles = torch.Tensor([np.pi / 2])
        self.axis = 0
        self.target_points_new = torch.Tensor([[[1.0000, -4.0000, 2.0000]]])

    def test_bboxes_to_corners_torch(self):
        """
        test bboxes_to_corners_torch can run
        """
        bbox_corners_torch = bboxes_to_corners_torch(self.bbox_torch)
        self.assertTrue(
            (torch.abs(bbox_corners_torch - self.target_bbox_corners_torch) <
             1e-4).all())

    def test_rotation_3d_in_axis(self):
        """
        test rotation_3d_in_axis can run
        """
        points_new = rotation_3d_in_axis(
            self.points, self.angles, axis=self.axis)
        self.assertTrue(
            (torch.abs(points_new - self.target_points_new) < 1e-4).all())


if __name__ == '__main__':
    unittest.main()
