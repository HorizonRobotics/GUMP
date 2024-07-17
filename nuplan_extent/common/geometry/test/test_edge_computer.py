import unittest

import numpy as np

from nuplan_extent.common.geometry.edge_computer import EdgeComputer


class TestEdgeComputer(unittest.TestCase):
    def setUp(self):
        self.bev_map = np.ones((100, 100))
        self.bev_map[:, 40] = 0
        self.bev_map[:, 60] = 0
        self.trajectory = np.array([
            [0, 0, 0],
            [1, 1, np.pi / 4],
            [2, 2, np.pi / 4],
        ])
        self.edge_computer = EdgeComputer(self.bev_map, self.trajectory)

    def test_to_pixel(self):
        pixel_trajectory = self.edge_computer.to_pixel(self.trajectory[:, :2])
        expected_pixel_trajectory = np.array([
            [50, 50],
            [52, 48],
            [54, 46],
        ])
        np.testing.assert_array_equal(pixel_trajectory,
                                      expected_pixel_trajectory)

    def test_to_coords(self):
        pixel_trajectory = np.array([
            [50, 50],
            [52, 48],
            [54, 46],
        ])
        coords_trajectory = self.edge_computer.to_coords(pixel_trajectory)
        expected_coords_trajectory = self.trajectory[:, :2]
        np.testing.assert_array_almost_equal(
            coords_trajectory, expected_coords_trajectory, decimal=5)

    def test_get_edges(self):
        left_edges, right_edges = self.edge_computer.get_edges(max_steps=20)
        expected_left_edges = np.array([
            [40.5, 50.],
            [45.28248558, 41.28248558],
            [47.28248558, 39.28248558],
        ])
        expected_right_edges = np.array([
            [59.5, 50.],
            [58.71751442, 54.71751442],
            [59.65685425, 51.65685425],
        ])
        np.testing.assert_array_almost_equal(
            left_edges, expected_left_edges, decimal=5)
        np.testing.assert_array_almost_equal(
            right_edges, expected_right_edges, decimal=5)


if __name__ == '__main__':
    unittest.main()
