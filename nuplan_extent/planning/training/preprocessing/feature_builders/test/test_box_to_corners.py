import unittest

import numpy as np

import torch
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import \
    agent_bbox_to_corners


class TestAgentBboxToCorners(unittest.TestCase):
    """Test the agent_bbox_to_corners function."""

    def test_agent_bbox_to_corners(self):
        """Test agent_bbox_to_corners function."""
        half_pi = np.pi / 2.0

        bboxes = [
            torch.tensor([[0, 0], [0, 0], [0, 0]]),
            torch.tensor([[1], [1], [1]]),
            torch.tensor([[1], [1], [1]]),
            torch.tensor([[0], [half_pi], [half_pi / 2]])
        ]

        # Call the function
        corners = agent_bbox_to_corners(bboxes)

        # Define the expected output
        expected_corners = torch.tensor([
            [[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
            [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
            [[-0.7071, 0.0000], [0.0000, 0.7071], [0.7071, 0.0000],
             [0.0000, -0.7071]],
        ])

        self.assertTrue((abs(corners - expected_corners) < 1e-04).all())


if __name__ == '__main__':
    unittest.main()
