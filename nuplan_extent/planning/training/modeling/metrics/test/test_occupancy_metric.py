import unittest

import torch
from nuplan_extent.planning.training.modeling.metrics.occupancy_metric import \
    IoUError
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestAgentsOccupancyTargetBuilder(unittest.TestCase):
    """Test builder that constructs agents occupancy targets during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """

        self.iou_error = IoUError(num_frames=1, num_classes=2)
        self.predictions = dict(
            occupancy=Tensor(data=torch.ones((1, 2, 128, 128))))
        self.target = dict(
            agents_occupancy_target=Tensor(
                data=torch.zeros((1, 1, 128, 128), dtype=torch.long)))
        self.loss_target = 0.5

    def test_iou_error_builder(self):
        """
        Test AgentOccTargetBuilder.
        """
        loss = self.iou_error.compute(self.predictions, self.target)

        self.assertTrue((loss - self.loss_target < 1e-04).all())
