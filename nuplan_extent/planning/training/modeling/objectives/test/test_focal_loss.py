import unittest

import torch
from nuplan_extent.planning.training.modeling.objectives.focal_loss import \
    FocalLoss


class TestFocalLoss(unittest.TestCase):
    """Test focal loss."""

    def setUp(self) -> None:
        """
        Set up test case
        """
        self.focal_loss = FocalLoss()

        self.predictions = torch.ones([1, 10])
        self.target = torch.zeros([1]).long()
        self.loss_target = 2.3025

    def test_loss_can_compute(self):
        """
        Test focal loss
        """
        loss = self.focal_loss(self.predictions, self.target)
        self.assertTrue((abs(loss - self.loss_target) < 1e-04).all())
