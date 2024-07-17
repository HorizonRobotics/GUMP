import unittest

import torch
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.objectives.mtp_objective import \
    MTPObjective


class TestMTPObjective(unittest.TestCase):
    """Test Bevmap objective can build and train."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = MTPObjective(
            scenario_type_loss_weighting=None, weight=1.0)
        self.predictions = dict(
            multimode_trajectory=torch.ones((2, 6, 8, 3)),
            pred_log_prob=torch.ones((2, 6)))
        self.target = dict(trajectory=Trajectory(data=torch.ones((2, 8, 3))), )
        self.scenarios = None
        self.loss_target = 0

    def test_objective_can_build_and_run(self):
        loss = self.objective.compute(self.predictions, self.target,
                                      self.scenarios)
        self.assertTrue((loss - self.loss_target < 1e-04).all())


if __name__ == "__main__":
    unittest.main()
