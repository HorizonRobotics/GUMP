import unittest

import torch
from nuplan_extent.planning.training.modeling.objectives.tcp_objective import \
    TCPObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestTCPObjective(unittest.TestCase):
    """Test Bevmap objective can build and train."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        weight = dict(action_weight=1.0, speed_weight=0.05, wp_weight=1.0)
        self.objective = TCPObjective(
            scenario_type_loss_weighting=None, weight=weight, pred_steps=8)
        self.predictions = dict(
            mu_branches=Tensor(data=torch.ones((2, 2))),
            pred_speed=Tensor(data=torch.ones((2, 1))),
            trajectory=Tensor(data=torch.ones((2, 8, 3))),
            future_mu=Tensor(data=torch.ones((2, 7, 2))))
        self.target = dict(
            control=Tensor(data=torch.ones((2, 8, 3))),
            trajectory=Tensor(data=torch.ones((2, 8, 3))),
        )
        self.scenarios = None
        self.loss_target = 0

    def test_objective_can_build_and_run(self):
        loss = self.objective.compute(self.predictions, self.target,
                                      self.scenarios)
        self.assertTrue((loss - self.loss_target < 1e-04).all())


if __name__ == "__main__":
    unittest.main()
