import unittest

import torch
from nuplan_extent.planning.training.modeling.objectives.e2e_objectives.bevmap_objective import \
    BEVMapObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestBevmapObjective(unittest.TestCase):
    """Test Bevmap objective can build and train."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = BEVMapObjective(scenario_type_loss_weighting=None)
        self.predictions = dict(
            pred_bevmap=Tensor(data=torch.ones((128, 128))))
        self.target = dict(bevmap=Tensor(data=torch.zeros((128, 128))))
        self.scenarios = None
        self.loss_target = 1.3133

    def test_objective_can_build_and_run(self):
        loss = self.objective.compute(self.predictions, self.target,
                                      self.scenarios)
        self.assertTrue((loss - self.loss_target < 1e-04).all())


if __name__ == "__main__":
    unittest.main()
