import unittest

import torch
from nuplan_extent.planning.training.modeling.objectives.policy_occupancy_objective import \
    PolicyOccupancyObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestOccupancyRasterObjective(unittest.TestCase):
    """Test Occupancy objective can build and train."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = PolicyOccupancyObjective(
            scenario_type_loss_weighting=None, num_frames=1)
        self.predictions = dict(
            occupancy=Tensor(data=torch.ones((1, 2, 128, 128))))
        self.target = dict(
            agents_occupancy_target=Tensor(data=torch.zeros((1, 1, 128, 128))))
        self.scenarios = None
        self.loss_target = 0.6931

    def test_objective_can_build_and_run(self):
        loss = self.objective.compute(self.predictions, self.target,
                                      self.scenarios)
        self.assertTrue((loss - self.loss_target < 1e-04).all())


if __name__ == "__main__":
    unittest.main()
