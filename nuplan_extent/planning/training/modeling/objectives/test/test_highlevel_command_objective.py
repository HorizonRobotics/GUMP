import unittest

import torch
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.objectives.highlevel_command_objective import \
    HighLevelCommandObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestHeatmapObjective(unittest.TestCase):
    """Test Bevmap objective can build and train."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = HighLevelCommandObjective(
            scenario_type_loss_weighting=None)
        self.predictions = dict(
            raster=Tensor(data=torch.zeros((1, 16, 224, 224))),
            pred_longitudinal_bin=Tensor(data=torch.zeros((1, 3))),
            pred_latitudinal_bin=Tensor(data=torch.zeros((1, 3))))
        self.target = dict(trajectory=Trajectory(data=torch.ones((1, 16, 3))))
        self.scenarios = None
        self.loss_target = 2.1972

    def test_objective_can_build_and_run(self):
        loss = self.objective.compute(self.predictions, self.target,
                                      self.scenarios)
        self.assertTrue((loss - self.loss_target < 1e-03).all())


if __name__ == "__main__":
    unittest.main()
