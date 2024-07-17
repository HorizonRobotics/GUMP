import unittest

import torch
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.objectives.heatmap_objective import \
    HeatmapObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestHeatmapObjective(unittest.TestCase):
    """Test Bevmap objective can build and train."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = HeatmapObjective(
            scenario_type_loss_weighting=None,
            name='heatmap_objective',
            weight=1.0,
            alpha=2.,
            beta=4.,
            use_logits=True,
            bev_range=[-56., -56., 56., 56.],
            bev_meshgrid=[0.5, 0.5],
            render_steps=[0],
            sigma=[1., 1.],
            decay=1.0)
        self.predictions = dict(
            pred_heatmap=Tensor(data=torch.zeros((1, 16, 224, 224))))
        self.target = dict(trajectory=Trajectory(data=torch.ones((2, 8, 3))))
        self.scenarios = None
        self.loss_target = 138975.0781

    def test_objective_can_build_and_run(self):
        loss = self.objective.compute(self.predictions, self.target,
                                      self.scenarios)
        self.assertTrue((loss - self.loss_target < 1e-03).all())


if __name__ == "__main__":
    unittest.main()
