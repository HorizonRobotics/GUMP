import unittest

import torch
from nuplan_extent.planning.training.modeling.objectives.speed_heatmap_objective import \
    SpeedHeatmapObjective


class TestSpeedHeatmapObjective(unittest.TestCase):
    """Test Occupancy objective can build and train."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        # Create a test input batch
        batch_size = 2
        num_steps = 8
        height = 64
        width = 64
        self.predictions = {
            'pred_heatmap': torch.zeros(batch_size, num_steps, height, width),
            'current_speed_limit': torch.ones(batch_size, ) * 6
        }
        self.targets = {'trajectory': torch.randn(batch_size, num_steps, 2)}

        # Instantiate the objective and compute the loss
        self.objective = SpeedHeatmapObjective(
            scenario_type_loss_weighting=None,
            weight=1.0,
            alpha=0.0,
            tolerance=0.5,
            use_logits=True)
        self.scenarios = None
        self.loss_target = 4.4958

    def test_objective_can_build_and_run(self):
        loss = self.objective.compute(self.predictions, self.targets,
                                      self.scenarios)
        self.assertTrue((loss - self.loss_target < 1e-04).all())


if __name__ == "__main__":
    unittest.main()
