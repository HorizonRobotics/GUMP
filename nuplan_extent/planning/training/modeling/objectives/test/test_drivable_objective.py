import unittest

import numpy as np

import torch
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.objectives.drivable_objective import \
    DrivableObjective


class TestDrivableObjective(unittest.TestCase):
    """Test Drivable objective can compute."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = DrivableObjective(
            scenario_type_loss_weighting=None,
            weight=1.0,
            bot_margin=1.8,
            up_margin=7,
            calculate_step=6)
        pred = np.array(
            [
                [[0.5, 0.0, 0.0], [2.5, 2.0, 2.0], [4.5, 4.0, 4.0],
                 [5.5, 5.0, 5.0], [6.5, 6.0, 6.0], [7.5, 7.0, 7.0]],
                [[1.5, 0.0, 0.0], [2.5, 2.0, 2.0], [6.5, 6.0, 6.0],
                 [7.5, 7.0, 7.0], [8.5, 8.0, 8.0], [9.5, 9.0, 9.0]],
            ])

        map_lane = np.array(
            [[[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [4.0, 4.0]],
              [[1.1, 1.1], [2.1, 2.1], [3.1, 3.1], [6.1, 6.1]],
              [[1.2, 1.2], [2.2, 2.2], [3.2, 3.2], [6.2, 6.2]],
              [[1.3, 1.3], [2.3, 2.3], [3.3, 3.3], [6.3, 6.3]],
              [[1.4, 1.4], [2.4, 2.4], [3.4, 3.4], [6.4, 6.4]]],
             [[[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
              [[1.1, 1.1], [2.1, 2.1], [4.1, 4.1], [6.1, 6.1]],
              [[1.2, 1.2], [2.2, 2.2], [4.2, 4.2], [6.2, 6.2]],
              [[1.3, 1.3], [2.3, 2.3], [4.3, 4.3], [6.3, 6.3]],
              [[1.4, 1.4], [2.4, 2.4], [4.4, 4.4], [6.4, 6.4]]]])

        pred = Trajectory(data=torch.from_numpy(pred))
        map_lane = torch.from_numpy(map_lane)

        self.predictions = {"trajectory": pred, "vector_map_lane": map_lane}
        self.target = None
        self.scenarios = None
        self.loss_target = 0

    def test_objective_can_compute(self):
        loss = self.objective.compute(self.predictions, self.target,
                                      self.scenarios)
        self.assertTrue((abs(loss - self.loss_target) < 1e-04).all())


if __name__ == "__main__":
    unittest.main()
