import unittest

import torch
from nuplan_extent.planning.training.modeling.objectives.e2e_objectives.road_edges_objective import \
    RoadEdgesObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestRoadEdgeObjective(unittest.TestCase):
    """Test RoadEdgeObjective can compute."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = RoadEdgesObjective(
            scenario_type_loss_weighting={},
            weight=1.0,
        )
        self.predictions = {
            "pred_lanes": Tensor(data=torch.ones((2, 2, 16))),
            "pred_edges": Tensor(data=torch.ones((2, 2, 16)) * 2),
        }
        self.targets = {
            "road_edges": Tensor(data=torch.ones((2, 32, 2)) * 3),
        }
        self.scenarios = None
        self.loss_target = torch.Tensor([2.5000])[0]

    def test_objective_can_compute(self):
        loss = self.objective.compute(self.predictions, self.targets,
                                      self.scenarios)
        torch.testing.assert_close(self.loss_target, loss)


if __name__ == "__main__":
    unittest.main()
