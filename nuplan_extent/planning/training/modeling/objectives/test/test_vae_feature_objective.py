import unittest

import torch
from nuplan_extent.planning.training.modeling.objectives.e2e_objectives.vae_feature_objective import \
    VAEFeatureObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestRoadEdgeObjective(unittest.TestCase):
    """Test RoadEdgeObjective can compute."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = VAEFeatureObjective(
            scenario_type_loss_weighting={},
            weight=1.0,
            target_sigma=2.0,
        )
        self.predictions = {
            "out_feature": Tensor(data=torch.ones((2, 128)) * 2),
        }
        self.targets = None
        self.scenarios = None
        self.loss_target = torch.Tensor([5.5179])[0]

    def test_objective_can_compute(self):
        loss = self.objective.compute(self.predictions, self.targets,
                                      self.scenarios)
        torch.testing.assert_close(self.loss_target, loss)


if __name__ == "__main__":
    unittest.main()
