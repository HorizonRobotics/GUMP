import unittest

import torch
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.modeling.objectives.collision_heatmap_objective import \
    CollisionHeatmapObjective
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestCollisionHeatmapObjective(unittest.TestCase):
    """Test CollisionHeatmapObjective can compute."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.objective = CollisionHeatmapObjective(
            scenario_type_loss_weighting={},
            name="collision_heatmap_objective",
            weight=1.0,
            ego_width=1,
            ego_front_length=1,
            ego_rear_length=1,
            delta_width=0.0,
            delta_length=0.0,
            ego_kernel_range=[-1, -1, 1, 1],
            ego_meshgrid=[1., 1.],
            alpha=1.0,
            use_logits=True,
            compute_step_indexes=[0, 1])
        # Create a dummy input and target
        self.predictions = {
            "pred_heatmap": Tensor(data=torch.ones(1, 2, 4, 4).float())
        }
        self.targets = {
            "trajectory":
                Trajectory(
                    data=torch.tensor([[[0, 0, 0], [0, 0, 0]]]).float()),
            "agents_occupancy_target":
                Tensor(
                    data=torch.tensor([[[
                        [0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]
                    ], [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
                                        ]]).float())
        }
        self.scenarios = None
        self.loss_target = 8.8207

    def test_objective_can_compute(self):
        loss = self.objective.compute(self.predictions, self.targets,
                                      self.scenarios)
        self.assertTrue((abs(loss - self.loss_target) < 1e-04).all())


if __name__ == "__main__":
    unittest.main()
