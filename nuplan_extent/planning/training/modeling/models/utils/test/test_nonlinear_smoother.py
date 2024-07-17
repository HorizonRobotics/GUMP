import unittest

import numpy as np

from nuplan_extent.planning.training.modeling.models.utils import \
    PostSolverSmoother


class TestPostSolverSmoother(unittest.TestCase):
    def setUp(self):
        self.trajectory_len = 10
        self.dt = 0.5
        self.max_col_num_per_step = 40
        self.max_hm_num_per_step = 40
        self.collision_flatten = np.zeros((40, self.trajectory_len, 3))
        self.heatmap_flatten = np.zeros((40, self.trajectory_len, 3))
        self.speed_limit = 10.0

        self.smoother = PostSolverSmoother(
            trajectory_len=self.trajectory_len,
            dt=self.dt,
            max_col_num_per_step=self.max_col_num_per_step,
            max_hm_num_per_step=self.max_hm_num_per_step,
            collision_flatten=self.collision_flatten,
            heatmap_flatten=self.heatmap_flatten,
            speed_limit=self.speed_limit)

    def test_set_reference_trajectory(self):
        x_curr = [0.0, 0.0, 0.0, 1.0]
        reference_trajectory = [[0.0, 0.0, 0.0]] * (self.trajectory_len + 1)
        self.smoother.set_reference_trajectory(x_curr, reference_trajectory)
        self.assertEqual(
            self.smoother._optimizer.value(self.smoother.x_curr).tolist(),
            x_curr)
        self.assertEqual(
            self.smoother._optimizer.value(self.smoother.ref_traj).T.tolist(),
            reference_trajectory)

    def test_solve(self):
        x_curr = [0.0, 0.0, 0.0, 1.0]
        reference_trajectory = [[0.0, 0.0, 0.0]] * (self.trajectory_len + 1)
        self.smoother.set_reference_trajectory(x_curr, reference_trajectory)
        solution = self.smoother.solve()
        self.assertIsNotNone(solution)


if __name__ == "__main__":
    unittest.main()
