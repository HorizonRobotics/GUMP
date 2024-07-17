import unittest

import torch
from nuplan_extent.planning.training.modeling.models.utils.brake_utils import (
    calculate_position, calculate_speed, generate_deceleration_trajectory,
    interpolate_trajectory, sample_trajectory)


class TestUtils(unittest.TestCase):
    """
    Tests utils functions in models/utils.
    """

    def test_trajectory_functions(self):
        trajectory = torch.tensor([[0, 0, 0], [10, 0, 0], [20, 0, 0]],
                                  dtype=torch.float32)
        num_steps = 320

        # Test interpolate_trajectory
        interpolated_traj = interpolate_trajectory(trajectory, num_steps)
        assert interpolated_traj.shape == (num_steps + 1, 3)

        # Test calculate_speed
        v = torch.tensor([10.0], dtype=torch.float32)
        a = torch.tensor([-2.0], dtype=torch.float32)
        dt = 0.5
        num_steps_speed = 16
        speed = calculate_speed(v, a, dt, num_steps_speed)
        assert speed.shape == (num_steps_speed, )

        # Test calculate_position
        position = calculate_position(speed, dt)
        assert position.shape == (num_steps_speed, )

        # Test generate_deceleration_trajectory
        deceleration_trajectory = generate_deceleration_trajectory(
            v, a, dt, num_steps_speed)
        assert deceleration_trajectory.shape == (num_steps_speed, 3)

        # Test sample_trajectory
        sampled_traj = sample_trajectory(
            interpolated_traj, deceleration_trajectory, num_steps_speed)
        assert sampled_traj.shape == (num_steps_speed, 3)


if __name__ == '__main__':
    unittest.main()
