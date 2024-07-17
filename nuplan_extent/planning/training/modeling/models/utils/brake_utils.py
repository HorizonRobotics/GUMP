import torch


def interpolate_trajectory(trajectory: torch.Tensor,
                           num_steps: int = 320) -> torch.Tensor:
    """
    Interpolates a trajectory with a specified number of steps.

    Args:
        trajectory (torch.Tensor): A tensor of shape (timesteps, dims) representing the input trajectory.
        num_steps (int, optional): The desired number of steps for the output interpolated trajectory. Default is 320.

    Returns:
        torch.Tensor: A tensor of shape (num_steps + 1, dims) representing the interpolated trajectory.
    """
    timesteps, dims = trajectory.shape
    device, dtype = trajectory.device, trajectory.dtype
    interpolated_trajectory = []
    for j in range(dims):
        for i in range(timesteps - 1):
            # Linearly interpolate between adjacent points in the trajectory
            step_values = torch.linspace(trajectory[i, j],
                                         trajectory[i + 1, j],
                                         num_steps // (timesteps - 1) + 1)
            # Remove the last value of the step_values to avoid duplicates
            step_values = step_values[:-1]
            interpolated_trajectory.append(step_values)
    interpolated_trajectory = torch.stack(interpolated_trajectory).reshape(
        dims, -1).T
    interpolated_trajectory = interpolated_trajectory.to(device).to(dtype)
    # Add the last point of the trajectory
    interpolated_trajectory = torch.cat(
        (interpolated_trajectory, trajectory[-1].unsqueeze(0)), dim=0)
    return interpolated_trajectory


def calculate_speed(v: torch.Tensor,
                    a: torch.Tensor,
                    dt: float = 0.5,
                    num_steps: int = 16) -> torch.Tensor:
    """
    Calculates the speed given initial speed, acceleration, time step and number of steps.

    Args:
        v (torch.Tensor): A tensor representing the initial speed.
        a (torch.Tensor): A tensor representing the acceleration.
        dt (float, optional): The time step. Default is 0.5.
        num_steps (int, optional): The number of steps for the speed calculation. Default is 16.

    Returns:
        torch.Tensor: A tensor representing the speed.
    """
    device, dtype = v.device, v.dtype
    time = torch.arange(dt, (num_steps + 1) * dt, dt)
    time = time.to(device).to(dtype)
    speed = v + a * time
    speed[speed < 0] = 0
    return speed


def calculate_position(speed: torch.Tensor, dt: float = 0.5):
    """
    Calculates the position given speed and time step.

    Args:
        speed (torch.Tensor): A tensor representing the speed.
        dt (float, optional): The time step. Default is 0.5.

    Returns:
        torch.Tensor: A tensor representing the position.
    """
    position = torch.cumsum(speed * dt, dim=0)
    return position


def generate_deceleration_trajectory(v: torch.Tensor,
                                     a: torch.Tensor,
                                     dt: float = 0.5,
                                     num_steps: int = 16):
    """
    Generates a deceleration trajectory given initial speed, acceleration, time step and number of steps.

    Args:
        v (torch.Tensor): A tensor representing the initial speed.
        a (torch.Tensor): A tensor representing the acceleration.
        dt (float, optional): The time step. Default is 0.5.
        num_steps (int, optional): The number of steps for the deceleration trajectory. Default is 16.

    Returns:
        torch.Tensor: A tensor of shape (num_steps, 3) representing the deceleration trajectory.
    """
    speed = calculate_speed(v, a, dt, num_steps)
    position = calculate_position(speed, dt)

    deceleration_trajectory = torch.zeros((num_steps, 3))
    deceleration_trajectory[:, 0] = position
    deceleration_trajectory[:, 1] = 0
    deceleration_trajectory[:, 2] = 0

    return deceleration_trajectory


def sample_trajectory(interpolated_trajectory: torch.Tensor,
                      deceleration_trajectory: torch.Tensor,
                      num_steps: int = 16):
    """
    Samples a trajectory from the interpolated_trajectory based on the deceleration_trajectory and num_steps.
    Args:
        interpolated_trajectory (torch.Tensor): A tensor representing the interpolated trajectory.
        deceleration_trajectory (torch.Tensor): A tensor representing the deceleration trajectory.
        num_steps (int): The number of steps for the sampled trajectory.
    Returns:
        torch.Tensor: A tensor of shape (num_steps, 3) representing the sampled trajectory.
    """
    sampled_trajectory = torch.zeros((num_steps, 3))
    current_step = 0
    distance_so_far = 0
    for i in range(num_steps):
        # Find the next point in the interpolated trajectory that is at least
        # as far as the next point in the deceleration trajectory
        while current_step + 1 < len(interpolated_trajectory) and \
                distance_so_far < torch.norm(deceleration_trajectory[i, :2]):
            current_step += 1
            distance_so_far += torch.norm(
                interpolated_trajectory[current_step, :2] -
                interpolated_trajectory[current_step - 1, :2])
        sampled_trajectory[i, :] = interpolated_trajectory[current_step, :]
    return sampled_trajectory
