import numpy as np

import torch


def create_rear_region_mask(height: int, width: int,
                            fov_theta: float = 90) -> torch.Tensor:
    """
    Creates a mask that identifies the region behind the ego vehicle.
    The ego vehicle is located at the center of the image, and the rear end is defined as the region behind the ego vehicle.

    :param height (int): height of the image
    :param width (int): width of the image
    :param fov_theta (float): field of view angle in degrees
    """
    # Determine the ego vehicle's position in the center of the image
    ego_vehicle_position = (height // 2, width // 2)

    # Calculate the tangent of half of the fov angle
    tan_half_fov = np.tan(np.radians(fov_theta / 2))

    # Create a triangular mask that identifies the region behind the ego vehicle
    rear_region_mask = torch.zeros((height, width), dtype=torch.bool)
    for y in range(ego_vehicle_position[0], height):
        x_left = int(ego_vehicle_position[1] -
                     tan_half_fov * (y - ego_vehicle_position[0]))
        x_right = int(ego_vehicle_position[1] +
                      tan_half_fov * (y - ego_vehicle_position[0]))
        rear_region_mask[y, x_left:x_right + 1] = 1

    return rear_region_mask
