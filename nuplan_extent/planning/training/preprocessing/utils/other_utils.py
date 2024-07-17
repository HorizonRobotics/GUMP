import numba
import numpy as np


def normalize_angle(angle: np.ndarray):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@numba.njit
def rotate_round_z_axis(points: np.ndarray, angle: float):
    rotate_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return points @ rotate_mat
