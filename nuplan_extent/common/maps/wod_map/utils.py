from typing import List

import numpy as np

import shapely.geometry as geom
from nuplan.common.actor_state.state_representation import StateSE2


def compute_baseline_path_heading(
        baseline_path: geom.linestring.LineString) -> List[float]:
    """
    Compute the heading of each coordinate to its successor coordinate. The last coordinate will have the same heading
    as the second last coordinate.
    :param baseline_path: baseline path as a shapely LineString
    :return: a list of headings associated to each starting coordinate
    """

    coords = np.asarray(baseline_path.coords)
    vectors = np.diff(coords, axis=0)  # type: ignore
    angles = np.arctan2(vectors.T[1], vectors.T[0])
    # type: ignore  # pad end with duplicate heading
    angles = np.append(angles, angles[-1])

    assert len(angles) == len(
        coords
    ), "Calculated heading must have the same length as input coordinates"

    return list(angles)


def extract_discrete_baseline(
        baseline_path: geom.LineString) -> List[StateSE2]:
    """
    Returns a discretized baseline composed of StateSE2 as nodes
    :param baseline_path: the baseline of interest
    :returns: baseline path as a list of waypoints represented by StateSE2
    """
    assert baseline_path.length > 0.0, "The length of the path has to be greater than 0!"

    headings = compute_baseline_path_heading(baseline_path)
    x_coords, y_coords = baseline_path.coords.xy
    return [
        StateSE2(x, y, heading)
        for x, y, heading in zip(x_coords, y_coords, headings)
    ]
