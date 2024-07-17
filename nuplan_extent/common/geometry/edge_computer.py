import math

import numpy as np


class EdgeComputer:
    """
    Estimate the left and right road edges given a raster bev map and expert future waypoints.
    For each future waypoint, find its corresponding left and right road edge.
    """

    def __init__(
            self,
            bev_map: np.ndarray,
            trajectory: np.ndarray,
            resolution: float = 0.5,
            y_flip: bool = True,
    ):
        """
        :param bev_map: one channel bev map [H, W], with 1 stands for drivable area and 0 stands for
                        non-drivable area (currently only supports H==W).
        :param trajectory: future waypoints with shape [N, 3], where N is the number of waypoints and
                           for each waypoint it contains (x, y, yaw).
        :param resolution: resolution of the bev map, used to convert trajectory coords value into pixels.
        :param y_flip: whether to flip y coord value while converting, defaults to True for Carla data.
        """
        assert len(
            bev_map.shape) == 2, "The bev map should only have one channel."
        h, w = bev_map.shape
        assert h == w, "The bev map should have same H and W."
        self.map_half_size = h / 2
        self.resolution = resolution
        self.y_flip = y_flip
        self.bev_map = bev_map

        # convert trajectory xy to pixel value in bev map
        self.traj_in_pixel = self.to_pixel(trajectory[:, :2])
        self.angles = trajectory[:, 2]
        # number of waypoints
        self.num_waypoints = trajectory.shape[0]

    def to_pixel(self, trajectory):
        """Convert coords into pixels."""
        assert trajectory.shape[1] == 2
        k = -1 if self.y_flip else 1
        traj_in_pixel = np.zeros_like(trajectory)
        traj_in_pixel[:, 0] = trajectory[:, 0] / \
            self.resolution + self.map_half_size
        traj_in_pixel[:, 1] = k * trajectory[:, 1] / \
            self.resolution + self.map_half_size
        return traj_in_pixel

    def to_coords(self, traj_in_pixel):
        """Convert pixels back into coords."""
        assert traj_in_pixel.shape[1] == 2
        k = -1 if self.y_flip else 1
        trajectory = np.zeros_like(traj_in_pixel)
        trajectory[:, 0] = (
            traj_in_pixel[:, 0] - self.map_half_size) * self.resolution
        trajectory[:, 1] = (
            traj_in_pixel[:, 1] - self.map_half_size) * k * self.resolution
        return trajectory

    def get_traj_in_pixel(self):
        return self.traj_in_pixel

    @staticmethod
    def get_slope(angle):
        """Get the slope given yaw angle."""
        tan_angle = math.tan(angle)
        return tan_angle

    def get_edges(self,
                  max_steps: int,
                  delta_l: float = 0.5,
                  tolerance: int = 0,
                  ideal_steps: int = -1):
        """Estimate left and right edges given the expert waypoints.

        The basic idea is to sample along the line that passes through the waypoint
        and is perpendicular to the expert route continuesly until a non-drivable area
        is found.
        :param max_steps: maximum steps to sample along one direction.
        :param delta_l: the interval between each sample alone the line.
        :param tolerance: the 'delay' to stop sampling, if it's 0 then stop immediately after
                          find the edge, otherwise keep sampling for {tolerance} steps after
                          find the edge.
        :param ideal_steps: the most possible steps to sample that could find the edge, used
                            when bev map is incomplete. In other words, this gives the bev map
                            an 'imaginary' drivable boundary.
        """
        left_edges = []
        right_edges = []
        # record the search steps for both direction in previous waypoints
        self.search_steps = np.zeros((self.num_waypoints, 2))
        self.max_steps = max_steps
        self.delta_l = delta_l
        self.tolerance = tolerance
        self.ideal_steps = ideal_steps if ideal_steps > 0 else max_steps

        for i in range(self.num_waypoints):
            left_edge = self._find_points_one_direction(i, right=False)
            right_edge = self._find_points_one_direction(i, right=True)
            left_edges.append(left_edge)
            right_edges.append(right_edge)
        left_edges = np.array(left_edges)
        right_edges = np.array(right_edges)
        return left_edges, right_edges

    def _find_points_one_direction(
            self,
            waypoint_step: int,
            right: bool,
    ):
        """Sample along one direction.
        :param waypoint_step: indicate the current searching waypoint.
        :param right: indicate the searching direction, whether it's left or right.
        """
        x = self.traj_in_pixel[waypoint_step, 0]
        y = self.traj_in_pixel[waypoint_step, 1]
        slope = self.get_slope(self.angles[waypoint_step])
        delta_x = self.delta_l / (math.sqrt(1 + slope**2))
        road_edge = None
        # if direction is right, x value is incremented
        k = 1 if right else -1
        # column index, left is 0 and right is 1
        col_ind = 1 if right else 0

        # get the sampling steps from the last waypoint for reference.
        if waypoint_step == 0:
            last_steps = self.ideal_steps
        else:
            last_steps = self.search_steps[waypoint_step - 1, col_ind]

        # sample along x in one direction
        found = -1
        step = 1
        while (found != 0) and (step <= self.max_steps):
            if found > 0:
                found -= 1
            x_next = x + k * step * delta_x
            y_next = slope * x_next + (y - slope * x)
            if found == -1 and self.bev_map[round(y_next), round(x_next)] == 0:
                found = self.tolerance
            if found == 0:
                road_edge = (x_next, y_next)
                steps = step
            step += 1

        # if edge is not found after reaching max sampling steps, use the sampling
        # steps from last waypoint as the imaginary edge.
        if road_edge is None:
            x_next = x + k * last_steps * delta_x
            y_next = slope * x_next + (y - slope * x)
            road_edge = (x_next, y_next)
            steps = last_steps

        self.search_steps[waypoint_step, col_ind] = steps
        return road_edge
