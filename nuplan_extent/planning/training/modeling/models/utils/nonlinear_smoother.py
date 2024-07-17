from typing import Tuple

import numpy as np
import numpy.typing as npt

from casadi import (DM, Opti, diff, exp, repmat, reshape, sum1, sum2, sumsqr,
                    vertcat)
from nuplan.planning.training.data_augmentation.data_augmentation_util import \
    ConstrainedNonlinearSmoother

Pose = Tuple[float, float, float]  # (x, y, yaw)


class PostSolverSmoother(ConstrainedNonlinearSmoother):
    """
    Smoothing a set of xy observations with a vehicle dynamics model.
    Solved with direct multiple-shooting.
    """

    def __init__(
            self,
            trajectory_len: int,
            dt: float,
            max_col_num_per_step: int,
            max_hm_num_per_step: int,
            collision_flatten: np.ndarray,
            heatmap_flatten: np.ndarray,
            speed_limit: float,
            sigma: float = 0.2,
            sigma_collision: float = 0.2,
            alpha_xy: float = 0.1,
            alpha_yaw: float = 0.1,
            comfort_alpha: float = 1.0,
            alpha_heatmap: float = 0.2,
            alpha_collision: float = 0.5,
    ):
        """
        Constructor method for PostSolverSmoother class.

        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        :param max_col_num_per_step: maximum number of collision points considered per step.
        :param max_hm_num_per_step: maximum number of heatmap points considered per step.
        :param collision_flatten: N x 3 numpy array of collision points, where each row has (x, y, weight) values.
        :param heatmap_flatten: N x 3 numpy array of heatmap points, where each row has (x, y, weight) values.
        :param speed_limit: maximum speed limit.
        :param sigma: sigma for the gaussian kernel, used for heatmap.
        :param sigma_collision: sigma for the gaussian kernel, used for collision.
        :param alpha_xy: weight for the xy cost.
        :param alpha_yaw: weight for the yaw cost.
        :param comfort_alpha: weight for the comfort cost.
        :param alpha_heatmap: weight for the heatmap cost.
        :param alpha_collision: weight for the collision cost.
        """
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.speed_limit = speed_limit

        # fixed parameters
        self.current_index = 0
        self.sigma = sigma  # sigma for the gaussian kernel, used for heatmap
        # sigma for the gaussian kernel, used for collision
        self.sigma_collision = sigma_collision
        self.alpha_xy = alpha_xy
        self.alpha_yaw = alpha_yaw
        self.comfort_alpha = comfort_alpha  # weight for the comfort cost
        self.alpha_heatmap = alpha_heatmap  # weight for the heatmap cost
        self.alpha_collision = alpha_collision  # weight for the collision cost

        self.max_col_num_per_step = max_col_num_per_step
        self.max_hm_num_per_step = max_hm_num_per_step
        self._collision_flatten = collision_flatten
        self._heatmap_flatten = heatmap_flatten

        # Use a array of dts to make it compatible to situations with varying
        # dts across different time steps.
        self._dts: npt.NDArray[np.float32] = np.asarray(
            [[dt] * trajectory_len])
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        Initialize related variables and constraints for optimization.
        """
        self.nx = 4  # state dim
        self.nu = 2  # control dim

        self._optimizer = Opti()  # Optimization problem
        self._create_decision_variables()
        self._create_parameters()
        self._set_dynamic_constraints()
        self._set_state_constraints()
        self._set_control_constraints()
        self._set_objective()

        # Set default solver options (quiet)
        self._optimizer.solver(
            "ipopt", {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.sb": "yes",
                "ipopt.max_cpu_time": 0.35
            })

    def _set_control_constraints(self) -> None:
        """Set the hard control constraints."""
        curvature_limit = 1.0 / 4.0  # 1/m
        self._optimizer.subject_to(
            self._optimizer.bounded(-curvature_limit, self.curvature,
                                    curvature_limit))
        accel_limit = 5.0  # m/s^2
        self._optimizer.subject_to(
            self._optimizer.bounded(-accel_limit, self.accel, accel_limit))

    def _set_state_constraints(self) -> None:
        """Set the hard state constraints."""
        # Constrain the current time -- NOT start of history
        # initial boundary condition
        self._optimizer.subject_to(
            self.state[:, self.current_index] == self.x_curr)

        max_speed = self.speed_limit  # m/s
        self._optimizer.subject_to(
            self._optimizer.bounded(0.0, self.speed,
                                    max_speed))  # only forward
        max_yaw_rate = 2.5  # rad/s
        self._optimizer.subject_to(
            self._optimizer.bounded(-max_yaw_rate,
                                    diff(self.yaw) / self._dts, max_yaw_rate))
        max_lateral_accel = 5.0  # m/s^2, assumes circular motion acc_lat = speed^2 * curvature
        self._optimizer.subject_to(
            self._optimizer.bounded(
                -max_lateral_accel,
                self.speed[:, :self.trajectory_len]**2 * self.curvature,
                max_lateral_accel))

    def _set_objective(self) -> None:
        """Set the objective function. Use care when modifying these weights."""
        # Follow reference, minimize control rates and absolute inputs
        # alpha_rate, alpha_abs, alpha_lat_accel are weights for the cost function
        # those weights are used to balance the cost function, which is fixed by nuplan-devkit
        alpha_rate = self.comfort_alpha * 0.04
        alpha_abs = self.comfort_alpha * 0.08
        alpha_lat_accel = self.comfort_alpha * 0.06
        cost_mlp_trajectory = (
            self.alpha_xy * sumsqr(self.ref_traj[:2, :] -
                                   vertcat(self.position_x, self.position_y)) +
            self.alpha_yaw * sumsqr(self.ref_traj[2, :] - self.yaw) +
            alpha_rate * (sumsqr(self.curvature_rate) + sumsqr(self.jerk)) +
            alpha_abs * (sumsqr(self.curvature) + sumsqr(self.accel)) +
            alpha_lat_accel * sumsqr(self.lateral_accel))

        cost_heatmap = 0.0
        normalizer = 1 / (2.507 * self.sigma)

        # xy shape (2, 17, 40)
        for c in range(1, self.trajectory_len + 1):
            xy_var = repmat(
                reshape(self.state[:2, c], (2, 1)),
                (1, self.max_hm_num_per_step))
            xyw_heatmap = DM(self._heatmap_flatten[:, c - 1].T)  # (3, 40)
            cost_heatmap -= sum2(xyw_heatmap[2, :] * normalizer * exp(-sum1(
                (xy_var - xyw_heatmap[:2, :])**2) / 2 / self.sigma**2))

        cost_collision = 0.0
        normalizer = 1 / (2.507 * self.sigma_collision)
        for c in range(1, self._collision_flatten.shape[1] + 1):
            xy_var = repmat(
                reshape(self.state[:2, c], (2, 1)),
                (1, self.max_col_num_per_step))
            xyw_collision = DM(self._collision_flatten[:, c - 1].T)  # (3, 40)
            cost_collision += sum2(xyw_collision[2, :] * normalizer * exp(
                -sum1((xy_var - xyw_collision[:2, :])**2) / 2 /
                self.sigma_collision**2))

        final_cost = cost_mlp_trajectory + self.alpha_heatmap * \
            cost_heatmap + self.alpha_collision * cost_collision
        self._optimizer.minimize(final_cost)
