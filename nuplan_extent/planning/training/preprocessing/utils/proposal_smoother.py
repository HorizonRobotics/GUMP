from typing import Tuple

import numpy as np
import numpy.typing as npt

from casadi import (DM, Opti, diff, exp, repmat, reshape, sum1, sum2, sumsqr,
                    vertcat)
from nuplan.planning.training.data_augmentation.data_augmentation_util import \
    ConstrainedNonlinearSmoother

Pose = Tuple[float, float, float]  # (x, y, yaw)


class ProposalSmoother(ConstrainedNonlinearSmoother):
    """
    Smoothing a set of xy observations with a vehicle dynamics model.
    Solved with direct multiple-shooting.
    """

    def __init__(
            self,
            trajectory_len: int,
            dt: float
    ):
        """
        Constructor method for PostSolverSmoother class.

        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        """
        self.dt = dt
        self.trajectory_len = trajectory_len

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
            })

    def _set_control_constraints(self) -> None:
        """Set the hard control constraints."""
        curvature_limit = 1.0 / 4.0  # 1/m
        self._optimizer.subject_to(
            self._optimizer.bounded(-curvature_limit, self.curvature,
                                    curvature_limit))
        accel_limit = 4.0  # m/s^2
        self._optimizer.subject_to(
            self._optimizer.bounded(-accel_limit, self.accel, accel_limit))

    def _set_state_constraints(self) -> None:
        """Set the hard state constraints."""
        # Constrain the current time -- NOT start of history
        # initial boundary condition
        self._optimizer.subject_to(
            self.state[:, 0] == self.x_curr)

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
        alpha_xy = 1.0
        alpha_yaw = 0.0
        alpha_rate = 0.08
        alpha_abs = 0.08
        alpha_lat_accel = 0.06
        cost_stage = (
            alpha_xy *
            sumsqr(self.ref_traj[:2, :] -
                   vertcat(self.position_x, self.position_y))
            + alpha_yaw * sumsqr(self.ref_traj[2, :] - self.yaw)
            + alpha_rate * (sumsqr(self.curvature_rate) + sumsqr(self.jerk))
            + alpha_abs * (sumsqr(self.curvature) + sumsqr(self.accel))
            + alpha_lat_accel * sumsqr(self.lateral_accel)
        )

        # Take special care with the final state
        alpha_terminal_xy = 1.0
        alpha_terminal_yaw = 0.0  # really care about final heading to help with lane changes
        cost_terminal = alpha_terminal_xy * sumsqr(
            self.ref_traj[:2, -1] -
            vertcat(self.position_x[-1], self.position_y[-1])
        ) + alpha_terminal_yaw * sumsqr(self.ref_traj[2, -1] - self.yaw[-1])

        self._optimizer.minimize(
            cost_stage +
            self.trajectory_len /
            4.0 *
            cost_terminal)
