from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import \
    TrajectorySampling
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import \
    AbstractTargetBuilder
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    filter_tracked_objects, get_agents_raster)


class AgentOccTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self,
                 target_width: int,
                 target_height: int,
                 target_pixel_size: float,
                 ego_longitudinal_offset: float,
                 future_trajectory_sampling: TrajectorySampling,
                 with_instance_mask: Optional[bool] = False) -> None:
        """
        Initializes the builder.
        :param target_width: [pixels] target width of the raster
        :param target_height: [pixels] target height of the raster
        :param target_pixel_size: [m] target pixel size in meters
        :param ego_width: [m] width of the ego vehicle
        :param ego_front_length: [m] distance between the rear axle and the front bumper
        :param ego_rear_length: [m] distance between the rear axle and the rear bumper
        :param ego_longitudinal_offset: [%] offset percentage to place the ego vehicle in the raster.
                                        0.0 means place the ego at 1/2 from the bottom of the raster image.
                                        0.25 means place the ego at 1/4 from the bottom of the raster image.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        :param with_instance_mask: whether to include instance mask in the target
        """
        self.target_width = target_width
        self.target_height = target_height
        self.target_pixel_size = target_pixel_size

        self.ego_longitudinal_offset = ego_longitudinal_offset
        self.raster_shape = (self.target_width, self.target_height)

        x_size = self.target_width * self.target_pixel_size / 2.0
        y_size = self.target_height * self.target_pixel_size / 2.0
        x_offset = 2.0 * self.ego_longitudinal_offset * x_size
        self.x_range = (-x_size + x_offset, x_size + x_offset)
        self.y_range = (-y_size, y_size)

        self._num_future_poses = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon
        self._with_instance_mask = with_instance_mask

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "agents_occupancy_target"

    @classmethod
    def get_feature_type(cls) -> Raster:
        """Inherited, see superclass."""
        return Raster  # type: ignore

    def get_targets(self, scenario: AbstractScenario, iteration=0) -> Raster:
        """Inherited, see superclass."""
        ego_state = scenario.initial_ego_state

        # Retrieve present/future agent boxes
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        future_tracked_objects_lst = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=iteration,
                time_horizon=self._time_horizon,
                num_samples=self._num_future_poses)
        ]
        sampled_future_observations = [present_tracked_objects
                                       ] + future_tracked_objects_lst

        filtered_future_objects_lst = filter_tracked_objects(
            sampled_future_observations, reverse=False)

        future_agents_raster_list = []
        for tracked_objects in filtered_future_objects_lst:
            if self._with_instance_mask:
                future_agents_raster, future_instances_raster = get_agents_raster(
                    ego_state,
                    tracked_objects,
                    self.x_range,
                    self.y_range,
                    self.raster_shape,
                    with_instance_mask=True)
                future_agents_raster_list.append(future_agents_raster)
                future_agents_raster_list.append(future_instances_raster)
            else:
                future_agents_raster = get_agents_raster(
                    ego_state, tracked_objects, self.x_range, self.y_range,
                    self.raster_shape)
                future_agents_raster_list.append(future_agents_raster)

        collated_layers: npt.NDArray[np.int32] = np.dstack(
            future_agents_raster_list).astype(np.int32)

        return Raster(data=collated_layers)
