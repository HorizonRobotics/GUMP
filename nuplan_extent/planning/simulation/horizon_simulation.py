from __future__ import annotations

import logging
from typing import Optional

from nuplan.planning.simulation.callback.abstract_callback import \
    AbstractCallback
from nuplan.planning.simulation.history.simulation_history_buffer import \
    SimulationHistoryBuffer
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan_extent.planning.simulation.planner.abstract_planner import \
    HorizonPlannerInitialization
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.history.simulation_history import SimulationHistorySample

logger = logging.getLogger(__name__)


class HorizonSimulation(Simulation):
    """
    This class queries data for initialization of a planner, and propagates simulation a step forward based on the
        planned trajectory of a planner.
    """

    def __init__(
            self,
            simulation_setup: SimulationSetup,
            callback: Optional[AbstractCallback] = None,
            simulation_history_buffer_duration: float = 2,
    ):
        """
        Create Simulation.
        :param simulation_setup: Configuration that describes the simulation.
        :param callback: A callback to be executed for this simulation setup
        :param simulation_history_buffer_duration: [s] Duration to pre-load scenario into the buffer.
        """
        super(HorizonSimulation, self).__init__(
            simulation_setup,
            callback=callback,
            simulation_history_buffer_duration=
            simulation_history_buffer_duration)

    def initialize(self) -> HorizonPlannerInitialization:
        """
        Initialize the simulation
         - Initialize Planner with goals and maps
        :return data needed for planner initialization.
        """
        self.reset()

        # Initialize history from scenario
        self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
            self._history_buffer_size, self._scenario,
            self._observations.observation_type())

        # Initialize observations
        self._observations.initialize()

        # Add the current state into the history buffer
        self._history_buffer.append(self._ego_controller.get_state(),
                                    self._observations.get_observation())

        # Return the planner initialization structure for this simulation
        return HorizonPlannerInitialization(
            expert_goal_state=self._scenario.get_expert_goal_state(),
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            mission_goal=self._scenario.get_mission_goal(),
            map_api=self._scenario.map_api,
            scenario=self._scenario)

    def propagate(self, trajectory: AbstractTrajectory) -> None:
        """
        Propagate the simulation based on planner's trajectory and the inputs to the planner
        This function also decides whether simulation should still continue. This flag can be queried through
        reached_end() function
        :param trajectory: computed trajectory from planner.
        """
        if self._history_buffer is None:
            raise RuntimeError("Simulation was not initialized!")

        if not self.is_simulation_running():
            raise RuntimeError("Simulation is not running, simulation can not be propagated!")

        # Measurements
        iteration = self._time_controller.get_iteration()
        ego_state = self._ego_controller.get_state()
        observation = self._observations.get_observation()
        traffic_light_status = list(self._scenario.get_traffic_light_status_at_iteration(iteration.index))

        # Add new sample to history
        logger.debug(f"Adding to history: {iteration.index}")
        self._history.add_sample(
            SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status)
        )

        # Propagate state to next iteration
        next_iteration = self._time_controller.next_iteration()

        # Propagate state
        if next_iteration:
            self._ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            self._observations.update_observation(iteration, next_iteration, self._history_buffer)
            # self._observations.update_observation(iteration, next_iteration, self._history_buffer, self._ego_controller.get_state())
        else:
            self._is_simulation_running = False

        # Append new state into history buffer
        self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())
