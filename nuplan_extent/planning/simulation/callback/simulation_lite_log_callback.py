import logging
import pathlib
from concurrent.futures import Future
from typing import List, Optional, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan_extent.planning.simulation.simulation_lite_log import SimulationLiteLog

logger = logging.getLogger(__name__)


def _save_log_to_file(file_name: pathlib.Path, scenario: AbstractScenario,
                      history: SimulationHistory) -> None:
    """
    Create SimulationLiteLog and save it to disk.
    :param file_name: to write to.
    :param scenario: to store in the log.
    :param history: to store in the log.
    """
    simulation_log = SimulationLiteLog(file_path=file_name,
                                       scenario=scenario,
                                       simulation_history=history)
    simulation_log.save_to_file()
    print(f"Saved simulation log to {file_name}")


class SimulationLiteLogCallback(SimulationLogCallback):
    """
    Callback for simulation logging/object serialization to disk.
    """

    def on_simulation_end(self, setup: SimulationSetup,
                          planner: AbstractPlanner,
                          history: SimulationHistory) -> None:
        """
        On reached_end validate that all steps were correctly serialized.
        :param setup: simulation setup.
        :param planner: planner when simulation ends.
        :param history: resulting from simulation.
        """
        number_of_scenes = len(history)
        if number_of_scenes == 0:
            raise RuntimeError("Number of scenes has to be greater than 0")

        # Create directory
        scenario_directory = self._get_scenario_folder(planner.name(),
                                                       setup.scenario)

        scenario = setup.scenario
        file_name = scenario_directory / scenario.scenario_name
        file_name = file_name.with_suffix(self._file_suffix)
        if self._pool is not None:
            self._futures = []
            self._futures.append(
                self._pool.submit(
                    Task(_save_log_to_file, num_cpus=1, num_gpus=0), file_name,
                    scenario, history))
        else:
            _save_log_to_file(file_name, scenario, history)
