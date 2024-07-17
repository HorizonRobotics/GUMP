import glob
import logging
import pathlib
from concurrent.futures import Future
from typing import List, Optional, Union

import cv2
from abc import ABC, abstractmethod
import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.callback.abstract_callback import \
    AbstractCallback
from nuplan.planning.simulation.history.simulation_history import (
    SimulationHistory, SimulationHistorySample)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import \
    AbstractTrajectory
from nuplan_extent.planning.simulation.main_callback.utils import save_video
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner

logger = logging.getLogger(__name__)


class SimulationFeatureVideoCallback(AbstractCallback):
    """
    Feature video callback. Render features of selected scenarios during simulation.
    """

    def __init__(
            self,
            simulation_directory: Union[str, pathlib.Path],
            videos_output_dir: Union[str, pathlib.Path],
            feature_log_dir: Union[str, pathlib.Path],
            visualized_scenario_tokens: Optional[List[str]] = [],
            visualize_all_scenarios: bool = False,
            bev_range: List[float] = [-56., -56., 56., 56.],
            image_subfix: str = ".png",
    ):
        """
        Construct simulation feature callback.
        :param simulation_directory: where scenes should be serialized.
        :param videos_output_dir: Folder where to save video logs.
        :param feature_log_dir: Folder where to save feature logs, i.e rendered images.
        :param visualized_scenario_tokens: List of scenario tokens to visualize.
        :param visualize_all_scenarios: Visualize all scenarios.
        :param bev_range: Range of BEV image.
        :param image_subfix: Subfix of image files.
        """
        assert isinstance(visualized_scenario_tokens,
                          list), "visualized_scenario_tokens must be a list"

        self._futures: List[Future[None]] = []
        self._visualized_scenario_tokens = visualized_scenario_tokens
        self._visualize_all_scenarios = visualize_all_scenarios
        self._videos_output_path = pathlib.Path(
            simulation_directory) / videos_output_dir
        self._feature_log_directory = pathlib.Path(
            simulation_directory) / feature_log_dir
        self._subfix = image_subfix
        self._bev_range = bev_range

    def on_initialization_start(self, setup: SimulationSetup,
                                planner: AbstractPlanner) -> None:
        """
        Create directory at initialization
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        scenario_token = setup.scenario.token
        if self._visualize_all_scenarios or (
                scenario_token in self._visualized_scenario_tokens):
            scenario_directory = self._get_scenario_folder(
                planner.name(), setup.scenario)
            feature_log_directory = scenario_directory / "features"
            feature_log_directory.mkdir(exist_ok=True, parents=True)

    def on_initialization_end(self, setup: SimulationSetup,
                              planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""

    def on_step_start(self, setup: SimulationSetup,
                      planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner,
                    sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""

    def on_planner_start(self, setup: SimulationSetup,
                         planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        scenario_token = setup.scenario.token
        if self._visualize_all_scenarios or (
                scenario_token in self._visualized_scenario_tokens):
            scenario_directory = self._get_scenario_folder(
                planner.name(), setup.scenario)
            feature_log_directory = scenario_directory / "features"
            simulation_iteration_index = setup.time_controller.get_iteration(
            ).index
            if isinstance(planner, MLPlanner):
                planner._model_loader._model.set_vis_features(
                    True, feature_log_directory / "{:04d}{}".format(
                        simulation_iteration_index, self._subfix))
            else:
                raise NotImplementedError()
        else:
            planner._model_loader._model.set_vis_features(False, None)

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner,
                       trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""

    def on_simulation_end(self, setup: SimulationSetup,
                          planner: AbstractPlanner,
                          history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        scenario_token = setup.scenario.token
        database_interval = setup.scenario.database_interval
        if self._visualize_all_scenarios or (
                scenario_token in self._visualized_scenario_tokens):
            video_images = []
            scenario_directory = self._get_scenario_folder(
                planner.name(), setup.scenario)
            feature_log_directory = scenario_directory / "features"
            feature_paths = sorted(glob.glob(str(feature_log_directory / "*")))
            for p in feature_paths:
                image = cv2.imread(p)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                video_images.append(image)
            video_save_path = pathlib.Path(
                self._videos_output_path / "feature_selected_scenarios")
            video_name = scenario_token + ".webm"
            if not video_save_path.exists():
                video_save_path.mkdir(parents=True, exist_ok=True)
            save_video(video_images[0].shape[:2],
                       video_images,
                       video_save_path / video_name,
                       database_interval)

    def _get_scenario_folder(self, planner_name: str,
                             scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored.
        :param planner_name: planner name.
        :param scenario: for which to compute directory name.
        :return directory path.
        """
        return self._feature_log_directory / planner_name / scenario.scenario_type / \
            scenario.log_name / scenario.scenario_name  # type: ignore
