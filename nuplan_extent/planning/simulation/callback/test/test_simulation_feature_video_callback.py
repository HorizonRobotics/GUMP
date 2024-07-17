import os
import pathlib
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import \
    AbstractTrajectory
from nuplan_extent.planning.simulation.callback.simulation_feature_video_callback import \
    SimulationFeatureVideoCallback


class MockScenario:
    token = "test_scenario"
    scenario_type = "test_scenario"
    scenario_name = "test_scenario"
    log_name = "test_scenario"
    database_interval = 1.0


class MockSimulationSetup:
    scenario = MockScenario()


# Set up a planner object


class MockPlanner:
    def __init__(self):
        self._model_loader = MockModelLoader()
        self._name = "mock_planner"

    def plan(self, setup: SimulationSetup) -> AbstractTrajectory:
        pass

    def name(self) -> str:
        return self._name


class MockModelLoader:
    def __init__(self):
        self._model = MockModel()


class MockModel:
    _is_vis_features = False
    _vis_features_path = None

    def set_vis_features(self, is_vis_features: bool, vis_features_path: str):
        self._is_vis_features = is_vis_features
        self._vis_features_path = vis_features_path


class TestSimulationFeatureVideoCallback(unittest.TestCase):
    """Test SimulationFeatureVideoCallback."""

    def fake_and_save_images(self, image_dir: str, image_subfix: str,
                             num_images: int):
        """Fake and save images."""
        os.makedirs(image_dir, exist_ok=True)
        for i in range(num_images):
            image = np.random.randint(
                0, 255, size=(100, 100, 3), dtype=np.uint8)
            cv2.imwrite(image_dir + "/{:04d}{}".format(i, image_subfix), image)

    def remove_temp_dir(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def setUp(self) -> None:
        """Set up."""
        self.temp_dir = tempfile.mkdtemp()
        # Set up the parameters for the SimulationFeatureVideoCallback class
        tmp_path = pathlib.Path(self.temp_dir)
        simulation_directory = tmp_path / "simulation"
        videos_output_dir = "videos"
        feature_log_dir = "features"
        visualized_scenario_tokens = ["test_scenario"]
        visualize_all_scenarios = True
        image_subfix = ".png"
        self.callback = SimulationFeatureVideoCallback(
            simulation_directory=simulation_directory,
            videos_output_dir=videos_output_dir,
            feature_log_dir=feature_log_dir,
            visualized_scenario_tokens=visualized_scenario_tokens,
            visualize_all_scenarios=visualize_all_scenarios,
            image_subfix=image_subfix)
        self.setup = MockSimulationSetup()
        self.planner = MockPlanner()
        self.callback._visualize_all_scenarios = True

    def test_simulation_feature_video_callback(self):
        # import pdb;pdb.set_trace()
        self.callback.on_initialization_start(self.setup, self.planner)
        self.fake_and_save_images(
            self.temp_dir +
            "/simulation/features/mock_planner/test_scenario/test_scenario/test_scenario/features",
            ".png", 10)
        self.callback.on_simulation_end(self.setup, self.planner, None)
        self.assertTrue(
            pathlib.Path(
                self.temp_dir +
                "/simulation/videos/feature_selected_scenarios/test_scenario.webm"
            ).exists())
        self.remove_temp_dir()


if __name__ == '__main__':
    unittest.main()
