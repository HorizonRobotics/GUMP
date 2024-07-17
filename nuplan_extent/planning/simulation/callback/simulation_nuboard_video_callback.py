import logging
import pathlib
import pickle
from typing import Dict, List, Optional, Union

import numpy as np
import pandas
from tqdm import tqdm
import time
import os

from bokeh.document.document import Document
from bokeh.io.export import get_screenshot_as_png
from bokeh.io.export import export_png
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, Slider, Title
from bokeh.plotting.figure import Figure
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.vehicle_parameters import \
    get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.common.maps.nuplan_map.map_factory import (NuPlanMapFactory,
                                                       get_maps_db)
from nuplan.planning.metrics.metric_engine import JSON_FILE_EXTENSION
from nuplan.planning.nuboard.base.plot_data import MapPoint, SimulationFigure
from nuplan.planning.nuboard.base.simulation_tile import \
    extract_source_from_states
from nuplan.planning.nuboard.style import (simulation_map_layer_color,
                                           simulation_tile_style)
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
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)


class SimulationNuboardVideoCallback(AbstractCallback):
    """Callback for render case at the end of the simulation."""

    def __init__(self, simulation_directory: Union[str, pathlib.Path],
                 metric_file_output_path: Union[str, pathlib.Path],
                 videos_output_dir: Union[str, pathlib.Path],
                 visualized_scenario_tokens: List,
                 visualize_all_scenarios: bool, map_root: str,
                 selected_metrics: Dict[str, float], map_version: str,
                 driver_path: str, chrome_path: str, figure_size: List,
                 radius: int) -> None:
        """
        Build A failure case callback.

        :param simulation_directory (Union[str, pathlib.Path]): Path to the simulation directory.
        :param videos_output_dir (Union[str, pathlib.Path]): Path to the directory where the output videos will be saved.
        :param visualized_scenario_tokens (List[str]): A list of scenario tokens to be visualized.
        :param visualize_all_scenarios (bool): A boolean flag indicating whether to visualize all scenarios.
        :param selected_metrics: A dictionary containing the metrics to use for selecting bad cases and their corresponding thresholds.
        :param num_bad_cases_per_type: The number of bad cases to select per metric.
        :param map_root (str): The root directory of the maps database.
        :param map_version (str): The version of the maps to use.
        :param driver_path (str): The path to the web driver.
        :param chrome_path (str): The path to the Chrome browser.
        :param figure_size (List[int]): The size of the figures to be created.
        :param radius (int): The radius to be used for various calculations.
        """
        self._doc = Document()
        self._videos_output_path = pathlib.Path(
            simulation_directory) / videos_output_dir

        self.visualized_scenario_tokens = visualized_scenario_tokens
        self.visualize_all_scenarios = visualize_all_scenarios
        self.selected_metrics = selected_metrics
        self._map_factory = NuPlanMapFactory(
            get_maps_db(map_root, map_version))
        self._vehicle_parameters = get_pacifica_parameters()

        self._driver_path = driver_path
        self._chrome_path = chrome_path
        self._firefox_path = '/usr/bin/firefox'
        self._geckodriver_path = '/mnt/nas25/yihan01.hu/miniconda3/envs/hoplan_cuda118/bin/geckodriver'
        self._figure_size = figure_size
        self._radius = radius
        self._maps: Dict[str, AbstractMap] = {}

        self.metric_file_output_path = metric_file_output_path

    def on_initialization_start(self, setup: SimulationSetup,
                                planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""

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

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner,
                       trajectory: AbstractTrajectory) -> None:
        """Inherited, see superclass."""

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Inherited, see superclass."""

    def _create_simulation_figure(
            self,
            scenario: AbstractScenario,
            planner: AbstractPlanner,
            history: SimulationHistory,
            backend: Optional[str] = "webgl") -> SimulationFigure:
        """
        Creates a simulation figure for a given scenario, planner, history and backend.

        :param scenario (AbstractScenario): The scenario to be visualized.
        :param planner (AbstractPlanner): The planner to be visualized.
        :param history (SimulationHistory): The history to be visualized.
        :param backend (Optional[str]): An optional string argument representing the output backend of the simulation figure. The default value is "webgl".
        :return (SimulationFigure): A simulation figure for a given scenario, planner, history and backend.
        """
        planner_name = planner.name
        presented_planner_name = f'"MLplanner_{scenario.scenario_name}'

        # Create a new figure instance for the simulation
        simulation_figure = Figure(
            x_range=(-self._radius, self._radius),
            y_range=(-self._radius, self._radius),
            width=self._figure_size[0],
            height=self._figure_size[1],
            title=f"{presented_planner_name}",
            tools=["pan", "wheel_zoom", "save", "reset"],
            match_aspect=True,
            active_scroll="wheel_zoom",
            margin=simulation_tile_style["figure_margins"],
            background_fill_color=simulation_tile_style["background_color"],
            output_backend=backend,
        )

        # Set some visual properties of the simulation figure
        simulation_figure.axis.visible = False
        simulation_figure.xgrid.visible = False
        simulation_figure.ygrid.visible = False
        simulation_figure.title.text_font_size = simulation_tile_style[
            "figure_title_text_font_size"]
        x_y_coordinate_title = Title(text="x [m]: , y [m]: ")
        simulation_figure.add_layout(x_y_coordinate_title, 'below')

        # Create a slider for the simulation figure
        slider = Slider(
            start=0,
            end=1,
            value=0,
            step=1,
            title="Frame",
            margin=simulation_tile_style["slider_margins"],
            css_classes=["scenario-frame-slider"],
        )

        # Create a video button for the simulation figure
        video_button = Button(
            label="Render video",
            margin=simulation_tile_style["video_button_margins"],
            css_classes=["scenario-video-button"],
        )

        # Create a new SimulationFigure instance representing the created simulation figure
        simulation_figure_data = SimulationFigure(
            figure=simulation_figure,
            file_path_index=0,
            figure_title_name=presented_planner_name,
            slider=slider,
            first_button=None,
            prev_button=None,
            play_button=None,
            next_button=None,
            last_button=None,
            video_button=video_button,
            vehicle_parameters=self._vehicle_parameters,
            planner_name=planner_name,
            scenario=scenario,
            simulation_history=history,
            x_y_coordinate_title=x_y_coordinate_title,
        )
        self._last_frame_time = time.time()
        self._current_frame_index = 0
        self._last_frame_index = 0
        self._playback_callback_handle: Optional[PeriodicCallback] = None

        return simulation_figure_data

    def _render_scenario(
            self,
            main_figure: SimulationFigure,
            hidden_glyph_names: Optional[List[str]] = None) -> None:
        """
        Render scenario.
        :param main_figure: Simulation figure object.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        """
        # Trigger background threads to fetch non-map dependent data
        main_figure.update_data_sources()

        # Load map data and then trigger thread(s) to fetch data sources that depend on it
        self._load_map_data(main_figure=main_figure)
        main_figure.update_map_dependent_data_sources()

        # Render the scenario
        self._render_map(main_figure=main_figure)

        self._render_expert_trajectory(main_figure=main_figure)

        mission_goal = main_figure.scenario.get_mission_goal()
        if mission_goal is not None:
            main_figure.render_mission_goal(mission_goal_state=mission_goal)

        self._render_plots(main_figure=main_figure, frame_index=0, hidden_glyph_names=hidden_glyph_names)

    def _map_api(self, map_name: str) -> AbstractMap:
        """
        Get a map api.
        :param map_name: Map name.
        :return Map api.
        """
        if map_name not in self._maps:
            self._maps[map_name] = self._map_factory.build_map_from_name(
                map_name)

        return self._maps[map_name]

    def _load_map_data(self, main_figure: SimulationFigure) -> None:
        """
        Load the map data of the simulation tile.
        :param main_figure: Simulation figure.
        """
        # Load map data
        map_name = main_figure.scenario.map_api.map_name
        map_api = self._map_api(map_name)
        layer_names = [
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]

        assert main_figure.simulation_history.data, "No simulation history samples, unable to render the map."
        ego_pose = main_figure.simulation_history.data[0].ego_state.center
        center = Point2D(ego_pose.x, ego_pose.y)

        self._nearest_vector_map = map_api.get_proximal_map_objects(center, self._radius, layer_names)
        # Filter out stop polygons in turn stop
        if SemanticMapLayer.STOP_LINE in self._nearest_vector_map:
            stop_polygons = self._nearest_vector_map[SemanticMapLayer.STOP_LINE]
            self._nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]

        # Populate our figure's lane connectors. This variable is required by traffic_light_plot.update_data_sources
        main_figure.lane_connectors = {
            lane_connector.id: lane_connector
            for lane_connector in self._nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]
        }

    def _render_map_polygon_layers(self, main_figure: SimulationFigure) -> None:
        """Renders the polygon layers of the map."""
        polygon_layer_names = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.LANE]),
            (SemanticMapLayer.INTERSECTION, simulation_map_layer_color[SemanticMapLayer.INTERSECTION]),
            (SemanticMapLayer.STOP_LINE, simulation_map_layer_color[SemanticMapLayer.STOP_LINE]),
            (SemanticMapLayer.CROSSWALK, simulation_map_layer_color[SemanticMapLayer.CROSSWALK]),
            (SemanticMapLayer.WALKWAYS, simulation_map_layer_color[SemanticMapLayer.WALKWAYS]),
            (SemanticMapLayer.CARPARK_AREA, simulation_map_layer_color[SemanticMapLayer.CARPARK_AREA]),
        ]
        roadblock_ids = main_figure.scenario.get_route_roadblock_ids()
        if roadblock_ids:
            polygon_layer_names.append(
                (SemanticMapLayer.ROADBLOCK, simulation_map_layer_color[SemanticMapLayer.ROADBLOCK])
            )

        for layer_name, color in polygon_layer_names:
            map_polygon = MapPoint(point_2d=[])
            # Render RoadBlock
            if layer_name == SemanticMapLayer.ROADBLOCK:
                layer = (
                    self._nearest_vector_map[SemanticMapLayer.LANE]
                    + self._nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]
                )
                for map_obj in layer:
                    roadblock_id = map_obj.get_roadblock_id()
                    if roadblock_id in roadblock_ids:
                        coords = map_obj.polygon.exterior.coords
                        points = [Point2D(x=x, y=y) for x, y in coords]
                        map_polygon.point_2d.append(points)
            else:
                layer = self._nearest_vector_map[layer_name]
                for map_obj in layer:
                    coords = map_obj.polygon.exterior.coords
                    points = [Point2D(x=x, y=y) for x, y in coords]
                    map_polygon.point_2d.append(points)

            polygon_source = ColumnDataSource(
                dict(
                    xs=map_polygon.polygon_xs,
                    ys=map_polygon.polygon_ys,
                )
            )
            layer_map_polygon_plot = main_figure.figure.multi_polygons(
                xs="xs",
                ys="ys",
                fill_color=color["fill_color"],
                fill_alpha=color["fill_color_alpha"],
                line_color=color["line_color"],
                source=polygon_source,
            )
            # underlay = default rendering level for grids, one level below `glyph`, the default level for plots
            # https://docs.bokeh.org/en/latest/docs/user_guide/styling/plots.html#setting-render-levels
            layer_map_polygon_plot.level = "underlay"
            main_figure.map_polygon_plots[layer_name.name] = layer_map_polygon_plot

    def _render_map_line_layers(self, main_figure: SimulationFigure) -> None:
        """Renders the line layers of the map."""
        line_layer_names = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.BASELINE_PATHS]),
            (SemanticMapLayer.LANE_CONNECTOR, simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR]),
        ]
        for layer_name, color in line_layer_names:
            layer = self._nearest_vector_map[layer_name]
            map_line = MapPoint(point_2d=[])
            for map_obj in layer:
                path = map_obj.baseline_path.discrete_path
                points = [Point2D(x=pose.x, y=pose.y) for pose in path]
                map_line.point_2d.append(points)

            line_source = ColumnDataSource(dict(xs=map_line.line_xs, ys=map_line.line_ys))
            layer_map_line_plot = main_figure.figure.multi_line(
                xs="xs",
                ys="ys",
                line_color=color["line_color"],
                line_alpha=color["line_color_alpha"],
                line_width=0.5,
                line_dash="dashed",
                source=line_source,
            )
            # underlay = default rendering level for grids, one level below `glyph`, the default level for plots
            # https://docs.bokeh.org/en/latest/docs/user_guide/styling/plots.html#setting-render-levels
            layer_map_line_plot.level = "underlay"
            main_figure.map_line_plots[layer_name.name] = layer_map_line_plot

    def _render_map(self, main_figure: SimulationFigure) -> None:
        """
        Render a map.
        :param main_figure: Simulation figure.
        """

        def render() -> None:
            """Wrapper for the actual render logic, for multi-threading compatibility."""
            self._render_map_polygon_layers(main_figure)
            self._render_map_line_layers(main_figure)

        self._doc.add_next_tick_callback(lambda: render())

    @staticmethod
    def _render_expert_trajectory(main_figure: SimulationFigure) -> None:
        """
        Render expert trajectory.
        :param main_figure: Main simulation figure.
        """
        expert_ego_trajectory = main_figure.scenario.get_expert_ego_trajectory(
        )
        source = extract_source_from_states(expert_ego_trajectory)
        main_figure.render_expert_trajectory(
            expert_ego_trajectory_state=source)

    def _render_plots(self,
                      main_figure: SimulationFigure,
                      frame_index: int,
                      hidden_glyph_names: Optional[List[str]] = None) -> None:
        """
        Render plot with a frame index.
        :param main_figure: Main figure to render.
        :param frame_index: A frame index.
        :param hidden_glyph_names: A list of glyph names to be hidden.
        """
        # main_figure.lane_connectors might still be loading the first time the function is called (for the initial
        # frame), but if the user renders another frame and go back, it should be there if it's available for the tile.
        # if main_figure.lane_connectors is not None and len(main_figure.lane_connectors):
        #     main_figure.traffic_light_plot.update_plot(
        #         main_figure=main_figure.figure,
        #         frame_index=frame_index,
        #         doc=self._doc,
        #     )

        # Update ego state plot.
        main_figure.ego_state_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            radius=self._radius,
            doc=self._doc,
        )

        # Update ego pose trajectory state plot.
        main_figure.ego_state_trajectory_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            doc=self._doc,
        )

        # Update agent state plot.
        main_figure.agent_state_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            doc=self._doc,
        )

        # Update agent heading plot.
        main_figure.agent_state_heading_plot.update_plot(
            main_figure=main_figure.figure,
            frame_index=frame_index,
            doc=self._doc,
        )

        def update_decorations() -> None:
            main_figure.figure.title.text = main_figure.figure_title_name_with_timestamp(frame_index=frame_index)
            main_figure.update_glyphs_visibility(glyph_names=hidden_glyph_names)

        self._doc.add_next_tick_callback(lambda: update_decorations())

        self._last_frame_index = self._current_frame_index
        self._current_frame_index = frame_index


    def render_case(self, setup: SimulationSetup, planner: AbstractPlanner,
                    history: SimulationHistory, video_save_path: str) -> None:
        """
        Render a failure case.

        :param setup: the simulation setup object containing the scenario.
        :param planner: the planner object used for the simulation.
        :param history: the simulation history object containing the simulation data.
        :param video_save_path: Path to save the video.
        :return: None.
        """
        scenario = setup.scenario
        scenario_name = scenario.scenario_name
        database_interval = scenario.database_interval
        logger.info("Rendering failure case for scenario %s", scenario_name)

        # Create simulation figure
        simulation_figure = self._create_simulation_figure(scenario, planner, history)
        self._render_scenario(simulation_figure)

        length = len(history)

        images = []
        # Render the video frames
        for frame_index in tqdm(range(length), desc="Rendering video"):
            self._render_plots(main_figure=simulation_figure, frame_index=frame_index)
            
            # Get the screenshot as PNG
            image = get_screenshot_as_png(simulation_figure.figure)
            
            # Convert to numpy array and save to list
            image_array = np.array(image)
            images.append(image_array)
            
            # Save the image temporarily to use it later
            temp_image_path = f"temp_frame_{frame_index}.png"
            image.save(temp_image_path)

        # Save video and cleanup
        # if images:
        #     shape = images[0].shape
        #     save_video(shape, images, video_save_path, database_interval)

        # Clean up temporary files
        for frame_index in range(length):
            os.remove(f"temp_frame_{frame_index}.png")

        return images

    def _render_case_on_metric_failure(self, setup: SimulationSetup,
                                       planner: AbstractPlanner,
                                       history: SimulationHistory) -> None:
        """
        Renders simulation videos for selected scenarios based on selected metrics and their thresholds.

        For each metric file and threshold in `selected_metric_file_paths` and `selected_metric_thresholds` respectively,
        this method loads the data frame from the metric file, selects bad cases based on the metric and threshold, and
        renders simulation tiles for the bad cases.
        :param setup: the simulation setup object containing the scenario.
        :param planner: the planner object used for the simulation.
        :param history: the simulation history object containing the simulation data.
        """
        scenario = setup.scenario
        file_name = scenario.scenario_type + '_' + scenario.scenario_name + '_' + planner.name(
        )
        file_name = file_name + JSON_FILE_EXTENSION
        scenario_metric_file = pathlib.Path(
            self.metric_file_output_path) / file_name
        if not scenario_metric_file.exists():
            logger.info(f"Metric file {scenario_metric_file} does not exist.")
            return

        with open(scenario_metric_file, "rb") as f:
            json_dataframe = pickle.load(f)
        pandas_dataframe = pandas.DataFrame(json_dataframe)
        for metric_name in self.selected_metrics:
            metric_threshold = self.selected_metrics[metric_name]
            if metric_name not in pandas_dataframe[
                    'metric_statistics_name'].to_dict().values():
                continue
            metric_score = pandas_dataframe[
                pandas_dataframe['metric_statistics_name'] ==
                metric_name]['metric_score'].to_numpy()[0]
            if metric_score <= metric_threshold:
                video_save_dir = pathlib.Path(
                    self._videos_output_path / "failure_cases" / metric_name)
                if not video_save_dir.exists():
                    video_save_dir.mkdir(parents=True, exist_ok=True)
                video_name = scenario.token + ".webm"
                self.render_case(setup, planner, history,
                                 video_save_dir / video_name)

    def on_simulation_end(self, setup: SimulationSetup,
                          planner: AbstractPlanner,
                          history: SimulationHistory) -> None:
        """Inherited, see superclass."""
        scenario_token = setup.scenario.token
        images = None
        if self.visualize_all_scenarios or scenario_token in self.visualized_scenario_tokens:
            # Set the video save path
            video_save_dir = pathlib.Path(
                self._videos_output_path / "selected_scenarios")
            video_name = scenario_token + ".webm"
            if not video_save_dir.exists():
                video_save_dir.mkdir(parents=True, exist_ok=True)

            images = self.render_case(setup, planner, history,
                             video_save_dir / video_name)

        # for failure case rendering
        # self._render_case_on_metric_failure(setup, planner, history)

        return images
