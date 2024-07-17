import glob

import os, sys
from typing import Callable, List, Tuple, Optional, Dict, Generator, Union
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import pickle
from nuplan_extent.planning.scenario_builder.wod_db.wod_scenario import WodScenario
from nuplan_extent.planning.training.preprocessing.features.raster_utils import _polygon_to_coords, _linestring_to_coords, _cartesian_to_projective_coords
from nuplan.common.maps.maps_datatypes import (RasterLayer, RasterMap,
                                               SemanticMapLayer)
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.agent_state import AgentState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap

all_preds = glob.glob(
    '/mnt/nas26/yihan01.hu/test_wod/tmp_pkl_v2/*_rollouts.pkl'
)


def _get_layer_coords(
    ego_state: Union[AgentState, EgoState],
    map_api: AbstractMap,
    map_layer_name: SemanticMapLayer,
    map_layer_geometry: str,
    radius: float,
    longitudinal_offset: float = 0.0,
) -> Tuple[List[npt.NDArray[np.float64]], List[str]]:
    """
    Construct the map layer of the raster by converting vector map to raster map, based on the focus agent.
    :param ego_state (AgentState, EgoState): current state of focus agent.
    :param map_api: map api
    :param map_layer_name: name of the vector map layer to create a raster from.
    :param map_layer_geometry: geometric primitive of the vector map layer. i.e. either polygon or linestring.
    :param radius: [m] the radius of the square raster map.
    :param longitudinal_offset: [-0.5, 0.5] longitudinal offset of ego center
    :return
        object_coords: the list of 2d coordinates which represent the shape of the map.
        lane_ids: the list of ids for the map objects.
    """
    virtual_center = ego_state.center
    ego_position = Point2D(virtual_center.x, virtual_center.y)
    nearest_vector_map = map_api.get_proximal_map_objects(
        layers=[map_layer_name],
        point=ego_position,
        radius=radius,
    )
    geometry = nearest_vector_map[map_layer_name]

    if len(geometry):
        global_transform = np.linalg.inv(virtual_center.as_matrix())

        # By default the map is right-oriented, this makes it top-oriented.
        map_align_transform = R.from_euler(
            'z', 0, degrees=True).as_matrix().astype(np.float32)
        transform = map_align_transform @ global_transform

        if map_layer_geometry == 'polygon':
            _object_coords = _polygon_to_coords(geometry)
        elif map_layer_geometry == 'linestring':
            _object_coords = _linestring_to_coords(geometry)
        else:
            raise RuntimeError(
                f'Layer geometry {map_layer_geometry} type not supported')

        object_coords: List[npt.NDArray[np.float64]] = [
            np.vstack(coords).T for coords in _object_coords
        ]
        object_coords = [
            (transform @ _cartesian_to_projective_coords(coords).T).T[:, :2]
            for coords in object_coords
        ]

        lane_ids = [lane.id for lane in geometry]
    else:
        object_coords = []
        lane_ids = []

    return object_coords, lane_ids


def rotation_matrix_z(angle_rad):
    """Generate a rotation matrix for a given angle in radians around the z-axis."""
    r = R.from_euler('z', angle_rad)
    return r.as_matrix()


def draw_box(ax, center, length, width, height, color, heading, zorder,
             world_i):
    # Define the rotation matrix
    rotation_mat = rotation_matrix_z(heading)

    # Define the vertices of the 3D box
    x = length / 2
    y = width / 2
    z = height / 2
    vertices = np.array([[x, y, -z], [-x, y, -z], [-x, -y, -z], [x, -y, -z],
                         [x, y, z], [-x, y, z], [-x, -y, z], [x, -y, z]])

    # Rotate and translate the vertices
    vertices = np.dot(vertices, rotation_mat.T) + center

    # Create the sides of the box
    faces = [[vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]],
             [vertices[j] for j in [0, 3, 7, 4]],
             [vertices[j] for j in [2, 1, 5, 6]],
             [vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [2, 3, 7, 6]]]
    color = [(1 - world_i * 0.3) * c for c in color]
    front_color = [(1 - world_i * 0.3) * c for c in [0.1, 0.8, 0.3]]
    colors = [color, color, front_color, color, color, color]
    # Add the sides to the plot with light grey edge color
    box = Poly3DCollection(faces,
                           facecolors=colors,
                           linewidths=1,
                           edgecolors='lightgrey',
                           alpha=1)
    box.set_zorder(zorder)  # Set a high zorder value
    # print(box.zorder)
    ax.add_collection3d(box)
    zorder += 1
    return zorder


# Function to draw roads as lines
def draw_road(ax, path, width, color, zorder):
    # Create the road edges
    for offset in [width / 2, -width / 2]:
        edge_path = path + offset * np.array([0, 1, 0
                                              ])  # Offset in the y direction
        line = Line3DCollection([edge_path], colors=color, linewidths=1)
        line.set_zorder(zorder)  # Set a low zorder value
        # print(line.zorder)
        # ax.add_collection3d(line)
        ax.plot(edge_path[:, 0],
                edge_path[:, 1],
                edge_path[:, 2],
                color=color,
                linewidth=1,
                zorder=zorder)
        zorder += 1
    return zorder


def draw_3d_points(ax, points, color='blue', size=10):
    """
    Draw 3D points on the given axes.
    
    Parameters:
    - ax: The axes object to draw the points on.
    - points: A numpy array of points with shape (N, 3).
    - color: The color of the points.
    - size: The size of the points.
    """
    # Unpack points into x, y, and z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Use scatter method to draw the points
    ax.scatter(x, y, z, c=color, s=size)


# Define the function to draw a large plane at z=0
def draw_plane(ax, z=0, size=1000, color='lightgrey', alpha=0.5):
    """
    Draw a large plane at z=0.

    Parameters:
    - ax: The axes object to draw the plane on.
    - z: The z coordinate of the plane.
    - size: The length and width of the plane.
    - color: The color of the plane.
    - alpha: The transparency of the plane.
    """
    # Create a meshgrid for the plane
    x = np.linspace(-size / 2, size / 2, 2)
    y = np.linspace(-size / 2, size / 2, 2)
    x, y = np.meshgrid(x, y)
    z = np.full_like(x, z)

    # Plot the surface
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def get_color_for_id(object_id):
    # if object_id != 2:
    #     return [0, 0.5, 1]
    np.random.seed(object_id)  # Ensure repeatability for the same ID
    return np.random.rand(3)


def update(frame_idx, world_number=1, bev_view=False):
    ax.clear()
    zorder = 1

    # for baseline in baselines:
    #     baseline = np.concatenate([baseline, np.zeros((len(baseline), 1))], axis=1)
    #     # draw_road(ax, baseline, 0.1, color='green')
    #     draw_3d_points(ax, baseline, color='green', size=0.1)
    draw_plane(ax, z=-0.5, size=250, color='lightgrey', alpha=0.15)

    # Your existing code, modified to use frame_idx
    for road_line in road_lines:
        road_line = np.concatenate(
            [road_line, np.zeros((len(road_line), 1))], axis=1)
        draw_road(ax,
                  road_line,
                  0.1,
                  color=[112 / 255, 128 / 255, 144 / 255],
                  zorder=zorder)

    for road_edge in road_edges:
        road_edge = np.concatenate(
            [road_edge, np.zeros((len(road_edge), 1))], axis=1)
        draw_road(ax,
                  road_edge,
                  0.1,
                  color=[160 / 255, 68 / 255, 51 / 255],
                  zorder=zorder)

    for world_i in range(world_number):
        for idx, object_states in enumerate(all_lf_values[world_i]):
            tem_states = object_states[frame_idx]
            x, y, z = tem_states[:3]
            heading, length, width = tem_states[3:6]
            height = 0.6
            color = get_color_for_id(idx)
            draw_box(ax, [x, y, z], width, length, height, color, heading,
                     zorder, world_i)
            zorder += 1

    # Set the axes limits
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-5, 15])
    ax.axis('off')
    if bev_view:
        ax.view_init(elev=90, azim=0)


for i in range(len(all_preds)):
    pred_file = all_preds[i]

    with open(pred_file, 'rb') as f:
        sim_agents_rollouts = pickle.load(f)
        all_lf_values = sim_agents_rollouts['values']
        all_lf_valid = sim_agents_rollouts['valid']
        waymo_scenario_paths = sim_agents_rollouts['waymo_scenario_paths']
        local_to_global_transforms = sim_agents_rollouts['transform']

    scenario_id, target_idx, _ = pred_file.split('/')[-1].split('_')

    scenario = WodScenario("/mnt/nas20/zhening.yang/wod/scenario_pkl_v1_2/",
                           "validation", scenario_id, target_idx)

    focus_agent = scenario.get_ego_state_at_iteration(10)
    map_api = scenario.map_api

    radius = 50

    road_lines, _ = _get_layer_coords(
        focus_agent, map_api, SemanticMapLayer.BASELINE_PATHS, 'linestring',
        [-2 * radius, 2 * radius, -2 * radius, 2 * radius])
    baselines, _ = _get_layer_coords(
        focus_agent, map_api, SemanticMapLayer.LANE, 'linestring',
        [-2 * radius, 2 * radius, -2 * radius, 2 * radius])
    road_edges, _ = _get_layer_coords(
        focus_agent, map_api, SemanticMapLayer.EXTENDED_PUDO, 'linestring',
        [-2 * radius, 2 * radius, -2 * radius, 2 * radius])

    # Initialize the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create the Animation Object
    ani = animation.FuncAnimation(fig, update, frames=17, blit=False)

    # Save the Animation
    output_file = 'test_scripts/draw/multi_world/{}_0_3d_animation.mp4'.format(
        scenario_id)
    ani.save(output_file, writer='ffmpeg', fps=2)
    print('Saved animation to', output_file)
