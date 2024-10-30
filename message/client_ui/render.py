   
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.collections import PolyCollection

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display of plots

# Define the maximum number
max_num = 20
# Create a colormap and normalize it
colormap = plt.cm.viridis  # Using 'viridis' colormap for a range of distinguishable colors
norm = plt.Normalize(vmin=0, vmax=max_num - 1)

# Generate a mapping dictionary from 0 to max_num-1, each mapped to an RGB color
mapping = {i: colormap(norm(i))[:3] for i in range(max_num)}
# Convert the mapping dictionary to a tensor for easy indexing in PyTorch
rgb_values = torch.tensor([mapping[i] for i in range(max_num)])  # Shape: (16, 3)


class Renderer:
    def __init__(self, n_world=2, show_n_world=2, range=[80, 80]):
        """
        Initializes the Renderer object with the number of worlds, number of worlds to show, and range.

        :param n_world: Number of worlds in the simulation
        :param show_n_world: Number of worlds to show in the visualization
        :param range: Tuple containing the range of the visualization (x, y)
        """
        self.n_world = n_world
        self.show_n_world = show_n_world
        self.range = range

    def _render_map(self, map_x, map_y, map_point_type):
        """
        Renders a map based on map points without displaying axes.

        :param map_x: Tensor containing the x-coordinates of map points
        :param map_y: Tensor containing the y-coordinates of map points
        :param map_point_type: Tensor indicating the type of each map point
        :return: Image tensor of the rendered map with shape (3, H, W)
        """
        # 1. Gather RGB values based on map point types
        rgb_tensor = rgb_values[map_point_type.astype(np.int), :]

        # 2. Prepare map data for plotting
        x_filtered, y_filtered, rgb_filtered = self._prepare_map_data(map_x, map_y, rgb_tensor)
        center_x, center_y = self.center_x, self.center_y
        range_x, range_y = self.range

        # 3. Initialize the plot
        plt.figure(figsize=(10, 10))
        # 4. Scatter plot of map points
        plt.scatter(
            x_filtered,
            y_filtered,
            c=rgb_filtered,
            s=1
        )
        plt.axis("equal")
        plt.xlim([center_x - range_x, center_x + range_x])
        plt.ylim([center_y - range_y, center_y + range_y])
        # 5. Remove axes, ticks, and labels for a pure image
        plt.axis('off')

        # 6. Adjust layout to minimize padding
        plt.tight_layout(pad=0)

        # 7. Save the figure to a buffer in PNG format

        buf = io.BytesIO()

        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)

        buf.seek(0)
        plt.close()

        # 8. Convert the buffer to an image tensor
        image = Image.open(buf).convert('RGB')
        map_array = np.array(image)
        return map_array
    
    def _render_agent(self, pred_traj, is_av_mask=None):
        """
        Renders agents as rectangles based on their predicted trajectories.
        Agents are divided into two groups and rendered differently.
        The plot limits are forcefully set to the fixed range, regardless of the data.

        Args:
            pred_traj (np.ndarray): [N, 6], columns are x, y, heading, w, l, h
            is_av_mask (np.ndarray, optional): Boolean array indicating which agents are AVs. Defaults to None.
        """
        # Use the fixed center_x and center_y
        center_x, center_y = self.center_x, self.center_y
        range_x, range_y = self.range

        # Split pred_traj into two groups (e.g., group 1 and group 2)
        N = pred_traj.shape[0]
        group3 = pred_traj[is_av_mask]
        pred_traj = pred_traj[~is_av_mask]
        half_N = N // 2
        group1 = pred_traj[:half_N]
        group2 = pred_traj[half_N:]
        # Compute agent polygons for both groups
        group1_polygons = self.compute_agent_polygons(group1)
        group2_polygons = self.compute_agent_polygons(group2)
        group3_polygons = self.compute_agent_polygons(group3)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Force the plot to display the full range, regardless of the data
        ax.set_xlim([center_x - range_x, center_x + range_x])
        ax.set_ylim([center_y - range_y, center_y + range_y])
        ax.set_aspect('equal')  # Ensure equal scaling on both axes
        ax.autoscale(False)     # Disable autoscaling

        # Plot all agents, even those outside the plotting range

        # Plot group 1 agents in red
        if group1_polygons:
            collection1 = PolyCollection(
                group1_polygons, facecolors='red', edgecolors='red', alpha=0.5
            )
            ax.add_collection(collection1)

        # Plot group 2 agents in green
        if group2_polygons:
            collection2 = PolyCollection(
                group2_polygons, facecolors='green', edgecolors='green', alpha=0.5
            )
            ax.add_collection(collection2)
        
        # Plot group 3 agents in blue
        if group3_polygons:
            collection3 = PolyCollection(
                group3_polygons, facecolors='blue', edgecolors='blue', alpha=0.5
            )
            ax.add_collection(collection3)

        # Remove axes for a clean image
        ax.axis('off')

        # Save the figure to a buffer in PNG format without adjusting the bounding box
        buf = io.BytesIO()
        plt.savefig(buf, format='png', pad_inches=0)
        buf.seek(0)
        plt.close(fig)

        # Convert the buffer to an image array
        image = Image.open(buf).convert('RGB')
        agent_array = np.array(image)
        buf.close()
        return agent_array

    
    def compute_agent_polygons(self, agent_data):
        """
        Computes the corner points of agents for visualization.

        Args:
            agent_data (np.ndarray): Array of shape [N, 6], columns are x, y, heading, w, l, h

        Returns:
            List of arrays: Each array contains the corner points of an agent's rectangle.
        """
        x = agent_data[:, 0]
        y = agent_data[:, 1]
        heading = agent_data[:, 2] + np.pi / 2
        w = agent_data[:, 3]
        l = agent_data[:, 4]
        N = len(x)
        agent_polygons = []

        for i in range(N):
            xi = x[i]
            yi = y[i]
            hi = heading[i]
            wi = w[i]
            li = l[i]

            # Define rectangle corners in the local frame
            local_corners = np.array([
                [-li / 2, -wi / 2],
                [ li / 2, -wi / 2],
                [ li / 2,  wi / 2],
                [-li / 2,  wi / 2]
            ])

            # Create rotation matrix
            cos_h = np.cos(hi)
            sin_h = np.sin(hi)
            R = np.array([
                [cos_h, -sin_h],
                [sin_h,  cos_h]
            ])

            # Rotate and translate corners to world frame
            world_corners = (R @ local_corners.T).T + np.array([xi, yi])

            agent_polygons.append(world_corners)

        return agent_polygons
    
    def _compute_agent_corners(self, agent_tensor):
        """
        Computes the corner points of agents for visualization.

        Args:
            agent_tensor (torch.Tensor): Tensor of shape [N, T, D], D includes x, y, heading, w, l, h

        Returns:
            List of arrays: Each array contains the corner points of an agent's rectangle.
        """
        N, T, D = agent_tensor.shape
        x = agent_tensor[:, :, 0].reshape(-1)
        y = agent_tensor[:, :, 1].reshape(-1)
        heading = agent_tensor[:, :, 2].reshape(-1)
        w = agent_tensor[:, :, 3].reshape(-1)
        l = agent_tensor[:, :, 4].reshape(-1)

        agent_polygons = []

        for xi, yi, hi, wi, li in zip(x, y, heading, w, l):
            # Define rectangle corners in the local frame
            local_corners = np.array([
                [ li / 2,  wi / 2],
                [ li / 2, -wi / 2],
                [-li / 2, -wi / 2],
                [-li / 2,  wi / 2]
            ])

            # Create rotation matrix
            cos_h = np.cos(hi.item())
            sin_h = np.sin(hi.item())
            R = np.array([
                [cos_h, -sin_h],
                [sin_h,  cos_h]
            ])

            # Rotate and translate corners to world frame
            world_corners = (R @ local_corners.T).T + np.array([xi.item(), yi.item()])

            agent_polygons.append(world_corners)

        return agent_polygons


    def _prepare_map_data(self, map_x, map_y, rgb_tensor):
        """Prepare and filter map data based on agent positions."""
        # Compute center position
        self.center_x = np.mean(map_x).item()
        self.center_y = np.mean(map_y).item()
        center_x, center_y = self.center_x, self.center_y
        range_x, range_y = self.range

        # Filter map points within the specified range
        mask = (
            (map_x >= center_x - range_x) & (map_x <= center_x + range_x) &
            (map_y >= center_y - range_y) & (map_y <= center_y + range_y)
        )
        x_filtered = map_x[mask]
        y_filtered = map_y[mask]
        rgb_filtered = rgb_tensor[mask]

        return x_filtered, y_filtered, rgb_filtered

    def _render_ego(self, x, y, heading, shape, select_act_traj, ego_hist):
        """
        Renders the ego vehicle and its selected action trajectory, then returns the image array.

        Args:
            x (float): x-coordinate of the ego vehicle in global coordinates.
            y (float): y-coordinate of the ego vehicle in global coordinates.
            heading (float): Heading angle of the ego vehicle in radians.
            shape (tuple): A tuple containing (length, width, height) of the ego vehicle.
            select_act_traj (np.ndarray): Array of shape (T, 3) containing the trajectory of the selected action, ego-centric.

        Returns:
            np.ndarray: Image array of the rendered ego vehicle with trajectory.
        """
        # Use fixed center point and range
        center_x, center_y = self.center_x, self.center_y
        range_x, range_y = self.range

        # Extract the length, width, and height of the vehicle
        l, w, h = shape

        # Adjust heading to match the coordinate system (if needed)
        hi = heading + np.pi / 2

        # Define the four corners of the vehicle in the local coordinate system
        local_corners = np.array([
            [-l / 2, -w / 2],
            [ l / 2, -w / 2],
            [ l / 2,  w / 2],
            [-l / 2,  w / 2]
        ])

        # Create rotation matrix
        cos_h = np.cos(hi)
        sin_h = np.sin(hi)
        R = np.array([
            [cos_h, -sin_h],
            [sin_h,  cos_h]
        ])

        # Rotate and translate the corners to the global coordinate system
        world_corners = (R @ local_corners.T).T + np.array([x, y])

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Add the ego vehicle polygon to the plot
        ego_polygon = plt.Polygon(world_corners, facecolor='blue', edgecolor='blue', alpha=0.7)
        ax.add_patch(ego_polygon)

        # Process and convert select_act_traj to the global coordinate system
        if select_act_traj is not None and select_act_traj.size > 0:
            # Assume the first two columns of select_act_traj are (dx, dy, ...), ignore the third column (if not needed)
            traj_ego = select_act_traj[:, :2]  # (T, 2)

            # Create rotation matrix for conversion
            cos_heading = np.cos(heading - np.pi / 2)
            sin_heading = np.sin(heading - np.pi / 2)
            R_traj = np.array([
                [cos_heading, -sin_heading],
                [sin_heading,  cos_heading]
            ])

            # Convert trajectory points to the global coordinate system
            traj_global = (R_traj @ traj_ego.T).T + np.array([x, y])  # (T, 2)

            # Plot the trajectory line
            ax.plot(traj_global[:, 0], traj_global[:, 1], color='green', linewidth=5, label='Selected Trajectory')

            # Plot trajectory points
            ax.scatter(traj_global[:, 0], traj_global[:, 1], color='red', s=20, label='Trajectory Points')

            ego_hist = np.array(ego_hist)
            ax.scatter(ego_hist[:, 0], ego_hist[:, 1], color='black', s=10, label='Ego History')

            # Add legend (optional)
            ax.legend(loc='upper right')

        # Force the plot to display the full range, regardless of the data
        ax.set_xlim([center_x - range_x, center_x + range_x])
        ax.set_ylim([center_y - range_y, center_y + range_y])
        ax.set_aspect('equal')  # Ensure equal scaling on both axes
        ax.autoscale(False)     # Disable autoscaling

        # Remove axes for a clean image
        ax.axis('off')

        # Save the plot to a buffer (PNG format)
        buf = io.BytesIO()
        # Remove bbox_inches='tight' to prevent cropping
        plt.savefig(buf, format='png', pad_inches=0)
        buf.seek(0)
        plt.close(fig)

        # Convert the buffer content to an image array
        image = Image.open(buf).convert('RGB')
        ego_array = np.array(image)
        buf.close()
        return ego_array
