import random
from typing import Any, List, Optional, Dict

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from copy import deepcopy

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.features.raster_utils import get_ego_raster
from nuplan_extent.planning.training.callbacks.utils.visualization_utils import draw_bev_bboxes, map_to_rgb, draw_bev_trajectory, draw_velocity
import nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils as kutils
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
import io
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
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

class VisualizationSMARTCallback(pl.Callback):
    """
    Callback that visualizes planner model inputs/outputs and logs them in Tensorboard.
    """

    def __init__(
        self,
        images_per_tile: int,
        num_train_tiles: int,
        num_val_tiles: int,
        pixel_size: float,
        canvas_size: int,
        raster_type: Dict[str, List[float]],
        dataset: str = 'nuplan'
    ):
        """
        Initialize the class.

        :param images_per_tile: number of images per tiles to visualize
        :param num_train_tiles: number of tiles from the training set
        :param num_val_tiles: number of tiles from the validation set
        :param pixel_size: [m] size of pixel in meters
        :param num_frames: number of frames to visualize
        """
        super().__init__()

        self.custom_batch_size = images_per_tile
        self.num_train_images = num_train_tiles * images_per_tile
        self.num_val_images = num_val_tiles * images_per_tile
        self.pixel_size = pixel_size
        self.canvas_size = canvas_size
        self.raster_type = raster_type
        self.dataset = dataset
        self.show_n_world = 4
        self.range = [80, 80]

        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None

    def _initialize_dataloaders(self,
                                datamodule: pl.LightningDataModule) -> None:
        """
        Initialize the dataloaders. This makes sure that the same examples are sampled
        every time for comparison during visualization.

        :param datamodule: lightning datamodule
        """
        train_set = datamodule.train_dataloader().dataset
        val_set = datamodule.val_dataloader().dataset

        self.train_dataloader = self._create_dataloader(
            train_set, self.num_train_images)
        self.val_dataloader = self._create_dataloader(val_set,
                                                      self.num_val_images)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset,
                           num_samples: int) -> torch.utils.data.DataLoader:
        dataset_size = len(dataset)
        num_keep = min(dataset_size, num_samples)
        sampled_idxs = random.sample(range(dataset_size), num_keep)
        subset = torch.utils.data.Subset(dataset=dataset, indices=sampled_idxs)
        return torch.utils.data.DataLoader(dataset=subset,
                                           batch_size=self.custom_batch_size,
                                           collate_fn=FeatureCollate())

    def _log_from_dataloader(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        loggers: List[Any],
        training_step: int,
        prefix: str,
    ) -> None:
        """
        Visualizes and logs all examples from the input dataloader.

        :param pl_module: lightning module used for inference
        :param dataloader: torch dataloader
        :param loggers: list of loggers from the trainer
        :param training_step: global step in training
        :param prefix: prefix to add to the log tag
        """
        for batch_idx, batch in enumerate(dataloader):
            features: FeaturesType = batch[0]
            targets: TargetsType = batch[1]
            predictions = self._infer_model(pl_module,
                                            move_features_type_to_device(features, pl_module.device))

            self._log_batch(loggers, features, targets, predictions, batch_idx,
                            training_step, prefix)

    def _log_batch(
        self,
        loggers: List[Any],
        features: FeaturesType,
        targets: TargetsType,
        predictions: TargetsType,
        batch_idx: int,
        training_step: int,
        prefix: str,
    ) -> None:
        """
        Visualizes and logs a batch of data (features, targets, predictions) from the model.

        :param loggers: list of loggers from the trainer
        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :param batch_idx: index of total batches to visualize
        :param training_step: global trainign step
        :param prefix: prefix to add to the log tag
        """
        self.n_world = len(predictions['scenario_id']) 
        n_batch = len(predictions['scenario_id'][0])
        batched_image_tensor = []
        batched_video_merge_tensor = []
        batched_video_sep_tensor = []
        for bi in range(n_batch):
            scenario_id = [predictions['scenario_id'][i][bi] for i in range(self.n_world)]
            map_batch_mask = predictions['map_batch_index'] == bi
            agent_batch_mask = torch.stack([batch_index % n_batch == bi for batch_index in predictions['batch_index']])
            
            is_av_mask = []
            for i in range(len(predictions['is_av_mask'])):
                if i % n_batch == bi:
                    is_av_mask.append(predictions['is_av_mask'][i])
            map_x = predictions['map_point_pos'][map_batch_mask, 0]
            map_y = predictions['map_point_pos'][map_batch_mask, 1]
            map_point_type = predictions['map_point_type'][map_batch_mask] # unit8
            pred_agent_shape = predictions['pred_agent_shape'][agent_batch_mask]
            pred_traj = predictions['pred_traj'][agent_batch_mask]
            pred_head = predictions['pred_head'][agent_batch_mask]

            agent_tensor_total = torch.cat([pred_traj, pred_head[..., None], pred_agent_shape[:, 11:, :]], dim=-1)
            image_tensor = self._render_image(map_x, map_y, map_point_type, agent_tensor_total, is_av_mask)
            video_merge_tensor = self._render_video(map_x, map_y, map_point_type, deepcopy(agent_tensor_total), self.show_n_world, 0, is_av_mask)
            video_sep_tensor = [self._render_video(map_x, map_y, map_point_type, deepcopy(agent_tensor_total), 1, i, is_av_mask) for i in range(4)]
            video_sep_tensor_stack = []
            num_time_steps = len(video_sep_tensor[0])
            for t in range(num_time_steps):
                img0 = video_sep_tensor[0][t] 
                img1 = video_sep_tensor[1][t]
                img2 = video_sep_tensor[2][t]
                img3 = video_sep_tensor[3][t]
                
                top = np.hstack((img0, img1))      
                bottom = np.hstack((img2, img3))  
                
                full = np.vstack((top, bottom))     
                video_sep_tensor_stack.append(full)
                                
            video_merge_tensor = self._convert_to_video_torch_tensor(video_merge_tensor)
            video_sep_tensor_stack = self._convert_to_video_torch_tensor(video_sep_tensor_stack)
            batched_image_tensor.append(image_tensor)
            batched_video_merge_tensor.append(video_merge_tensor)
            batched_video_sep_tensor.append(video_sep_tensor_stack)
        batched_image_tensor = torch.stack(batched_image_tensor, dim=0) # B, C, H, W
        batched_video_merge_tensor = torch.cat(batched_video_merge_tensor, dim=0) # B, T, C, H, W
        batched_video_sep_tensor = torch.cat(batched_video_sep_tensor, dim=0) # B, T, C, H, W
        fps = 10
        logger = loggers
        logger.add_images(
            f'{prefix}_image',
            batched_image_tensor,
            global_step=training_step)        
        logger.add_video(
            f'{prefix}_video_merge',
            batched_video_merge_tensor,
            global_step=training_step,
            fps=fps,
        )
        logger.add_video(
            f'{prefix}_video_sep',
            batched_video_sep_tensor,
            global_step=training_step,
            fps=fps,
        )
            

    def _convert_to_video_torch_tensor(self, frames):
        # Convert frames list to numpy array
        video_frames = np.stack(frames, axis=0)  # Shape (T, H, W, 3)

        # Convert to tensor and permute dimensions to (T, C, H, W)
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2)  # Shape (T, C, H, W)

        # Add batch dimension to make it (N, T, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Shape (1, T, C, H, W)

        # Ensure the tensor is of type torch.uint8
        video_tensor = video_tensor.to(torch.uint8)
        return video_tensor

    def _generate_agent_colors(self, N):
        """
        Generate a color map for different agents using a dark color scheme.

        :param N: Number of agents
        :return: Array of RGBA colors with shape (N, 4)
        """
        # Choose a dark colormap. Options include 'Dark2', 'tab20', etc.
        colormap = cm.get_cmap('Dark2', N)  # 'Dark2' can handle up to 8 distinct colors
        
        # If N exceeds the colormap's capacity, you might consider 'tab20'
        if N > colormap.N:
            colormap = cm.get_cmap('tab20', N)
        
        # Generate colors by sampling the colormap
        colors = colormap(np.linspace(0, 1, N))  # Shape: (N, 4)
        
        # Optional: Adjust alpha if needed (e.g., make colors semi-transparent)
        # alpha = 0.8
        # colors[:, -1] = alpha
        
        return colors

    def _prepare_map_data(self, map_x, map_y, rgb_tensor, agent_tensor):
        """Prepare and filter map data based on agent positions."""
        # Compute center position
        center_x = torch.mean(agent_tensor[:, :, 0]).item()
        center_y = torch.mean(agent_tensor[:, :, 1]).item()
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

    def _compute_agent_corners(self, agent_tensor):
        """Compute the world coordinates of agent bounding box corners."""
        # Convert agent_tensor to numpy array
        agent_tensor_np = agent_tensor.cpu().numpy()
        N, T, _ = agent_tensor.shape

        # Extract agent attributes and flatten over time
        x = agent_tensor_np[:, :, 0].reshape(-1)
        y = agent_tensor_np[:, :, 1].reshape(-1)
        heading = agent_tensor_np[:, :, 2].reshape(-1)
        length = agent_tensor_np[:, :, 3].reshape(-1)
        width = agent_tensor_np[:, :, 4].reshape(-1)

        # Compute unit corners of the rectangle (centered at origin)
        unit_corners = np.array([
            [-0.5, -0.5],
            [-0.5,  0.5],
            [ 0.5,  0.5],
            [ 0.5, -0.5],
            [-0.5, -0.5]  # Closing the polygon
        ])  # Shape: (5, 2)

        # Scale corners by length and width
        corners = unit_corners[np.newaxis, :, :] * np.stack([length, width], axis=-1)[:, np.newaxis, :]  # Shape: (N*T, 5, 2)

        # Compute rotation components
        cos_heading = np.cos(heading)
        sin_heading = np.sin(heading)

        # Rotate corners
        corners_rotated_x = corners[:, :, 0] * cos_heading[:, np.newaxis] - corners[:, :, 1] * sin_heading[:, np.newaxis]
        corners_rotated_y = corners[:, :, 0] * sin_heading[:, np.newaxis] + corners[:, :, 1] * cos_heading[:, np.newaxis]
        corners_rotated = np.stack([corners_rotated_x, corners_rotated_y], axis=-1)

        # Translate corners to world coordinates
        corners_world_x = corners_rotated[:, :, 0] + x[:, np.newaxis]
        corners_world_y = corners_rotated[:, :, 1] + y[:, np.newaxis]
        corners_world = np.stack([corners_world_x, corners_world_y], axis=-1)

        return corners_world

    def _compute_agent_corners_at_time(self, agent_tensor, t):
        """Compute agent corners for a specific time step."""
        agent_tensor_np = agent_tensor.cpu().numpy()
        x = agent_tensor_np[:, t, 0]
        y = agent_tensor_np[:, t, 1]
        heading = agent_tensor_np[:, t, 2]
        length = agent_tensor_np[:, t, 3]
        width = agent_tensor_np[:, t, 4]

        # Compute unit corners of the rectangle (centered at origin)
        unit_corners = np.array([
            [-0.5, -0.5],
            [-0.5,  0.5],
            [ 0.5,  0.5],
            [ 0.5, -0.5],
            [-0.5, -0.5]  # Closing the polygon
        ])  # Shape: (5, 2)

        # Scale corners by length and width
        corners = unit_corners[np.newaxis, :, :] * np.stack([length, width], axis=-1)[:, np.newaxis, :]

        # Compute rotation components
        cos_heading = np.cos(heading)
        sin_heading = np.sin(heading)

        # Rotate corners
        corners_rotated_x = corners[:, :, 0] * cos_heading[:, np.newaxis] - corners[:, :, 1] * sin_heading[:, np.newaxis]
        corners_rotated_y = corners[:, :, 0] * sin_heading[:, np.newaxis] + corners[:, :, 1] * cos_heading[:, np.newaxis]
        corners_rotated = np.stack([corners_rotated_x, corners_rotated_y], axis=-1)

        # Translate corners to world coordinates
        corners_world = corners_rotated + np.stack([x, y], axis=-1)[:, np.newaxis, :]

        return corners_world

    def _render_image(self, map_x, map_y, map_point_type, agent_tensor_total, is_av_mask):
        """
        Renders an image based on map points and agent tensors without displaying axes.
        Filled polygons are rendered for agents where is_av_mask=True.

        :param map_x: Tensor containing the x-coordinates of map points
        :param map_y: Tensor containing the y-coordinates of map points
        :param map_point_type: Tensor indicating the type of each map point
        :param agent_tensor_total: Tensor containing agent-related data
        :param is_av_mask: Tensor indicating which agents are AVs (shape: [N])
        :return: Image tensor of the rendered scene with shape (3, H, W)
        """
        # 1. Gather RGB values based on map point types
        rgb_tensor = rgb_values[map_point_type.to(torch.int64), :]  # Shape: (N, 3)

        # 2. Extract agent data
        show_n_world = self.show_n_world
        total_n_world = self.n_world
        # Calculate the number of agents per world
        agents_per_world = agent_tensor_total.shape[0] // total_n_world
        selected_agents = show_n_world * agents_per_world
        # Select the relevant agents based on the number of worlds to show
        agent_tensor = agent_tensor_total[:selected_agents]  # Shape: (selected_agents, T, D)
        # Concatenate the is_av_mask for the selected worlds
        is_av_mask = torch.cat(is_av_mask[:show_n_world], dim=0)  # Shape: (selected_agents,)

        N, T, _ = agent_tensor.shape

        # 3. Generate colors for each agent (ensure RGBA)
        colors = self._generate_agent_colors(N)  # Expected Shape: (N, 4)
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        elif not isinstance(colors, np.ndarray):
            colors = np.array(colors)
        
        # Validate the shape of 'colors'
        if colors.ndim != 2 or colors.shape[1] != 4:
            raise ValueError(f"'colors' should have shape (N, 4), but got {colors.shape}")

        # 4. Prepare map data for plotting
        x_filtered, y_filtered, rgb_filtered = self._prepare_map_data(map_x, map_y, rgb_tensor, agent_tensor)

        # 5. Initialize the plot
        plt.figure(figsize=(10, 10))
        
        # 6. Scatter plot of map points
        plt.scatter(
            x_filtered.cpu().numpy(),
            y_filtered.cpu().numpy(),
            c=rgb_filtered.cpu().numpy(),
            s=1
        )

        # 7. Calculate the center of the agent positions
        center_x = torch.mean(agent_tensor[:, :, 0]).item()
        center_y = torch.mean(agent_tensor[:, :, 1]).item()
        range_x, range_y = self.range

        # 8. Set the limits of the plot based on the center and range
        plt.xlim(center_x - range_x, center_x + range_x)
        plt.ylim(center_y - range_y, center_y + range_y)
        
        # 9. Remove axes, ticks, and labels for a pure image
        plt.axis('off')

        # 10. Compute the corners of agents for visualization
        corners_world = self._compute_agent_corners(agent_tensor)  # List of polygons

        # 11. Assign colors to agents based on their indices
        agent_indices = np.repeat(np.arange(N), T)  # Shape: (N*T,)
        rect_colors = colors[agent_indices]  # Shape: (N*T, 4)

        # 12. Repeat is_av_mask for each time step to match agent_indices
        is_av_mask_repeated = np.repeat(is_av_mask.cpu().numpy(), T)  # Shape: (N*T,)

        # 13. Separate polygons and colors based on is_av_mask
        AV_indices = np.where(is_av_mask_repeated)[0]
        non_AV_indices = np.where(~is_av_mask_repeated)[0]

        AV_corners_world = [corners_world[i] for i in AV_indices]
        AV_colors = rect_colors[AV_indices]

        non_AV_corners_world = [corners_world[i] for i in non_AV_indices]
        non_AV_colors = rect_colors[non_AV_indices]

        # 14. Create PolyCollection for AVs (filled polygons)
        if len(AV_corners_world) > 0:
            # For filled polygons, set facecolors with desired alpha
            AV_facecolors = AV_colors.copy()
            AV_facecolors[:, 3] = 0.3  # Set alpha to 0.3 for semi-transparency

            AV_collection = PolyCollection(
                AV_corners_world,
                edgecolors=AV_colors[:, :3],   # RGB colors for edges
                facecolors=AV_facecolors,      # RGBA colors for fills
                linewidths=1,
                alpha=1.0                        # Overall alpha; individual alphas are handled in facecolors
            )
            plt.gca().add_collection(AV_collection)

        # 15. Create PolyCollection for non-AVs (unfilled polygons)
        if len(non_AV_corners_world) > 0:
            non_AV_collection = PolyCollection(
                non_AV_corners_world,
                edgecolors=non_AV_colors[:, :3],  # RGB colors for edges
                facecolors='none',                # No fill
                linewidths=1,
                alpha=0.5                          # Transparency for edges
            )
            plt.gca().add_collection(non_AV_collection)

        # 16. Adjust layout to minimize padding
        plt.tight_layout(pad=0)

        # 17. Save the figure to a buffer in PNG format
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()

        # 18. Convert the buffer to an image tensor
        image = Image.open(buf).convert('RGB')
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # Shape: (3, H, W)
        buf.close()

        return image_tensor

    def _render_video(
        self,
        map_x: torch.Tensor,
        map_y: torch.Tensor,
        map_point_type: torch.Tensor,
        agent_tensor_total: torch.Tensor,
        show_n_world: int,
        world_index: int,
        is_av_mask: list,
    ) -> list:
        """
        Renders a video by generating a sequence of image frames based on map points and agent tensors.
        The rendered video frames are pure images without any axes, labels, or ticks.

        :param map_x: Tensor containing the x-coordinates of map points
        :param map_y: Tensor containing the y-coordinates of map points
        :param map_point_type: Tensor indicating the type of each map point
        :param agent_tensor_total: Tensor containing agent-related data
        :param show_n_world: Number of worlds to display
        :param world_index: Index of the world to render
        :param is_av_mask: List of tensors indicating which agents are AVs for each world
        :return: List of image frames as NumPy arrays
        """
        # 1. Gather RGB values based on map point types
        rgb_tensor = rgb_values[map_point_type.to(torch.int64), :]  # Shape: (N, 3)

        # 2. Extract agent data for the specified worlds
        total_n_world = self.n_world
        agents_per_world = agent_tensor_total.shape[0] // total_n_world
        start_idx = agents_per_world * world_index
        end_idx = start_idx + agents_per_world * show_n_world
        agent_tensor = agent_tensor_total[start_idx:end_idx]  # Shape: (N, T, D)
        is_av_mask = torch.cat(is_av_mask[world_index:world_index + show_n_world], dim=0)  # Shape: (N,)

        N, T, _ = agent_tensor.shape

        # 3. Generate colors for each agent
        colors = self._generate_agent_colors(N)  # Expected Shape: (N, 4)
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        elif not isinstance(colors, np.ndarray):
            colors = np.array(colors)

        # Validate the shape of 'colors'
        if colors.ndim != 2 or colors.shape[1] != 4:
            raise ValueError(f"'colors' should have shape (N, 4), but got {colors.shape}")

        # 4. Prepare map data for plotting
        x_filtered, y_filtered, rgb_filtered = self._prepare_map_data(
            map_x, map_y, rgb_tensor, agent_tensor
        )
        x_filtered = x_filtered.cpu().numpy()
        y_filtered = y_filtered.cpu().numpy()
        rgb_filtered = rgb_filtered.cpu().numpy()

        # 5. Initialize list to store video frames
        frames = []

        # 6. Calculate the center and range for plot limits based on agent positions
        center_x = torch.mean(agent_tensor[:, :, 0]).item()
        center_y = torch.mean(agent_tensor[:, :, 1]).item()
        range_x, range_y = self.range

        # 7. Initialize the plot
        fig, ax = plt.subplots(figsize=(5, 5))

        # 8. Scatter plot of map points
        scatter = ax.scatter(
            x_filtered,
            y_filtered,
            c=rgb_filtered,
            s=1
        )

        # 9. Set plot limits
        ax.set_xlim(center_x - range_x, center_x + range_x)
        ax.set_ylim(center_y - range_y, center_y + range_y)

        # 10. Remove axes, ticks, and labels for a pure image
        ax.axis('off')

        # 11. Initialize PolyCollections for agent representations
        # Separate AVs and non-AVs
        is_av_mask_np = is_av_mask.cpu().numpy()
        AV_indices = np.where(is_av_mask_np)[0]
        non_AV_indices = np.where(~is_av_mask_np)[0]

        # Colors for AVs and non-AVs
        AV_colors = colors[AV_indices]
        non_AV_colors = colors[non_AV_indices]

        # Initialize empty PolyCollections
        AV_collection = PolyCollection(
            [],
            edgecolors=AV_colors[:, :3],
            facecolors=AV_colors.copy(),
            linewidths=1,
            alpha=1.0
        )
        ax.add_collection(AV_collection)

        non_AV_collection = PolyCollection(
            [],
            edgecolors=non_AV_colors[:, :3],
            facecolors='none',
            linewidths=1,
            alpha=0.8
        )
        ax.add_collection(non_AV_collection)

        # 12. Optimize layout to minimize padding
        plt.tight_layout(pad=0)

        # 13. Draw the initial canvas
        fig.canvas.draw()

        # 14. Iterate over each time step to generate frames
        for t in range(T):
            # Compute the corners of agents at the current time step
            corners_world = self._compute_agent_corners_at_time(agent_tensor, t)  # List of (N,) polygons

            # Separate corners for AVs and non-AVs
            AV_corners_world = [corners_world[i] for i in AV_indices]
            non_AV_corners_world = [corners_world[i] for i in non_AV_indices]

            # Update the PolyCollections with new agent positions
            AV_collection.set_verts(AV_corners_world)
            non_AV_collection.set_verts(non_AV_corners_world)

            # Redraw the canvas with updated agent positions
            fig.canvas.draw()

            # Convert the current figure canvas to a NumPy array
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(np.copy(image_array))

        # 15. Close the figure to free up memory
        plt.close(fig)

        return frames


    def _infer_model(self, pl_module: pl.LightningModule,
                     features: FeaturesType) -> TargetsType:
        """
        Make an inference of the input batch features given a model.

        :param pl_module: lightning model
        :param features: model inputs
        :return: model predictions
        """
        with torch.no_grad(), autocast(enabled=True):
            pl_module.eval()
            pl_module.float()
            predictions = move_features_type_to_device(pl_module(features),
                                                       torch.device('cpu'))
            pl_module.train()
        # predictions = convert_features_type_to_float(predictions, torch.device('cpu'))
        return predictions

    def on_train_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs training examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer,
                       'datamodule'), "Trainer missing datamodule attribute"
        assert hasattr(trainer,
                       'global_step'), "Trainer missing global_step attribute"

        if self.train_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(
            pl_module,
            self.train_dataloader,
            trainer.logger.experiment,
            trainer.global_step,
            'train',
        )

    def on_validation_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs validation examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer,
                       'datamodule'), "Trainer missing datamodule attribute"
        assert hasattr(trainer,
                       'global_step'), "Trainer missing global_step attribute"

        if self.val_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(
            pl_module,
            self.val_dataloader,
            trainer.logger.experiment,
            trainer.global_step,
            'val',
        )
