import random
from typing import Any, List, Optional

import os
import imageio
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.features.raster_utils import get_ego_raster
from nuplan_extent.planning.training.callbacks.utils.visualization_utils import draw_bev_bboxes, map_to_rgb, draw_bev_trajectory, draw_velocity

from einops import rearrange
from copy import deepcopy
from PIL import Image

import cv2
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

dim_factor = 0.8  # Adjust this factor to control the dimness

BASELINE_TL_COLOR = {
    TrafficLightStatusType.RED.value: tuple(int(dim_factor * x) for x in (220, 20, 60)),       # Dimmer Crimson Red
    TrafficLightStatusType.YELLOW.value: tuple(int(dim_factor * x) for x in (255, 215, 0)),    # Dimmer Gold
    TrafficLightStatusType.GREEN.value: tuple(int(dim_factor * x) for x in (60, 179, 113)),    # Dimmer Medium Sea Green
    TrafficLightStatusType.UNKNOWN.value: tuple(int(dim_factor * x) for x in (0, 43, 226))  # Dimmer Blue Violet
}

def save_png(tensor, filename):
    # Rearrange to (batch, time, height, width, channels)
    frames = rearrange(tensor, 'b t c h w -> b t h w c')
    for i, video in enumerate(frames):
        # Sum the frames along the time axis
        img = video.astype(np.float32).max(axis=0)
        # Normalize the summed image
        img = (img / img.max() * 255).astype('uint8')
        # Save the summed image as PNG
        img_path = f"{filename}_{i}.png"
        Image.fromarray(img).save(img_path)

class VisualizationVectorizedReconstructionCallback(pl.Callback):
    """
    Callback that visualizes planner model inputs/outputs and logs them in Tensorboard.
    """

    def __init__(
        self,
        images_per_tile: int,
        num_train_tiles: int,
        num_val_tiles: int,
        pixel_size: float,
        num_frames: int,
        num_future_imagine_frames: int,
        num_past_imagine_frames: int,
        canvas_size: int,
        vis_autoencoder: bool = False,
        vis_trajectory: bool = False,
        vis_topk: int = 3,
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
        self.num_frames = num_frames
        self.num_future_imagine_frames = num_future_imagine_frames
        self.num_past_imagine_frames = num_past_imagine_frames
        self.canvas_size = canvas_size
        self.vis_autoencoder = vis_autoencoder
        self.vis_trajectory = vis_trajectory
        self.vis_topk = vis_topk
        self.dataset = dataset

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
            
    def render_raster(self,
                      predictions):
        # raster data
        if self.dataset =='nuplan':
            raster_map = predictions['raster'].data[:, [3, 4, 9]].numpy() / 4  # B, C, H, W
            predict_raster_map = (raster_map * 255).astype(np.uint8).transpose(0, 2, 3, 1)[:, None]  # B, t, h, w, c
            predict_raster_map[..., 2] = 0 # clear the traffic light channel
            num_predict_timestep = predictions['all_seqeuence_tokens'].data[0].num_frames
            predict_raster_map = np.tile(predict_raster_map, (1, num_predict_timestep, 1, 1, 1))
            
            target_raster_map = (raster_map * 255).astype(np.uint8).transpose(0, 2, 3, 1)[:, None]  # B, t, h, w, c
            target_raster_map[..., 2] = 0 # clear the traffic light channel
            num_target_timestep = predictions['generic_agents'].ego[0].shape[0]
            target_raster_map = np.tile(target_raster_map, (1, num_target_timestep, 1, 1, 1))
            
            traffic_light_raster = predictions['raster'].data[:, 9].long().numpy()
            traffic_light_index_in_raster = set(np.unique(traffic_light_raster))
            traffic_light_index_in_raster.remove(0)
            
            for batch_index in range(target_raster_map.shape[0]):
                seqeuence_tokens = predictions['sequence_tokens'].data[batch_index]
                
                for t in range(num_target_timestep):
                    seqeuence_tokens_at_t = seqeuence_tokens.sample_frame_segment(start_local_index=t, end_local_index=t)
                    traffic_light_state = seqeuence_tokens_at_t.get_traffic_light_state(return_all=True)
                    for i in range(len(traffic_light_state)):    
                        color = [255, 255, 255]
                        x, y, heading, state, index = traffic_light_state[i]
                        x, y, w, l, heading = self._transform_bbox(x, y, 2, 4, heading)
                        target_raster_map[batch_index, t] = draw_bev_bboxes(
                            [x, y, w, l, heading], target_raster_map[batch_index, t] , color=color)
                        if int(index) in traffic_light_index_in_raster:
                            mask = traffic_light_raster[batch_index] == int(index)
                            target_raster_map[batch_index, t, mask] = BASELINE_TL_COLOR[int(state)]
                            
            for batch_index in range(target_raster_map.shape[0]):
                seqeuence_tokens = predictions['all_seqeuence_tokens'].data[batch_index]
                for t in range(num_target_timestep):
                    seqeuence_tokens_at_t = seqeuence_tokens.sample_frame_segment(start_local_index=t, end_local_index=t)
                    traffic_light_state = seqeuence_tokens_at_t.get_traffic_light_state(return_all=True)
                    
                    for i in range(len(traffic_light_state)):    
                        x, y, heading, state, index = traffic_light_state[i]
                        if int(index) in traffic_light_index_in_raster:
                            mask = traffic_light_raster[batch_index] == int(index)
                            predict_raster_map[batch_index, t, mask] = BASELINE_TL_COLOR[int(state)]
        else:
            raster_map = predictions['raster'].data[:, [3, 4, 7]].numpy() / 4  # B, C, H, W
            predict_raster_map = (raster_map * 255).astype(np.uint8).transpose(0, 2, 3, 1)[:, None]  # B, t, h, w, c
            predict_raster_map[..., 2] = 0 # clear the traffic light channel
            num_predict_timestep = predictions['all_seqeuence_tokens'].data[0].num_frames
            predict_raster_map = np.tile(predict_raster_map, (1, num_predict_timestep, 1, 1, 1))
            
            target_raster_map = (raster_map * 255).astype(np.uint8).transpose(0, 2, 3, 1)[:, None]  # B, t, h, w, c
            # target_raster_map[..., 2] = 0 # clear the traffic light channel
            num_target_timestep = predictions['generic_agents'].ego[0].shape[0]
            target_raster_map = np.tile(target_raster_map, (1, num_target_timestep, 1, 1, 1))            
        return predict_raster_map, target_raster_map

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

        predicted_occ_tag = f'{prefix}_visualization_reconstructed_pred'
        target_occ_tag = f'{prefix}_visualization_reconstructed_target'

        # B, T, C, H, W
        # preprocess target features
        
        batch_size = len(predictions['generic_agents'].ego)
        num_target_steps = predictions['generic_agents'].ego[0].shape[0]
        all_seqeuence_tokens = predictions['all_seqeuence_tokens']
        num_predict_steps = all_seqeuence_tokens.data[0].num_frames
        
        colors = plt.cm.rainbow(np.linspace(0, 1, 12))
        predict_raster_map, target_raster_map = self.render_raster(predictions)

        target_canvas = np.zeros(
            (batch_size,
             num_target_steps,
             self.canvas_size,
             self.canvas_size,
             3),
            dtype=np.uint8)
        predict_canvas = np.zeros(
            (batch_size,
             num_predict_steps,
             self.canvas_size,
             self.canvas_size,
             3),
            dtype=np.uint8)
        target_canvas += target_raster_map
        predict_canvas += predict_raster_map

        class_names = list(predictions['generic_agents'].agents.keys())

        target_sequence_tokens = predictions['sequence_tokens']
        pred_sequence_tokens = predictions['all_seqeuence_tokens']
        for batch_index in range(batch_size):
            for t in range(num_target_steps):
                current_target_sequence_tokens = target_sequence_tokens.data[batch_index].sample_frame_segment(start_local_index=t, end_local_index=t)
                # print(t, batch_index, current_target_sequence_tokens)
                target_canvas[batch_index, t] = self.render_vector_data(current_target_sequence_tokens, target_canvas[batch_index, t], colors)
                

            for t in range(num_predict_steps):
                if t >= predict_canvas.shape[1]:
                    break
                current_pred_sequence_tokens = pred_sequence_tokens.data[batch_index].sample_frame_segment(start_local_index=t, end_local_index=t)
                predict_canvas[batch_index, t] = self.render_vector_data(current_pred_sequence_tokens, predict_canvas[batch_index, t], colors)
        target_canvas = rearrange(target_canvas, 'b t h w c -> b t c h w')
        predict_canvas = rearrange(predict_canvas, 'b t h w c -> b t c h w')
        logger = loggers
        fps = 2
        logger.add_video(
            predicted_occ_tag,
            predict_canvas,
            global_step=training_step,
            fps=fps,
        )
        logger.add_images(
            predicted_occ_tag + '_image',
            predict_canvas.max(axis=1),
            global_step=training_step)
        logger.add_video(
            target_occ_tag,
            target_canvas,
            global_step=training_step,
            fps=fps,
        )
        logger.add_images(
            target_occ_tag+ '_image',
            target_canvas.max(axis=1),
            global_step=training_step)

        # Save videos as GIFs
        log_dir = logger.log_dir
        save_png(predict_canvas, os.path.join(log_dir, f'predicted_occ_step_{training_step}'))
        save_png(target_canvas, os.path.join(log_dir, f'target_occ_step_{training_step}'))
        if self.vis_autoencoder:
            colormap = torch.tensor([
                [0.5, 0, 0],  # Red
                [0, 0.5, 0],  # Green
                [0, 0, 0.5],  # Blue
                [0.5, 0.5, 0],  # Yellow
                [0, 0.5, 0.5]   # Cyan
            ]).float()
            reconstructed_image = predictions['reconstruct_image'].data.float(
            )
            rgb_reconstructed_image = map_to_rgb(
                reconstructed_image, colormap)
            logger.add_images(
                "Reconstructed Images",
                rgb_reconstructed_image,
                global_step=training_step)
                
    def render_vector_data(self, sequence_tokens, target_canvas, colors):
        agent_tokens = sequence_tokens.get_agent_tokens(return_all=True)
        for agent_token in agent_tokens:
            x, y, heading, w, l, type_id = agent_token.get_agent_state()
            token_id = agent_token.token_id
            x, y, w, l, heading = self._transform_bbox(x, y, w, l, heading)
            color = colors[int(token_id) % len(colors)]  # ego color
            is_ego = agent_token.is_ego
            # if is_ego: shift the center to the rear axle
            target_canvas = draw_bev_bboxes([x, y, w, l, heading], target_canvas , color=color*255, fill=is_ego, l_shift=-1.461 if is_ego else 0.0)
            
            if self.vis_trajectory:
                if agent_token.pred_future_prob is not None:
                    pred_future_trajectory = deepcopy(agent_token.predicted_future_trajectory)
                    pred_future_prob = agent_token.pred_future_prob
                    topk_inds = np.argsort(pred_future_prob)[::-1][:3]
                    pred_future_trajectory = pred_future_trajectory[topk_inds]
                    pred_future_trajectory = self._transform_trajectory(pred_future_trajectory)
                    pred_future_trajectory[..., 0] += x
                    pred_future_trajectory[..., 1] += y
                    target_canvas = draw_bev_trajectory(pred_future_trajectory, target_canvas, color=color*255)  
                if agent_token.future_trajectory is not None:
                    target_future_trajectory = deepcopy(agent_token.future_trajectory)
                    target_future_trajectory = self._transform_trajectory(target_future_trajectory)    
                    nan_mask = np.isnan(target_future_trajectory).any(axis=-1)
                    target_future_trajectory = target_future_trajectory[~nan_mask]
                    
                    target_future_trajectory[..., 0] += x
                    target_future_trajectory[..., 1] += y
                    color = [0, 255, 0]
                    target_canvas = draw_bev_trajectory(target_future_trajectory, target_canvas, color=color)   
                
        return target_canvas


    def _transform_bbox(self, x, y, w, l, heading):
        x, y, = -y, -x
        heading = - heading
        x = x / self.pixel_size + self.canvas_size / 2
        y = y / self.pixel_size + self.canvas_size / 2
        w, l = w / self.pixel_size, l / self.pixel_size
        return x, y, w, l, heading

    def _transform_trajectory(self, traj):
        traj[..., :2] = - traj[..., [1, 0]]
        traj[..., 2] = - traj[..., 2]
        traj[..., :2] = traj[..., :2] / self.pixel_size 
        return traj

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
