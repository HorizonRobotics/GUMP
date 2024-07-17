import random
from typing import Any, List, Optional, Dict

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
import nuplan_extent.planning.training.modeling.models.tokenizers.gump_tokenizer_utils as gutils

from einops import rearrange
from copy import deepcopy

import cv2
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

dim_factor = 0.8  # Adjust this factor to control the dimness

BASELINE_TL_COLOR = {
    TrafficLightStatusType.RED.value: tuple(int(dim_factor * x) for x in (220, 20, 60)),       # Dimmer Crimson Red
    TrafficLightStatusType.YELLOW.value: tuple(int(dim_factor * x) for x in (255, 215, 0)),    # Dimmer Gold
    TrafficLightStatusType.GREEN.value: tuple(int(dim_factor * x) for x in (60, 179, 113)),    # Dimmer Medium Sea Green
    TrafficLightStatusType.UNKNOWN.value: tuple(int(dim_factor * x) for x in (0, 43, 226))  # Dimmer Blue Violet
}


class VisualizationGUMPCallback(pl.Callback):
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
                      predictions, num_predict_steps, num_target_steps):
        # raster data
        if self.dataset =='nuplan':
            # import pickle
            # pickle.dump(predictions['raster'], open('/mnt/nas20/yihan01.hu/tmp/raster.pkl', 'wb'))
            for name, color in self.raster_type.items():
                if isinstance(predictions['raster'][name], torch.Tensor) and predictions['raster'][name].is_cuda:
                    predictions['raster'][name] = predictions['raster'][name].cpu().numpy()
            raster_map = sum([(predictions['raster'][name][..., None] @ np.array(color)[None]) for name, color in self.raster_type.items()])# B, H, W, C
            raster_map = np.clip(raster_map * 255, 0, 255).astype(np.uint8)
            predict_raster_map = np.tile(raster_map[:, None], (1, num_predict_steps, 1, 1, 1)) # B, t, h, w, c
            target_raster_map = np.tile(raster_map[:, None], (1, num_target_steps, 1, 1, 1))
        else:
            raster_map = predictions['raster'].data[:, [3, 4, 7]].numpy() / 4  # B, C, H, W
            predict_raster_map = (raster_map * 255).astype(np.uint8).transpose(0, 2, 3, 1)[:, None]  # B, t, h, w, c
            predict_raster_map[..., 2] = 0 # clear the traffic light channel
            num_predict_timestep = predictions['all_seqeuence_tokens'].data[0].num_frames
            predict_raster_map = np.tile(predict_raster_map, (1, num_predict_steps, 1, 1, 1))
            
            target_raster_map = (raster_map * 255).astype(np.uint8).transpose(0, 2, 3, 1)[:, None]  # B, t, h, w, c
            # target_raster_map[..., 2] = 0 # clear the traffic light channel
            num_target_timestep = predictions['generic_agents'].ego[0].shape[0]
            target_raster_map = np.tile(target_raster_map, (1, num_target_steps, 1, 1, 1))            
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
        render_index=0,
        return_res=False
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

        predicted_occ_tag = f'{prefix}_tokenized_array_pred'
        target_occ_tag = f'{prefix}_tokenized_array_target'
        tokenized_arrays = predictions['tokenized_arrays'][:, 0]
        predicted_tokenized_arrays = predictions['predicted_tokenized_arrays'][:, render_index]
        
        # import pdb; pdb.set_trace()
        # tokenized_arrays = gutils.detokenize_data(tokenized_arrays)
        # predicted_tokenized_arrays = gutils.detokenize_data(predicted_tokenized_arrays)

        # B, T, C, H, W
        # preprocess target features
        batch_size = tokenized_arrays.shape[0]
        num_target_steps = gutils.count_num_frame(tokenized_arrays)
        num_predict_steps = gutils.count_num_frame(predicted_tokenized_arrays)
        
        predict_raster_map, target_raster_map = self.render_raster(predictions, num_predict_steps, num_target_steps)

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

        for batch_index in range(batch_size):
            for t in range(num_target_steps):
                target_canvas[batch_index, t] = self.render_on_canvas_at_t(
                    tokenized_arrays[batch_index],
                    target_canvas[batch_index, t],
                    frame_index=t
                )  # render target tokenized array
                

            for t in range(num_predict_steps):
                predict_canvas[batch_index, t] = self.render_on_canvas_at_t(
                    predicted_tokenized_arrays[batch_index],
                    predict_canvas[batch_index, t],
                    frame_index=t
                )  # render target tokenized array
        target_canvas = rearrange(target_canvas, 'b t h w c -> b t c h w')
        predict_canvas = rearrange(predict_canvas, 'b t h w c -> b t c h w')
        if return_res:
            return target_canvas, predict_canvas
        logger = loggers
        fps = 2
        logger.add_video(
            predicted_occ_tag,
            predict_canvas,
            global_step=training_step,
            fps=fps,
        )
        logger.add_images(
            predicted_occ_tag + 'image',
            predict_canvas.max(axis=1),
            global_step=training_step)
        logger.add_video(
            target_occ_tag,
            target_canvas,
            global_step=training_step,
            fps=fps,
        )
        logger.add_images(
            target_occ_tag + 'image',
            target_canvas.max(axis=1),
            global_step=training_step)
    
    def render_on_canvas_at_t(self, tokenized_arrays, canvas, frame_index):
        colors = plt.cm.rainbow(np.linspace(0, 1, 12))
        tokenized_arrays = tokenized_arrays.view(gutils.NpTokenizedSequenceArray)
        current_tokenized_arrays = tokenized_arrays[tokenized_arrays.frame_index == frame_index]
        for i in range(len(current_tokenized_arrays)):
            # if tokenized_arrays:
            x = current_tokenized_arrays[i].x
            y = current_tokenized_arrays[i].y
            w = current_tokenized_arrays[i].width
            l = current_tokenized_arrays[i].length
            # import pdb; pdb.set_trace()
            heading = current_tokenized_arrays[i].heading
            token_id = current_tokenized_arrays[i].track_id
            is_ego = current_tokenized_arrays[i].is_ego
            x, y, w, l, heading = self._transform_bbox(x, y, w, l, heading)
            color = colors[int(token_id) % len(colors)]  # ego color
            # if is_ego: shift the center to the rear axle
            canvas = draw_bev_bboxes([x, y, w, l, heading], canvas , color=color*255, fill=is_ego, l_shift=-1.461 if is_ego else 0.0)
        return canvas

    def _transform_bbox(self, x, y, w, l, heading):
        x, y, = -y, -x
        heading = - heading
        x = x / self.pixel_size + self.canvas_size / 2
        y = y / self.pixel_size + self.canvas_size / 2
        w, l = w / self.pixel_size, l / self.pixel_size
        return x, y, w, l, heading

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
