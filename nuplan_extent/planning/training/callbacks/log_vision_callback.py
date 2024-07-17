import random
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.nn import functional as F

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate


class LogVisionCallback(pl.Callback):
    """
    Callback that visualizes planner model inputs/outputs and logs them in Tensorboard.
    """

    def __init__(
        self,
        log_iter: int = 20,
    ):
        """
        Initialize the class.
        """
        self.log_iter = log_iter

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        *args: Any,
    ) -> None:
        loggers = trainer.logger.experiment

        vision_x = pl_module.model.encoders[0].log_data['vision_x']
        logger = loggers
        if trainer.global_step % self.log_iter == self.log_iter - 1:
            logger.add_histogram('vision_x',
                                    vision_x.float().reshape(-1).cpu().detach(),
                                    trainer.global_step)
