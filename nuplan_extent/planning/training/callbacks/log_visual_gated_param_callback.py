import random
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.cuda.amp import autocast

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate


class LogVisualGatedParamCallback(pl.Callback):
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
        # pl_module.objectives[0].log_data['predict_logits']
        loggers = trainer.logger.experiment
        logger = loggers
        if trainer.global_step % self.log_iter == 0:
            # import pdb; pdb.set_trace()
            for k, v in trainer.model.state_dict().items():
                if ('ff_gate' in k) or ('attn_gate' in k):
                    logger.add_scalar(
                        'gate_value/' + k,
                        v.float().cpu().detach(),
                        trainer.global_step)