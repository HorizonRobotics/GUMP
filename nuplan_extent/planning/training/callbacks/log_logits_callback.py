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
from nuplan_extent.planning.training.modeling.models.modules.nanoGPT.state_machine.state_type import VocabularyStateType


class LogLogitsCallback(pl.Callback):
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
        self.statistic = {
            'token_len': [],
            'seq_len': []
        }

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

        predict_logits = pl_module.objectives[0].log_data['predict_logits']
        target_tokens = pl_module.objectives[0].log_data['target_tokens']
        token_mask = pl_module.objectives[0].log_data['token_mask']
        loss = pl_module.objectives[0].log_data['loss'][token_mask.reshape(-1)]
        token_len = token_mask.sum(1)
        seq_len = (target_tokens == VocabularyStateType.BOS_TOKEN.start).sum(1)
        self.statistic['token_len'].append(token_len)
        self.statistic['seq_len'].append(seq_len)
        prob = F.softmax(predict_logits, dim=-1)
        prob_topk, _ = torch.topk(prob, 10)

        logger = loggers
        if trainer.global_step % self.log_iter == self.log_iter - 1:
            for logger in loggers:
                logger.add_histogram(
                    'masked_cross_entrophy_hist',
                    loss.cpu().detach(),
                    trainer.global_step)
                logger.add_histogram('predict_prob',
                                     prob_topk.cpu().detach().reshape(-1),
                                     trainer.global_step)
                logger.add_histogram(
                    'token_len',
                    torch.cat(
                        self.statistic['token_len'],
                        dim=0).cpu().detach(),
                    trainer.global_step)
                logger.add_histogram(
                    'seq_len',
                    torch.cat(
                        self.statistic['seq_len'],
                        dim=0).cpu().detach(),
                    trainer.global_step)
                self.statistic['token_len'] = []
                self.statistic['seq_len'] = []
