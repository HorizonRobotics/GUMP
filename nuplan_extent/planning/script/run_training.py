import logging
import os
import warnings
from typing import Optional
import torch.multiprocessing

import hydra
import torch
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import \
    build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.utils.utils_config import \
    update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from nuplan.planning.training.experiments.training import (
    TrainingEngine, build_training_engine)
from omegaconf import DictConfig, OmegaConf
import cv2

cv2.setNumThreads(1)
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')
logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and
# experiment paths
set_default_path()

# Add a new resolver that supports eval
OmegaConf.register_new_resolver("eval", eval)

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config')

CONFIG_NAME = os.getenv('DEFAULT_CONFIG', 'horizon_training')


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    # Build plugins (compatible with mmdet)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_path = _module_dir.replace("/", ".")
            if _module_path.startswith("."):
                _module_path = _module_path[1:]
            logger.info(f"Plugin directory: {_module_path}")
            plg_lib = importlib.import_module(_module_path)

    if cfg.py_func == 'train':
        # Build training engine
        engine = build_training_engine(cfg, worker)

        # Run training
        logger.info('Starting training...')
        
        # compatible with pytorch_lightning 1.3.8 and 2.1.11
        if hasattr(engine.trainer, 'resume_from_checkpoint'):
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule, ckpt_path=engine.trainer.resume_from_checkpoint)
        else:
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'test':
        # Build training engine
        engine = build_training_engine(cfg, worker)

        # Test model
        logger.info('Starting testing...')

        my_ckpt_path = cfg.checkpoint.ckpt_path
        assert isinstance(my_ckpt_path, str), 'Checkpoint path must be a string'
        assert os.path.exists(my_ckpt_path), f'Checkpoint path {my_ckpt_path} does not exist'
        
        my_ckpt = torch.load(my_ckpt_path, map_location='cpu')
        engine.model.load_state_dict(my_ckpt['state_dict']) 

        engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache':
        # Precompute and cache all features
        cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


if __name__ == '__main__':
    main()
