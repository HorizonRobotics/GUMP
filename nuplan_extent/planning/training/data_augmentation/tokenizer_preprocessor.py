import numpy as np
import numpy.typing as npt
import torch
import logging
from typing import List, Optional, Tuple, cast
from copy import deepcopy

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ConstrainedNonlinearSmoother,
    GaussianNoise,
    ParameterToScale,
    ScalingDirection,
    UniformNoise,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_augmented_ego_raster,
    rotate_tilt_angle,
)
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan_extent.planning.training.preprocessing.feature_builders.utils import agent_bbox_to_corners

from numba import jit
from typing import Any, Dict, List, Tuple, Callable, Union

from nuplan_extent.planning.training.preprocessing.features.tensor import Tensor
from nuplan_extent.planning.training.preprocessing.features.attribute_table import AttributeTable
from nuplan_extent.planning.training.preprocessing.features.token import Token
from nuplan_extent.planning.training.preprocessing.feature_builders.vector_tokenizer import (
    VectorTokenizer
)

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
Pose = Tuple[float, float, float]  # (x, y, yaw)


class TokenizerPreprocessorAugmentor(AbstractAugmentor):
    """
    This class copies the target trajectory to the feature dictionary.
    Sometimes the target trajectory is used as a input feature during training for the model.
    such as multibin model, we use target trajectory to generate high level command.

    """

    def __init__(self,
            num_past_steps: int,
            dataset: str,
            target_width: int = None,
            target_height: int = None,
            augment_cfg: Dict[str, Any] = None,):
        super().__init__()
        self.tokenizer = VectorTokenizer(
            num_past_steps,
            dataset,
            target_width,
            target_height,
            augment_cfg=augment_cfg,
        )

    def augment(
        self,
        features: FeaturesType,
        targets: TargetsType,
        scenario: Optional[AbstractScenario] = None
    ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        features = self.tokenizer.tokenize(features)
        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['trajectory']

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f'{self._augment_prob}='.partition('=')[0].split('.')
            [1],
            scaling_direction=ScalingDirection.MAX,
        )
