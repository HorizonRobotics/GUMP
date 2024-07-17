import logging
from copy import deepcopy
from typing import List, Optional, Tuple

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import \
    AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import (
    ParameterToScale, ScalingDirection)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
Pose = Tuple[float, float, float]  # (x, y, yaw)


class CopyTargetToFeatureAugmentor(AbstractAugmentor):
    """
    This class copies the target trajectory to the feature dictionary.
    Sometimes the target trajectory is used as a input feature during training for the model.
    such as multibin model, we use target trajectory to generate high level command.

    """

    def __init__(
            self,
            trajectory_steps: int = 16,
    ) -> None:
        """
        :param trajectory_steps: int, total steps of expert trajectory.
        """
        self._augment_prob = 1.0
        self._trajectory_steps = trajectory_steps

    def augment(self,
                features: FeaturesType,
                targets: TargetsType,
                scenario: Optional[AbstractScenario] = None
                ) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        features['target_trajectory'] = deepcopy(targets['trajectory'])

        # if agents_occupancy_target shape is larger than 16, it has the
        # instance id mask, we need to remove it
        if 'agents_occupancy_target' in targets and targets[
                'agents_occupancy_target'].data.shape[-1] > self._trajectory_steps:
            targets['agents_occupancy_target'].data = targets['agents_occupancy_target'].data[..., 0::2]
        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['']

    @property
    def augmentation_probability(self) -> ParameterToScale:
        """Inherited, see superclass."""
        return ParameterToScale(
            param=self._augment_prob,
            param_name=f'{self._augment_prob}='.partition('=')[0].split('.')
            [1],
            scaling_direction=ScalingDirection.MAX,
        )

