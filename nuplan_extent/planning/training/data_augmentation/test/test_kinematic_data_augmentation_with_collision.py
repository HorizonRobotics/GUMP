import logging
import unittest

import numpy as np

from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.planning.training.data_augmentation.kinematic_data_augmentation import \
    RasterKinematicAgentAugmentor
from nuplan_extent.planning.training.preprocessing.features.raster import \
    HorizonRaster

logger = logging.getLogger(__name__)


class TestKinematicAgentAugmentationWithCollision(unittest.TestCase):
    """Test agent augmentation with kinematic constraints."""

    def setUp(self) -> None:
        """Set up test case."""
        np.random.seed(2022)

        self.features = {}
        self.features['raster'] = HorizonRaster(np.ones((224, 224, 18)))

        self.targets = {}
        self.targets['trajectory'] = Trajectory(
            data=np.array([
                [-1.2336078e-03, 2.2296980e-04, -2.0750620e-05],
                [3.2337871e-03, 3.5673147e-04, -1.1526359e-04],
                [2.5042057e-02, 4.6393462e-04, -4.5901173e-04],
                [2.4698858e-01, -1.5322007e-03, -1.3717031e-03],
                [8.2662332e-01, -7.1887751e-03, -3.9011773e-03],
                [1.7506398e00, -1.7746322e-02, -7.2191255e-03],
                [3.0178127e00, -3.3933811e-02, -9.0915877e-03],
                [4.5618219e00, -5.3034388e-02, -4.8586642e-03],
                [6.3618584e00, -6.5912366e-02, 2.6488048e-04],
                [8.3739414e00, -6.9805034e-02, 4.0571247e-03],
                [1.0576758e01, -4.4418037e-02, 7.4823718e-03],
                [1.2969443e01, -1.7768066e-02, 9.7025689e-03],
            ]))

        self.gaussian_aug_targets_gt = {}
        self.gaussian_aug_targets_gt['trajectory'] = Trajectory(
            data=np.array([
                [4.1521129e-01, 1.1039978e-01, 4.1797668e-01],
                [5.0462860e-01, 1.4907575e-01, 3.9849171e-01],
                [6.3200253e-01, 2.0065330e-01, 3.7100676e-01],
                [7.9846221e-01, 2.6203236e-01, 3.3552179e-01],
                [1.0052546e00, 3.2913640e-01, 2.9203683e-01],
                [1.2535783e00, 3.9687237e-01, 2.4055186e-01],
                [1.5443755e00, 4.5909974e-01, 1.8106690e-01],
                [1.8780817e00, 5.0862163e-01, 1.1358193e-01],
                [2.2541707e00, 5.3959757e-01, 5.0773341e-02],
                [2.6713488e00, 5.5327171e-01, 1.4758691e-02],
                [3.1287551e00, 5.5699998e-01, 1.5426531e-03],
                [3.6260972e00, 5.5770481e-01, 1.2917991e-03],
            ]))

        self.uniform_aug_targets_gt = {}
        self.uniform_aug_targets_gt['trajectory'] = Trajectory(
            data=np.array([
                [2.01252978e-02, 3.89992601e-05, 3.87564092e-03],
                [8.02475438e-02, 6.14335586e-04, 1.52626643e-02],
                [1.80342704e-01, 3.01835593e-03, 3.27628106e-02],
                [3.20336223e-01, 9.03815683e-03, 5.31853959e-02],
                [5.00104427e-01, 2.03316603e-02, 7.22948909e-02],
                [7.19533801e-01, 3.77986841e-02, 8.65741968e-02],
                [9.78589535e-01, 6.13556951e-02, 9.47952345e-02],
                [1.27731919e+00, 9.02655423e-02, 9.81558487e-02],
                [1.61579382e+00, 1.23733692e-01, 9.89620313e-02],
                [1.99405646e+00, 1.61319897e-01, 9.91185382e-02],
                [2.41211104e+00, 2.02980295e-01, 9.95315537e-02],
                [2.86986732e+00, 2.48882845e-01, 1.00354895e-01],
            ]))

        N = 12
        dt = 0.1
        augment_prob = 1.0
        mean = [0.3, 0.1, np.pi / 12]
        std = [0.5, 0.1, np.pi / 12]
        low = [-0.1, -0.1, -0.1]
        high = [0.1, 0.1, 0.1]
        target_width = 224
        target_height = 224
        target_pixel_size = 0.5
        ego_width = 2.297
        ego_front_length = 4.049
        ego_rear_length = 1.127
        ego_longitudinal_offset = 0.0
        min_displacement = 2.0

        self.gaussian_augmentor = RasterKinematicAgentAugmentor(
            N,
            dt,
            mean,
            std,
            low,
            high,
            augment_prob,
            target_width,
            target_height,
            target_pixel_size,
            ego_width,
            ego_front_length,
            ego_rear_length,
            ego_longitudinal_offset,
            min_displacement=min_displacement,
            use_uniform_noise=False,
            speed_channel_index=5)

    def test_augment_with_collision(self) -> None:
        """
        Test no augmentation when collision happens.
        """
        _, aug_targets = self.gaussian_augmentor.augment(
            self.features, self.targets)
        self.assertTrue(
            (aug_targets['trajectory'].data == self.targets['trajectory'].data
             ).all())


if __name__ == '__main__':
    unittest.main()
