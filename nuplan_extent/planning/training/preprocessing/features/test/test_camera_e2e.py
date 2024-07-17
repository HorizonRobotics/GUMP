import unittest

import numpy as np

import torch
from nuplan_extent.planning.training.preprocessing.features.camera_e2e import \
    CameraE2E


class TestCameraE2E(unittest.TestCase):
    """Test Camera end to end representation."""

    def setUp(self) -> None:
        self.bs = 4
        self.frames = 3
        self.num_camera = 6
        self.image_dim = [224, 480]
        self.cameras = np.zeros((self.bs, self.frames, self.num_camera, 3,
                                 *self.image_dim))
        self.intrinsics = np.zeros((self.bs, self.frames, self.num_camera, 3,
                                    3))
        self.extrinsics = np.zeros((self.bs, self.frames, self.num_camera, 4,
                                    4))
        self.future_egomotion = np.zeros((self.bs, self.frames, 6))
        self.command = np.zeros((self.bs, self.frames))

    def test_camera_e2e(self) -> None:
        """
        Test the core functionality of features.
        """
        feature = CameraE2E(
            cameras=self.cameras,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            future_egomotion=self.future_egomotion,
            command=self.command,
        )
        self.assertEqual(self.bs, feature.num_batches)
        self.assertIsInstance(feature.cameras, np.ndarray)
        self.assertIsInstance(feature.intrinsics, np.ndarray)
        self.assertIsInstance(feature.extrinsics, np.ndarray)
        self.assertIsInstance(feature.future_egomotion, np.ndarray)

        feature = feature.to_feature_tensor()
        self.assertIsInstance(feature.cameras, torch.Tensor)
        self.assertIsInstance(feature.intrinsics, torch.Tensor)
        self.assertIsInstance(feature.extrinsics, torch.Tensor)
        self.assertIsInstance(feature.future_egomotion, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
