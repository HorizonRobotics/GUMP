import unittest

from PIL import Image

from nuplan_extent.planning.scenario_builder.nuscenes_db.test.test_nuscenes_scenario import \
    mock_nuscenes_scenario
from nuplan_extent.planning.training.preprocessing.feature_builders.e2e_feature_builder import \
    E2EFeatureBuilder
from nuplan_extent.planning.training.preprocessing.features.abstract_cameras_pipelines import \
    AbstractCameraPipelines
from nuplan_extent.planning.training.preprocessing.features.camera_e2e import \
    CameraE2E
from nuplan_extent.planning.training.preprocessing.features.camera_pipelines import (
    NormalizeMultiviewImage, ResizeCropMultiViewImages)
from nuplan_extent.planning.training.preprocessing.features.cameras import \
    Cameras


class MockLoadMultiViewImages(AbstractCameraPipelines):
    """Fake LoadMultiViewImage class that generate images instead of load images."""

    def __init__(self):
        pass

    def __call__(self, cameras: Cameras) -> Cameras:
        """Call function to load multi-view image from files.

        :param cameras: Result cameras object containing multi-view image filenames.
        :return cameras: The result cameras object containing the multi-view image data.
        """
        cameras.imgs = []
        for image_filename in cameras.img_filename:
            img = Image.new(mode="RGB", size=(900, 1600))
            cameras.imgs.append(img)
        return cameras

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


class TestE2EFeatureBuilder(unittest.TestCase):
    """Test builder that constructs E2E features during training."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.image_cfg = dict(
            names=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            original_height=900,
            original_width=1600,
            resize_scale=0.3,
            top_crop=46,
            final_dim=[224, 480],
        )
        camera_pipelines = [
            MockLoadMultiViewImages(),
            ResizeCropMultiViewImages(image_params=self.image_cfg),
            NormalizeMultiviewImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.feature_builder = E2EFeatureBuilder(
            camera_pipelines=camera_pipelines, )

    def test_e2e_feature_builder(self):
        """
        Test E2EFeatureBuilder.
        """
        scenario = mock_nuscenes_scenario(2, 2, 4, False)
        feature = self.feature_builder.get_features_from_scenario(
            scenario=scenario, )
        self.assertIsInstance(feature, CameraE2E)
        camera_shape = (2, 6, 3, *self.image_cfg["final_dim"])
        self.assertEqual(camera_shape, feature.cameras.shape)
        self.assertEqual((2, 6, 3, 3), feature.intrinsics.shape)
        self.assertEqual((2, 6, 4, 4), feature.extrinsics.shape)
        self.assertEqual((2, 6), feature.future_egomotion.shape)
        self.assertEqual(2, feature.command)

    def test_e2e_feature_builder_with_iteration(self):
        """
        Test E2EFeatureBuilder with iteration.
        """
        scenario = mock_nuscenes_scenario(2, 2, 10, True)
        feature = self.feature_builder.get_features_from_scenario(
            scenario=scenario,
            iteration=3,
        )
        self.assertIsInstance(feature, CameraE2E)
        camera_shape = (2, 6, 3, *self.image_cfg["final_dim"])
        self.assertEqual(camera_shape, feature.cameras.shape)
        self.assertEqual((2, 6, 3, 3), feature.intrinsics.shape)
        self.assertEqual((2, 6, 4, 4), feature.extrinsics.shape)
        self.assertEqual((2, 6), feature.future_egomotion.shape)
        self.assertEqual(2, feature.command)


if __name__ == '__main__':
    unittest.main()
