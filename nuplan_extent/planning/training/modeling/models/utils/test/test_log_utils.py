import os
import unittest

import numpy as np

import torch
from nuplan_extent.planning.training.modeling.models.utils.log_utils import \
    render_and_save_features
from nuplan_extent.planning.training.preprocessing.features.tensor import \
    Tensor


class TestUtils(unittest.TestCase):
    """
    Tests utils functions in models/utils.
    """

    def test_render_and_save_features(self):
        # Create a temporary file for testing
        filename = 'test.jpg'

        # Create some dummy features
        features = {
            'raster':
                Tensor(data=torch.randn(1, 6, 224, 224)),
            'pred_heatmap':
                Tensor(data=torch.randn(1, 1, 224, 224)),
            'trajectory':
                Tensor(
                    data=torch.Tensor(
                        np.random.uniform(-56, 56, size=(1, 16, 2))))
        }

        # Call the function to save the features to file
        render_and_save_features(features, filename, [-56, 56, -56, 56])

        # Check that the file was created
        self.assertTrue(os.path.isfile(filename))

        # Clean up the temporary file
        os.remove(filename)

        # Call the function to save the features to file
        render_and_save_features(features, filename, [-84, 28, -56, 56])

        # Check that the file was created
        self.assertTrue(os.path.isfile(filename))

        # Clean up the temporary file
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
