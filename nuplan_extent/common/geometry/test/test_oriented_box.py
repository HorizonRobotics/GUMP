import unittest
import numpy as np
from shapely.geometry import Polygon
import time

import nuplan_extent.common.geometry.oriented_box as ob


class TestOrientedBox(unittest.TestCase):
    def test_box_box_dist(self):
        # compare with shapely Polygon.distance
        n = 100
        box1 = np.random.rand(n, 5).astype(np.float32)
        box2 = np.random.rand(n, 5).astype(np.float32)
        box1[:, :2] *= 10
        box1[:, 2] *= 2 * np.pi
        box1[:, 3:] *= 5
        box2[:, :2] *= 10
        box2[:, 2] *= 2 * np.pi
        box2[:, 3:] *= 5

        t0 = time.time()
        dists = ob.box_box_dist(box1, box2)
        t1 = time.time()
        print("Time for box_box_dist:", t1 - t0)

        t0 = time.time()
        corners1 = ob.box_to_corners(box1)
        corners2 = ob.box_to_corners(box2)
        dists2 = [Polygon(corners1[i]).distance(Polygon(corners2[i]))
                  for i in range(n)]
        t1 = time.time()
        print("Time for shapely:", t1 - t0)

        self.assertEqual(corners1.shape, (n, 4, 2))
        self.assertEqual(dists.shape, (n, ))

        for i in range(n):
            if abs(dists[i] - dists2[i]) > 1e-5:
                print("Failed for", box1[i], box2[i], dists[i], dists2[i])
            self.assertAlmostEqual(dists[i], dists2[i], places=5)


if __name__ == '__main__':
    unittest.main()
