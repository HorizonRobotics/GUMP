import unittest
import unittest.mock as mock

import torch
from nuplan.planning.training.preprocessing.features.agents_trajectories import \
    AgentsTrajectories
from nuplan.planning.training.preprocessing.features.trajectory import \
    Trajectory
from nuplan_extent.mocking import third_party

with mock.patch.dict('sys.modules', third_party=third_party):
    from nuplan_extent.planning.training.modeling.metrics.collision_trajectory_error_metric import CollisionTrajectoryError


class TestCollisionTrajectoryError(unittest.TestCase):
    def setUp(self) -> None:
        ego_width = 1.0
        ego_front_length = 1.0
        ego_rear_length = 1.0
        compute_step_indexes = [0, 1, 2, 3, 4]

        self.collision_trajectory_error_rotated_IoU = CollisionTrajectoryError(
            name="collision_trajectory_error_rotated_IoU",
            ego_front_length=ego_front_length,
            ego_rear_length=ego_rear_length,
            ego_width=ego_width,
            compute_step_indexes=compute_step_indexes)
        self.target_error_ratoted_iou = 1.0
        self.collision_trajectory_error_x_aligned = CollisionTrajectoryError(
            name="collision_trajectory_error_x_aligned",
            ego_front_length=ego_front_length,
            ego_rear_length=ego_rear_length,
            ego_width=ego_width,
            compute_step_indexes=compute_step_indexes)
        self.target_error_x_aligned = 1.0

    def test_compute(self):
        predictions = {
            "trajectory":
                Trajectory(
                    data=torch.tensor(
                        [[[-4.30500145e+01, -2.11315193e+01, -0.23804255],
                          [-4.47646561e+01, -7.76528478e-01, -0.54373522],
                          [-4.47567368e+01, -8.00000000e-01, -0.4509917],
                          [-3.19356441e+01, -2.60697193e+01, -0.32320212],
                          [-3.22201920e+01, -2.30107155e+01, 0.42091844]]],
                        dtype=torch.float32), )
        }
        targets = {
            "agents_trajectory_target":
                AgentsTrajectories(data=[
                    torch.tensor(
                        [[[
                            -4.30500145e+01, -2.11315193e+01, 1.01311588e+00,
                            2.04817796e+00, -6.55814695e+00, 1.66037560e-01,
                            7.61046112e-01, 7.88642585e-01
                        ],
                            [
                            -4.47646561e+01, -7.76528478e-01, 3.03043747e+00,
                            2.09344482e+00, -6.43565416e+00, -1.04125881e+00,
                            1.17841566e+00, 8.84633064e-01
                        ],
                            [
                            -4.47567368e+01, -8.06758225e-01,
                            -2.42091703e+00, 2.06792474e+00, -6.59513140e+00,
                            0.00000000e+00, 1.92152584e+00, 8.70497227e-01
                        ],
                            [
                            -3.19356441e+01, -2.60697193e+01, 2.64365911e-01,
                            1.97319114e+00, -6.65072775e+00, 1.72483921e-02,
                            5.06997442e+00, 2.01937127e+00
                        ],
                            [
                            -3.22201920e+01, -2.30107155e+01,
                            -2.92585564e+00, 1.97319114e+00, -6.65072775e+00,
                            0.00000000e+00, 4.84126759e+00, 1.91625988e+00
                        ]],
                            [[
                             -4.31054535e+01, -2.09199104e+01, 1.17915344e+00,
                             1.98526573e+00, -6.29647970e+00, 1.66035637e-01,
                             7.83899307e-01, 8.09854805e-01
                             ],
                             [
                                -4.43205795e+01, -4.98662204e-01, 1.98917866e+00,
                                2.07152915e+00, -6.22896194e+00, -1.38843715e+00,
                                1.23758805e+00, 9.19544816e-01
                            ],
                            [
                                -4.47567368e+01, -8.06758225e-01,
                                -2.42091703e+00, 2.06792474e+00, -6.59513140e+00,
                                -5.71200797e-16, 1.92152584e+00, 8.70497227e-01
                            ],
                            [
                                -3.19594040e+01, -2.59790306e+01, 2.81614304e-01,
                                1.97319114e+00, -6.65072775e+00, 2.70381421e-02,
                                5.21958208e+00, 2.06421971e+00
                            ],
                            [
                                -3.22201920e+01, -2.30107155e+01,
                                -2.92585564e+00, 1.97319114e+00, -6.65072775e+00,
                                2.11236416e-04, 4.84126759e+00, 1.91625988e+00
                            ]],
                            [[
                             -4.31450500e+01, -2.07687607e+01, 1.17915344e+00,
                             1.98526573e+00, -6.29647970e+00, 5.67331672e-01,
                             7.83899307e-01, 8.09854805e-01
                             ],
                             [
                             -4.43918571e+01, -2.26594314e-01, 1.64198422e+00,
                             2.05402613e+00, -6.41758871e+00, 1.36664715e-02,
                             1.21446788e+00, 9.21433687e-01
                             ],
                             [
                                -4.47567368e+01, -8.06758225e-01,
                                -2.42091703e+00, 2.06792474e+00, -6.59513140e+00,
                                -5.71200797e-16, 1.92152584e+00, 8.70497227e-01
                            ],
                            [
                                -3.20623627e+01, -2.55860424e+01, 2.91404366e-01,
                                1.97319114e+00, -6.65072775e+00, 1.45576696e-03,
                                5.17858839e+00, 2.11315465e+00
                            ],
                            [
                                -3.22993889e+01, -2.27084179e+01,
                                -2.92564440e+00, 1.97319114e+00, -6.65072775e+00,
                                3.14236805e-02, 5.32402086e+00, 2.01800752e+00
                            ]],
                            [[
                             -4.38346443e+01, -2.01095047e+01, 1.74649167e+00,
                             1.43190694e+00, -5.54124069e+00, 1.02594900e+00,
                             8.69534373e-01, 8.42625618e-01
                             ],
                             [
                             -4.44789772e+01, 1.05933085e-01, 2.00284529e+00,
                             1.87128127e+00, -6.27179289e+00, 1.26227701e+00,
                             1.19792938e+00, 9.28194761e-01
                             ],
                             [
                                -4.47567368e+01, -8.06758225e-01,
                                -2.42091703e+00, 2.06792474e+00, -6.59513140e+00,
                                -5.71200797e-16, 1.92152584e+00, 8.70497227e-01
                            ],
                            [
                                -3.21890793e+01, -2.51023674e+01, 2.83070087e-01,
                                1.97319114e+00, -6.65072775e+00, -8.33418220e-03,
                                4.91943693e+00, 1.98189723e+00
                            ],
                            [
                                -3.23310699e+01, -2.25874977e+01,
                                -2.89443159e+00, 1.97319114e+00, -6.65072775e+00,
                                -1.70139968e-01, 5.19577599e+00, 2.01238871e+00
                            ]],
                            [[
                             -4.33272095e+01, -2.00734768e+01, 2.20511436e+00,
                             1.93139279e+00, -6.49624681e+00, 4.58617359e-01,
                             8.76318872e-01, 8.65449727e-01
                             ],
                             [
                             -4.44868965e+01, 1.36162847e-01, 2.90427589e+00,
                             1.93834507e+00, -6.74094248e+00, 1.17898726e+00,
                             1.22306752e+00, 9.61075008e-01
                             ],
                             [
                                -4.47567368e+01, -8.06758225e-01,
                                -2.42091703e+00, 2.06792474e+00, -6.59513140e+00,
                                -5.71200797e-16, 1.92152584e+00, 8.70497227e-01
                            ],
                            [
                                -3.21890793e+01, -2.51023674e+01, 2.83070087e-01,
                                1.97319114e+00, -6.65072775e+00, 6.04570825e-17,
                                4.91943693e+00, 1.98189723e+00
                            ],
                            [
                                -3.23152313e+01, -2.26479588e+01,
                                -3.09578633e+00, 1.97319114e+00, -6.65072775e+00,
                                -1.04238257e-01, 4.96828175e+00, 1.94731128e+00
                            ]]],
                        dtype=torch.float32)
                ])
        }

        # Compute the metric and check that it is a scalar tensor
        error_rotated_iou = self.collision_trajectory_error_rotated_IoU.compute(
            predictions=predictions, targets=targets)
        self.assertTrue(torch.is_tensor(error_rotated_iou))
        self.assertEqual(error_rotated_iou, self.target_error_ratoted_iou)

        error_x_aligned = self.collision_trajectory_error_x_aligned.compute(
            predictions=predictions, targets=targets)
        self.assertTrue(torch.is_tensor(error_x_aligned))
        self.assertEqual(error_x_aligned, self.target_error_x_aligned)


if __name__ == '__main__':
    unittest.main()
