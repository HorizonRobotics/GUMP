from typing import Any, Dict
import pytest
import numpy as np
try:
    from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
except ModuleNotFoundError:
    # testing.nuplan_test is moved to test_utils.nuplan_test in newer versions of nuplan
    from nuplan.common.utils.test_utils.nuplan_test import NUPLAN_TEST_PLUGIN, nuplan_test
from nuplan_extent.planning.scenario_builder.p3c_db.p3c_scenario import \
    P3CScenario
from nuplan_extent.planning.training.preprocessing.feature_builders.p3c_raster_feature_builder import \
    P3CRasterFeatureBuilder

data_root = './nuplan_extent/common/maps/p3c_map/test/'
scenario_id = 'PNC_20221107-104916_497_0_SwitchLaneChange_1667789547590_8977_MergedMapNav_merged'
scenario = P3CScenario(data_root, scenario_id, future_horizon=4.0)
scenario._map_api.load_map()
scenario.load_agent_tracks()
feature_builder_cfg_no_offset = dict(
    map_features=dict(
        # name of map features to be drawn and its color [0.0, 1.0] for
        # encoding
        BOUNDARIES=1,
        TURN_STOP=2,
        LANE_EDGE=3,
        STOP_LINE=4,
        CROSSWALK=5,
    ),
    input_channel_indexes=[0, 1, 2, 3, 4,
                           5],  # number of input channel of the raster model
    target_width=512,  # width of raster passed to the model
    target_height=512,  # height of raster passed to the model
    target_pixel_size=0.2,  # [m] pixel size of raster
    ego_width=2.297,  # [m] width of ego vehicle
    # [m] rear axle to front bumper distance of ego vehicle
    ego_front_length=4.049,
    # [m] rear axle to rear bumper distance of ego vehicle
    ego_rear_length=1.127,
    # [%] offset percentage to move the ego vehicle inside the raster
    ego_longitudinal_offset=0.0,

    # [pixel] the thickness of baseline paths in the baseline_paths_raster
    baseline_path_thickness=1,
    past_time_horizon=2.0,  # [s] time horizon of all poses
    past_num_poses=4,  # number of poses in a trajectory
    feature_time_interval=0.5,  # [s] time interval of each pose
    # [%] ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)
    subsample_ratio_override=1.0,
)

feature_builder_cfg_offset = dict(
    map_features=dict(
        # name of map features to be drawn and its color [0.0, 1.0] for
        # encoding
        BOUNDARIES=1,
        TURN_STOP=2,
        LANE_EDGE=3,
        STOP_LINE=4,
        CROSSWALK=5,
    ),
    input_channel_indexes=[0, 1, 2, 3, 4,
                           5],  # number of input channel of the raster model
    target_width=512,  # width of raster passed to the model
    target_height=512,  # height of raster passed to the model
    target_pixel_size=0.2,  # [m] pixel size of raster
    ego_width=2.297,  # [m] width of ego vehicle
    # [m] rear axle to front bumper distance of ego vehicle
    ego_front_length=4.049,
    # [m] rear axle to rear bumper distance of ego vehicle
    ego_rear_length=1.127,
    # [%] offset percentage to move the ego vehicle inside the raster
    ego_longitudinal_offset=0.2,

    # [pixel] the thickness of baseline paths in the baseline_paths_raster
    baseline_path_thickness=1,
    past_time_horizon=2.0,  # [s] time horizon of all poses
    past_num_poses=4,  # number of poses in a trajectory
    feature_time_interval=0.5,  # [s] time interval of each pose
    # [%] ratio used sample the scenario (e.g. a 0.1 ratio means sample from 20Hz to 2Hz)
    subsample_ratio_override=1.0,
)

p3c_raster_feat_builder_no_offset = P3CRasterFeatureBuilder(
    **feature_builder_cfg_no_offset)
p3c_raster_feat_builder_offset = P3CRasterFeatureBuilder(
    **feature_builder_cfg_offset)


@nuplan_test(path='json/raster_info.json')
def test_raster_future(scene: Dict[str, Any]) -> None:
    for iter in [0, 10, 20]:
        res_no_offset = p3c_raster_feat_builder_no_offset.get_features_from_scenario(
            scenario, iteration=iter)
        res_offset = p3c_raster_feat_builder_offset.get_features_from_scenario(
            scenario, iteration=iter)
        merge_image_no_offset = p3c_raster_feat_builder_no_offset.merge_all_raster(
            res_no_offset)
        merge_image_offset = p3c_raster_feat_builder_offset.merge_all_raster(
            res_offset)

        # no_offset
        assert int(np.sum(
            res_no_offset.data[:, :, 0])) == scene["no_offset"]["iter"][str(
                iter)]["ego_raster"]["sum"]
        assert int(np.sum(
            res_no_offset.data[:, :, 1])) == scene["no_offset"]["iter"][str(
                iter)]["agents_raster"]["sum"]
        assert int(np.sum(
            res_no_offset.data[:, :, 2])) == scene["no_offset"]["iter"][str(
                iter)]["roadmap_raster"]["sum"]
        assert int(np.sum(
            res_no_offset.data[:, :, 3])) == scene["no_offset"]["iter"][str(
                iter)]["baseline_paths_raster"]["sum"]
        assert int(np.sum(
            res_no_offset.data[:, :, 4])) == scene["no_offset"]["iter"][str(
                iter)]["route_raster"]["sum"]
        assert int(np.sum(
            res_no_offset.data[:, :, 5])) == scene["no_offset"]["iter"][str(
                iter)]["ego_speed_raster"]["sum"]
        assert int(np.sum(
            res_no_offset.data[:, :, 6])) == scene["no_offset"]["iter"][str(
                iter)]["drivable_area_raster"]["sum"]
        assert int(np.sum(
            res_no_offset.data[:, :, 7])) == scene["no_offset"]["iter"][str(
                iter)]["speed_limit_raster"]["sum"]
        assert int(
            np.mean(merge_image_no_offset)) == scene["no_offset"]["iter"][str(
                iter)]["merge_raster"]["mean"]
        assert int(
            np.std(merge_image_no_offset)) == scene["no_offset"]["iter"][str(
                iter)]["merge_raster"]["std"]

        # # offset 0.2
        assert int(np.sum(
            res_offset.data[:, :, 0])) == scene["offset"]["iter"][str(
                iter)]["ego_raster"]["sum"]
        assert int(np.sum(
            res_offset.data[:, :, 1])) == scene["offset"]["iter"][str(
                iter)]["agents_raster"]["sum"]
        assert int(np.sum(
            res_offset.data[:, :, 2])) == scene["offset"]["iter"][str(
                iter)]["roadmap_raster"]["sum"]
        assert int(np.sum(
            res_offset.data[:, :, 3])) == scene["offset"]["iter"][str(
                iter)]["baseline_paths_raster"]["sum"]
        assert int(np.sum(
            res_offset.data[:, :, 4])) == scene["offset"]["iter"][str(
                iter)]["route_raster"]["sum"]
        assert int(np.sum(
            res_offset.data[:, :, 5])) == scene["offset"]["iter"][str(
                iter)]["ego_speed_raster"]["sum"]
        assert int(np.sum(
            res_offset.data[:, :, 6])) == scene["offset"]["iter"][str(
                iter)]["drivable_area_raster"]["sum"]
        assert int(np.sum(
            res_offset.data[:, :, 7])) == scene["offset"]["iter"][str(
                iter)]["speed_limit_raster"]["sum"]
        assert int(np.mean(merge_image_offset)) == scene["offset"]["iter"][str(
            iter)]["merge_raster"]["mean"]
        assert int(np.std(merge_image_offset)) == scene["offset"]["iter"][str(
            iter)]["merge_raster"]["std"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
