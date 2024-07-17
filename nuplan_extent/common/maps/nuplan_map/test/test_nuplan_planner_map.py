from typing import Any, Dict

import pytest
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.utils.testing.nuplan_test import (NUPLAN_TEST_PLUGIN,
                                                     nuplan_test)
from nuplan.database.tests.nuplan_db_test_utils import get_test_maps_db
from nuplan_extent.common.maps.nuplan_map.nuplan_planner_map import \
    NuPlanPlannerMap

DEFAULT_NUPLAN_MAP = "us-nv-las-vegas-strip"

maps_db = get_test_maps_db()
planner_map = NuPlanPlannerMap(maps_db, DEFAULT_NUPLAN_MAP)


@nuplan_test(path='../route_planner/test/json/lane_route_planner.json')
def test_map_path_search(scene: Dict[str, Any]) -> None:
    """
    Test path search function.
    """
    expected_path = scene["xtr"]["expected_path"]
    origin_point = StateSE2(scene["markers"][0]["pose"][0],
                            scene["markers"][0]["pose"][1],
                            scene["markers"][0]["pose"][2])
    destination_point = StateSE2(scene["markers"][1]["pose"][0],
                                 scene["markers"][1]["pose"][1],
                                 scene["markers"][1]["pose"][2])
    route_path = planner_map.path_search(origin_point, destination_point)
    assert len(route_path) == len(expected_path)


@nuplan_test(path='../route_planner/test/json/lane_route_planner.json')
def test_map_block_path_search(scene: Dict[str, Any]) -> None:
    """
    Test block path search function
    """
    expected_block_list = scene["expected_block_list"]
    expert_poses = scene["expert_poses"]
    origin_point = StateSE2(expert_poses[0][0], expert_poses[0][1],
                            expert_poses[0][2])
    destination_point = StateSE2(expert_poses[-1][0], expert_poses[-1][1],
                                 expert_poses[-1][2])
    route_blocks = planner_map.block_path_search(origin_point,
                                                 destination_point)
    assert [b.id for b in route_blocks] == expected_block_list


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
