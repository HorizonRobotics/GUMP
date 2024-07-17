import os
from typing import Any, Dict

import pytest
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.test_utils import add_map_objects_to_scene
from nuplan.common.utils.testing.nuplan_test import (NUPLAN_TEST_PLUGIN,
                                                     nuplan_test)
from nuplan_extent.common.maps.wod_map.wod_map import WodMap

DEFAULT_WOD_SCENARIO = "bc4e214d063187ae"
scenario_path = os.path.dirname(os.path.abspath(__file__))
wod_map = WodMap(os.path.join(scenario_path, 'json/pkl'), DEFAULT_WOD_SCENARIO)


def test_update_map_range() -> None:
    initial_map_range = wod_map.map_range
    new_range = [0, 0, 2000, 3000]
    wod_map.update_map_range(new_range)
    updated_map_range = [
        min(new_range[0], initial_map_range[0]),
        max(new_range[2], initial_map_range[1]),
        min(new_range[1], initial_map_range[2]),
        max(new_range[3], initial_map_range[3])
    ]
    assert wod_map.map_range == updated_map_range


@nuplan_test(path='json/all_map_objects.json')
def test_get_proximal_map_objects(scene: Dict[str, Any]) -> None:
    """
    Test get_proximal_map_objects.
    """

    marker = scene["markers"][0]
    pose = marker["pose"]
    map_objects = wod_map.get_proximal_map_objects(
        Point2D(pose[0], pose[1]),
        100.0,
        [
            SemanticMapLayer.BASELINE_PATHS,
            SemanticMapLayer.STOP_SIGN,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.SPEED_BUMP,
            SemanticMapLayer.LANE,
            SemanticMapLayer.BOUNDARIES,
            SemanticMapLayer.STOP_LINE,
        ],
    )

    assert len(map_objects[SemanticMapLayer.BASELINE_PATHS]
               ) == scene["xtr"]["expected_num_baseline_paths"]
    assert len(map_objects[
        SemanticMapLayer.STOP_SIGN]) == scene["xtr"]["expected_num_stop_signs"]
    assert len(map_objects[
        SemanticMapLayer.CROSSWALK]) == scene["xtr"]["expected_num_crosswalks"]
    assert len(
        map_objects[SemanticMapLayer.
                    SPEED_BUMP]) == scene["xtr"]["expected_num_speed_bumps"]
    assert len(map_objects[SemanticMapLayer.
                           LANE]) == scene["xtr"]["expected_num_lanes"]
    assert len(
        map_objects[SemanticMapLayer.
                    BOUNDARIES]) == scene["xtr"]["expected_num_boundaries"]
    assert len(map_objects[
        SemanticMapLayer.STOP_LINE]) == scene["xtr"]["expected_num_stop_lines"]

    for layer, map_objects in map_objects.items():
        add_map_objects_to_scene(scene, map_objects, layer)


@nuplan_test(path='json/all_map_objects.json')
def test_get_map_object(scene: Dict[str, Any]) -> None:
    """
    Test get_map_object.
    """
    marker = scene["objects"][0]
    pose = marker["pose"]
    map_object = wod_map.get_map_object(marker["id"],
                                        SemanticMapLayer.STOP_SIGN)

    assert map_object is not None
    assert map_object.point.x == pose[0]
    assert map_object.point.y == pose[1]

    marker = scene["objects"][1]
    map_object = wod_map.get_map_object(marker["id"],
                                        SemanticMapLayer.BASELINE_PATHS)

    assert map_object is not None
    assert map_object.speed_limit_mps == marker["speed_limt"]
    assert map_object.length == marker["length"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
