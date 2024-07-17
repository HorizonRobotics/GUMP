import numpy as np

import pytest
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.utils.testing.nuplan_test import NUPLAN_TEST_PLUGIN
from nuplan_extent.common.maps.wod_map.utils import extract_discrete_baseline
from shapely.geometry.polygon import LineString


def test_extract_discrete_baseline():
    path1 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
    path2 = LineString([(0, 0), (0, 1), (1, 1), (1, 0)])

    # check the expected StateSE2 objects for each path
    assert extract_discrete_baseline(path1) == [
        StateSE2(0, 0, 0),
        StateSE2(1, 0, np.pi / 2),
        StateSE2(1, 1, np.pi),
        StateSE2(0, 1, np.pi)
    ]
    assert extract_discrete_baseline(path2) == [
        StateSE2(0, 0, np.pi / 2),
        StateSE2(0, 1, 0),
        StateSE2(1, 1, -np.pi / 2),
        StateSE2(1, 0, -np.pi / 2)
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__], plugins=[NUPLAN_TEST_PLUGIN]))
