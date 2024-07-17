from absl.testing import parameterized

import numpy as np
import os
import unittest

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_scenarios_from_db,
    get_lidarpc_token_timestamp_from_db,
    get_lidarpc_token_map_name_from_db,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import NUPLAN_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION

from nuplan_extent.planning.scenario_builder.prepared_scenario import PreparedScenario
from nuplan_extent.planning.simulation.progress_reward import ProgressReward, _coords_length


def get_test_scenarios():
    scenario_extraction_info = ScenarioExtractionInfo(
        DEFAULT_SCENARIO_NAME,
        scenario_duration=15.0,
        extraction_offset=-2.0,
        subsample_ratio=1.0)

    db_file_root = NUPLAN_DATA_ROOT + "/nuplan-v1.1/mini/"
    # list of (db_file, token)
    db_files = [
        # ("2021.08.27.16.46.47_veh-45_01810_02137.db", "847841ed44f050be"), # in full nuplan dataset
        ("2021.10.05.07.10.04_veh-52_01442_01802.db", "609d35101d615769"),
        ("2021.08.17.16.57.11_veh-08_01200_01636.db",
         "c2f034a294dc5a89"),  # very skewed lane
        ("2021.07.16.18.06.21_veh-38_04933_05307.db",
         None),  # two lanes in one roadblock crossing
        ("2021.06.09.11.54.15_veh-12_04366_04810.db", None),  # not connected
        ("2021.10.11.02.57.41_veh-50_01522_02088.db", None),  # loop
        ("2021.10.11.02.57.41_veh-50_01522_02088.db",
         "b371212fa2785ad5"),  # self-crossing route, complex roadblock shape
        ("2021.05.12.22.00.38_veh-35_01008_01518.db", None),
        ("2021.05.12.22.28.35_veh-35_00620_01164.db", None),
        ("2021.05.12.23.36.44_veh-35_00152_00504.db", None),
        ("2021.10.11.08.31.07_veh-50_01750_01948.db", None),
        ("2021.06.09.14.58.55_veh-35_01095_01484.db", None),
        ("2021.06.09.14.58.55_veh-35_01894_02311.db", None),
        ("2021.06.28.16.29.11_veh-38_01415_01821.db", None),
        ("2021.06.28.16.29.11_veh-38_03263_03766.db", None),
        ("2021.08.17.18.54.02_veh-45_00665_01065.db", None),
        ("2021.07.24.20.37.45_veh-17_00015_00375.db", None),
    ]

    scenarios = []
    for f, token in db_files:
        db_file = db_file_root + f
        assert os.path.exists(db_file), f"cannot find {db_file}"
        rows = list(
            get_scenarios_from_db(
                db_file,
                filter_tokens=None,
                filter_types=None,
                filter_map_names=None,
                include_invalid_mission_goals=False))
        if token is None:
            token = rows[0]['token'].hex()

        timestamp = get_lidarpc_token_timestamp_from_db(db_file, token)
        map_name = get_lidarpc_token_map_name_from_db(db_file, token)

        scenario = NuPlanScenario(
            data_root=NUPLAN_DATA_ROOT,
            log_file_load_path=db_file,
            initial_lidar_token=token,
            initial_lidar_timestamp=timestamp,
            scenario_type=DEFAULT_SCENARIO_NAME,
            map_root=NUPLAN_MAPS_ROOT,
            map_version=NUPLAN_MAP_VERSION,
            map_name=map_name,
            scenario_extraction_info=scenario_extraction_info,
            ego_vehicle_parameters=get_pacifica_parameters(),
        )
        scenarios.append(scenario)
    return scenarios


class ProgressRewardTest(parameterized.TestCase, unittest.TestCase):
    """Test rewards."""

    def setUp(self) -> None:
        self.scenarios = get_test_scenarios()

    @parameterized.parameters(
        (False, 2.0),
        (True, 5.0),
    )
    def test_progress_rewards(self, sparse=False, delta=2.):
        """Check the distance traveled by the ego vehicle is approximately equal
        to the sum of the progress reward.
        """
        batch_size = len(self.scenarios)
        rewarder = ProgressReward(batch_size, weight=1.0, sparse=sparse)
        for i, scenario in enumerate(self.scenarios):
            prepared_scenario = PreparedScenario()
            iterations = range(0, scenario.get_number_of_iterations(), 2)
            prepared_scenario.prepare_scenario(scenario, iterations)
            rewarder.prepare_scenario(scenario, prepared_scenario, iterations)
            self.assertIsNotNone(prepared_scenario.get_feature("waypoints"))
            rewarder.new_scenario(i, prepared_scenario, start_iteration=0)
            prepared_scenario.backup_feature("ego_state")

        iterations = range(0, 300, 2)
        total_reward = np.zeros(batch_size)
        for iteration in iterations:
            reward, status, info = rewarder.compute(None,
                                                    [iteration] * batch_size)
            total_reward += reward
        for i, scenario in enumerate(self.scenarios):
            locations = [
                scenario.get_ego_state_at_iteration(iter).rear_axle.point
                for iter in iterations
            ]
            locations = [(p.x, p.y) for p in locations]
            distance = _coords_length(locations)
            self.assertAlmostEqual(distance, total_reward[i], delta=delta)


if __name__ == '__main__':
    unittest.main()
