import os
import unittest
from pathlib import Path

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan_extent.planning.scenario_builder.wod_db.wod_scenario import \
    WodScenario


class TestWodScenario(unittest.TestCase):
    def setUp(self):
        self.data_root = (
            Path(__file__).resolve().parent.parent.parent.parent.parent /
            'common/maps/wod_map/test/json').resolve()
        self.split = "pkl"
        self.scenario_id = "bc4e214d063187ae"
        self.agent_idx = "36"
        self.wod_scenario = WodScenario(self.data_root, self.split,
                                        self.scenario_id, self.agent_idx)

    def test_init(self):
        """
        Test the initialization of the WodScenario class.
        """
        self.assertEqual(self.wod_scenario._data_root, self.data_root)
        self.assertEqual(self.wod_scenario._split, self.split)
        self.assertEqual(self.wod_scenario._scenario_id, self.scenario_id)
        self.assertEqual(self.wod_scenario._agent_idx, int(self.agent_idx))
        self.assertEqual(self.wod_scenario._scenario_type,
                         "waymo_open_dataset")
        self.assertEqual(
            self.wod_scenario._pickle_path,
            os.path.join(self.data_root, self.split,
                         self.scenario_id + ".pkl"))
        self.assertEqual(self.wod_scenario.current_index, 10)
        self.assertEqual(self.wod_scenario.scenario_length, 91)
        self.assertEqual(self.wod_scenario._database_row_interval, 0.1)

    def test_transform_state_to_EgoState(self):
        """Test the transform_state_to_EgoState function.
        """
        all_tracks = self.wod_scenario.load_agent_tracks()
        iteration = 5
        state = all_tracks[int(self.agent_idx)].states[iteration]

        ego_state = self.wod_scenario.transform_state_to_EgoState(
            state, iteration)

        self.assertAlmostEqual(ego_state.waypoint.x, -989.3889948016736)
        self.assertAlmostEqual(ego_state.waypoint.y, -3453.7120194642425)
        self.assertAlmostEqual(ego_state.waypoint.heading, 1.5519533157348633)
        self.assertAlmostEqual(ego_state.waypoint.velocity.x,
                               0.0718463882803917)
        self.assertAlmostEqual(ego_state.waypoint.velocity.y,
                               2.9960923194885254)
        self.assertEqual(ego_state.time_point, TimePoint(iteration * 1e5))

    def test_transform_state_to_TrackedObject(self):
        """Test the transform_state_to_TrackedObject function.
        """
        all_tracks = self.wod_scenario.load_agent_tracks()
        iteration = 5
        state = all_tracks[int(self.agent_idx)].states[iteration]

        object_type = 1
        agent = self.wod_scenario.transform_state_to_TrackedObject(
            state, object_type, iteration, int(self.agent_idx))

        self.assertEqual(agent.tracked_object_type.name, 'VEHICLE')
        self.assertAlmostEqual(agent.velocity.x, 0.0718463882803917)
        self.assertAlmostEqual(agent.velocity.y, 2.9960923194885254)
        self.assertAlmostEqual(agent.box.width, 2.3320000171661377)
        self.assertAlmostEqual(agent.box.length, 5.285999774932861)
        self.assertAlmostEqual(agent.box.height, 2.3299999237060547)
        self.assertAlmostEqual(agent.box.center.x, -989.4138943661987)
        self.assertAlmostEqual(agent.box.center.y, -3455.0332848096496)
        self.assertAlmostEqual(agent.box.center.heading, 1.5519533157348633)


if __name__ == "__main__":
    unittest.main()
