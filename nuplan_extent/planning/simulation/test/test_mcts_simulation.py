import unittest

from collections import deque
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import \
    MockAbstractScenario
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.perfect_tracking import \
    PerfectTrackingController
from nuplan.planning.simulation.observation.tracks_observation import \
    TracksObservation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan_extent.planning.simulation.simulation_time_controller.step_simulation_with_prev_time_controller import \
    StepSimulationWithPrevTimeController
from nuplan_extent.planning.simulation.mcts_simulation import \
    MonteCarloSimulation, ParentSwitchMode, SampleNode
from nuplan_extent.planning.simulation.mcts.branch_splitter.multi_mode_splitter import \
    MultiModeSplitter
from nuplan_extent.planning.simulation.history.simulation_tree_history import SimulationTreeHistorySample
from nuplan_extent.common.actor_state.vehicle_parameters import get_ideal_one_parameters


class TestMCTSSimulation(unittest.TestCase):
    """
    Tests Simulation class which is updating simulation
    """

    def setUp(self) -> None:
        """Setup Mock classes."""
        self.scenario = MockAbstractScenario(number_of_past_iterations=10)

        self.sim_manager = StepSimulationWithPrevTimeController(self.scenario)
        self.observation = TracksObservation(self.scenario)
        self.controller = PerfectTrackingController(self.scenario)
        self.branch_splitter = MultiModeSplitter(branch_num=3,
                                                 work_steps=10,
                                                 default_command=0)

        self.setup = SimulationSetup(
            time_controller=self.sim_manager,
            observations=self.observation,
            ego_controller=self.controller,
            scenario=self.scenario,
        )
        self.simulation_history_buffer_duration = 2
        self.stepper = MonteCarloSimulation(
            simulation_setup=self.setup,
            callback=MultiCallback([]),
            simulation_history_buffer_duration=self.
            simulation_history_buffer_duration,
            metric_aggregator_cfg=None,
            branch_splitter=self.branch_splitter,
        )

    def mock_history_sample(self,
                            parent_idx: int) -> SimulationTreeHistorySample:
        """Mock history sample."""
        assert parent_idx >= 0, "parent_idx must be >= 0"
        ego_state = EgoState.build_from_center(
            StateSE2(parent_idx, parent_idx, 0), StateVector2D(0, 0),
            StateVector2D(0, 0), None, None, get_ideal_one_parameters())
        if parent_idx == 0:
            return SimulationTreeHistorySample(
                iteration=SimulationIteration(parent_idx, parent_idx),
                ego_state=ego_state,
                trajectory=None,
                observation=None,
                traffic_light_status=None,
                command=0,
            )
        else:
            return SimulationTreeHistorySample(
                iteration=SimulationIteration(parent_idx, parent_idx),
                ego_state=ego_state,
                trajectory=None,
                observation=None,
                traffic_light_status=None,
                command=0,
                parent=self.mock_history_sample(parent_idx - 1),
            )

    def test_stepper_initialize(self) -> None:
        """Test initialization method."""
        initialization = self.stepper.initialize()
        self.assertEqual(initialization.mission_goal,
                         self.scenario.get_mission_goal())

        # Check current ego state
        self.assertEqual(
            self.stepper._history_buffer.current_state[0].rear_axle,
            self.scenario.get_ego_state_at_iteration(0).rear_axle,
        )

    def test_check_score_dict(self) -> None:
        """Test check_score_dict method."""
        score_dict = {
            'no_ego_at_fault_collisions': 1,
            'drivable_area_compliance': 0,
            'driving_direction_compliance': 1
        }
        self.assertEqual(self.stepper.check_score_dict(score_dict), False)
        score_dict = {
            'no_ego_at_fault_collisions': 0,
            'drivable_area_compliance': 1,
            'driving_direction_compliance': 1
        }
        self.assertEqual(self.stepper.check_score_dict(score_dict), False)
        score_dict = {
            'no_ego_at_fault_collisions': 1,
            'drivable_area_compliance': 1,
            'driving_direction_compliance': 0
        }
        self.assertEqual(self.stepper.check_score_dict(score_dict), False)
        score_dict = {
            'no_ego_at_fault_collisions': 1,
            'drivable_area_compliance': 1,
            'driving_direction_compliance': 1
        }
        self.assertEqual(self.stepper.check_score_dict(score_dict), True)

    def test_check_traj_redundancy(self) -> None:
        """Test check_traj_redundancy method."""
        self.stepper.initialize()
        self.stepper._prev_history_traj = [StateSE2(x, x, 0) for x in range(3)]
        mock_history_sample = self.mock_history_sample(2)
        self.assertEqual(
            self.stepper.check_traj_redundancy(mock_history_sample), True)
        self.stepper._prev_history_traj = [StateSE2(0, 0, 0) for x in range(3)]
        self.assertEqual(
            self.stepper.check_traj_redundancy(mock_history_sample), False)

    def test_refill_history_buffer_no_error(self) -> None:
        """Test refill_history_buffer method."""
        self.stepper.initialize()
        self.stepper._refill_history_buffer(None, ParentSwitchMode.PARALLEL)

    def test_split_branch(self) -> None:
        """Test split_branch method."""
        self.stepper.initialize()
        sample_node = SampleNode(0, None, None)
        queue_size = len(self.stepper._queue)
        self.stepper._split_branch(sample_node, None)
        self.assertLess(queue_size, len(self.stepper._queue))

    def test_reset(self) -> None:
        """Test reset method."""
        self.stepper.initialize()
        self.stepper.reset()
        self.assertEqual(self.stepper._queue, deque([]))
        self.assertEqual(self.stepper._history.data, {0: []})


if __name__ == '__main__':
    unittest.main(verbosity=2)
