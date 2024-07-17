from typing import List, Optional

from dataclasses import dataclass
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData


@dataclass(frozen=True)
class HorizonPlannerInitialization:
    """
    This class represents required data to initialize a planner.
    """

    # The state which was achieved by expert driver in a scenario
    expert_goal_state: StateSE2
    route_roadblock_ids: List[str]  # Roadblock ids comprising goal route
    # The mission goal which commonly is not achievable in a single scenario
    mission_goal: StateSE2
    map_api: AbstractMap  # The API towards maps.
    scenario: AbstractScenario


@dataclass
class P3CPlannerInitialization:
    """
    This class represents required data to initialize a planner.
    """

    # The state which was achieved by expert driver in a scenario
    expert_goal_state: StateSE2
    # route_roadblock_ids: List[str]  # Roadblock ids comprising goal route
    # The mission goal which commonly is not achievable in a single scenario
    mission_goal: StateSE2
    map_api: AbstractMap  # The API towards maps.
    scenario: AbstractScenario


@dataclass(frozen=True)
class MonteCarloPlannerInput:
    """
    Input to a planner for which a trajectory should be computed.
    """

    iteration: SimulationIteration  # Iteration and time in a simulation progress
    history: SimulationHistoryBuffer  # Rolling buffer containing past observations and states.
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None  # The traffic light status data
    command: Optional[int] = None
