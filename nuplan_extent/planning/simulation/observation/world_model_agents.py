from typing import cast, List, Dict, Optional
import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.convert import numpy_array_to_absolute_pose, numpy_array_to_absolute_velocity
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.simulation.observation.abstract_ml_agents import AbstractMLAgents
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import sort_dict
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan_extent.planning.simulation.planner.abstract_planner import HorizonPlannerInitialization


def _convert_prediction_to_predicted_trajectory(
        agent: TrackedObject, poses: List[StateSE2],
        xy_velocities: List[StateVector2D],
        step_interval_us: float) -> PredictedTrajectory:
    """
    Convert each agent predictions into a PredictedTrajectory.
    :param agent: The agent the predictions are for.
    :param poses: A list of poses that makes up the predictions
    :param xy_velocities: A list of velocities in world frame corresponding to each pose.
    :return: The predictions parsed into PredictedTrajectory.
    """
    waypoints = [Waypoint(TimePoint(0), agent.box, agent.velocity)]
    waypoints += [
        Waypoint(
            # step + 1 because the first Waypoint is the current state.
            TimePoint(int((step + 1) * step_interval_us)),
            OrientedBox.from_new_pose(agent.box, pose),
            velocity,
        ) for step, (pose, velocity) in enumerate(zip(poses, xy_velocities))
    ]
    return PredictedTrajectory(1.0, waypoints)


class WorldModelAgents(AbstractMLAgents):
    """
    Simulate agents based on World model.
    """

    def __init__(self, model: TorchModuleWrapper,
                 scenario: AbstractScenario) -> None:
        """
        Initializes the WorldModelAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        super().__init__(model, scenario)
        self.prediction_type = 'next_agents'
        self.selected_class = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE']

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        unique_agents = {
            tracked_object.track_token: tracked_object
            for tracked_object in
            self._scenario.initial_tracked_objects.tracked_objects
            if tracked_object.tracked_object_type in [
                TrackedObjectType.VEHICLE,
                TrackedObjectType.PEDESTRIAN,
                TrackedObjectType.BICYCLE,
            ]
        }
        # TODO: consider agents appearing in the future (not just the first frame)
        self._agents = sort_dict(unique_agents)

    @property
    def _ego_velocity_anchor_state(self) -> StateSE2:
        """
        Returns the ego's velocity state vector as an anchor state for transformation.
        :return: A StateSE2 representing ego's velocity state as an anchor state
        """
        ego_velocity = self._ego_anchor_state.dynamic_car_state.rear_axle_velocity_2d
        return StateSE2(ego_velocity.x, ego_velocity.y,
                        self._ego_anchor_state.rear_axle.heading)

    def update_observation(self, iteration: SimulationIteration,
                           next_iteration: SimulationIteration,
                           history: SimulationHistoryBuffer,
                           ego_state: Optional[StateSE2] = None
                           ) -> None:
        """Inherited, see superclass."""
        self.step_time = next_iteration.time_point - iteration.time_point
        # using current frame ego state instead of history.ego_states[-1]
        self._ego_anchor_state = history.ego_states[-1]
        next_global_pose = ego_state.rear_axle
        
        ego_to_global = self._ego_anchor_state.rear_axle.as_matrix()
        global_to_ego = np.linalg.inv(ego_to_global)
        next_relative_matrix = global_to_ego @ next_global_pose.as_matrix()
        next_relative_pose = StateSE2.from_matrix(next_relative_matrix)

        # Construct input features
        initialization = HorizonPlannerInitialization(
            mission_goal=self._scenario.get_mission_goal(),
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            map_api=self._scenario.map_api,
            scenario=self._scenario,
            expert_goal_state=self._scenario.get_expert_goal_state(),
        )
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(
            next_iteration.index)
        current_input = PlannerInput(next_iteration, history,
                                     traffic_light_data)
        features = self._model_loader.build_features(current_input,
                                                     initialization)

        # Infer model
        next_ego_state = np.array(
            [next_relative_pose.x,
             next_relative_pose.y,
             next_relative_pose.heading])
        features['next_ego_state'] = next_ego_state[None, None, :]
        predictions = self._infer_model(features)

        # Update observations
        self._update_observation_with_predictions(predictions)

    def _infer_model(self, features: FeaturesType) -> TargetsType:
        """Inherited, see superclass."""
        # Propagate model
        predictions = self._model_loader.infer(features)
        generic_agents_input = predictions['generic_agents'].agents

        # Extract trajectory prediction
        if self.prediction_type not in predictions:
            raise ValueError(
                f"Prediction does not have the output '{self.prediction_type}'"
            )

        predicted_generic_agents = predictions[self.prediction_type]
        agents_prediction = {}
        id_to_track_token = {}
        for type_idx, class_name in enumerate(self.selected_class):
            key = f'{class_name}.token_mapping'
            track_token_to_ids_per_class = generic_agents_input[key][0]
            id_to_track_token_per_class = {
                v: k
                for k, v in track_token_to_ids_per_class.items()
            }
            assert len(id_to_track_token_per_class) == len(
                track_token_to_ids_per_class
            ), f"Token mapping should have {len(track_token_to_ids_per_class)} elements, but has {len(track_token_to_ids_per_class)}"
            if type_idx not in id_to_track_token:
                id_to_track_token[type_idx] = {}
            id_to_track_token[type_idx].update(id_to_track_token_per_class)

        for predicted_agent in predicted_generic_agents:
            track_id = predicted_agent.track_id
            type_idx = predicted_agent.type_idx
            # skip killed agents
            if not predicted_agent._within_range():
                continue
            id_to_track_token_per_type = id_to_track_token[type_idx]
            assert track_id in id_to_track_token_per_type, f"Track id {track_id} not found in token mapping"
            agent_trajectory = np.array([
                predicted_agent.x,
                predicted_agent.y,
                predicted_agent.heading,
                predicted_agent.vx,
                predicted_agent.vy,
            ])
            track_token = id_to_track_token_per_type[track_id]
            agents_prediction[track_token] = agent_trajectory

        return agents_prediction

    def _update_observation_with_predictions(self,
                                             predictions: TargetsType) -> None:
        """Inherited, see superclass."""
        assert self._agents, "The agents have not been initialized. Please make sure they are initialized!"
        count = 0
        for agent_token, agent_prediction in predictions.items():
            # if agent token not in self._agents, skip
            if agent_token not in self._agents:
                count += 1
                continue
            poses_horizon = agent_prediction[:3][None, :]
            xy_velocity_horizon = agent_prediction[3:5][None, :]

            agent_meta = self._agents[agent_token]
            poses = numpy_array_to_absolute_pose(
                self._ego_anchor_state.rear_axle, poses_horizon)
            xy_velocities = numpy_array_to_absolute_velocity(
                self._ego_velocity_anchor_state, xy_velocity_horizon)
            future_trajectory = _convert_prediction_to_predicted_trajectory(
                agent_meta, poses, xy_velocities, self._step_interval_us)

            # Propagate agent according to simulation time
            new_state = future_trajectory.trajectory.get_state_at_time(
                self.step_time)

            new_agent = Agent(
                tracked_object_type=agent_meta.tracked_object_type,
                oriented_box=new_state.oriented_box,
                velocity=new_state.velocity,
                metadata=agent_meta.metadata,
            )
            new_agent.predictions = [future_trajectory]

            self._agents[agent_token] = new_agent
        if count > 0:
            print(f"num of agents not in self._agents: {count}")
