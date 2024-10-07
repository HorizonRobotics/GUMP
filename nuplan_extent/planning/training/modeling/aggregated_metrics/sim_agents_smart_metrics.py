from typing import List
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import pickle

from torchmetrics import Metric
from collections import defaultdict
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils
from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan_extent.planning.training.modeling.models.tokenizers.kinetics_tokenizer_utils import NpKineticsSequenceArray

import torch
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2, sim_agents_metrics_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import estimators

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils import trajectory_utils


def smooth_window(arr, window_size):
    """
    Smooth a 1D array using a simple moving average.
    :param arr: 1D array of values (e.g., x, y, or heading).
    :param window_size: Size of the moving average window.
    :return: Smoothed array.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    window = np.ones(int(window_size)) / float(window_size)
    res = np.convolve(arr, window, 'valid')
    return np.concatenate([arr[:int(window_size/2)], res, arr[-int(window_size/2):]])

class SimAgentsMetric(Metric):
    """
    Metric representing the probability density of GT agents trajectories over the distribution predicted by model
    """

    def __init__(self, 
                 name: str = 'sim_agents_metric',
                 basepath: str = '/tmp'
                 ) -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        super(SimAgentsMetric, self).__init__()
        self._name = name  
        self.basepath = basepath 
        self.all_saved_scenario_paths = []
        self.add_state("all_results", default=[], dist_reduce_fx=None)

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["sim_agents"]
    
    @staticmethod
    def joint_scene_from_states(
            states: tf.Tensor, object_ids: tf.Tensor
            ) -> sim_agents_submission_pb2.JointScene:
        states = states.numpy()
        simulated_trajectories = []
        for i_object in range(len(object_ids)):
            simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
                center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
                object_id=int(object_ids[i_object])
            ))
        return sim_agents_submission_pb2.JointScene(
            simulated_trajectories=simulated_trajectories)

    @staticmethod
    def extract_map_points(waymo_scenario):
        map_points = []
        for map_point in waymo_scenario.map_features:
            for polyline in map_point.road_line.polyline:
                map_points.append([polyline.x, polyline.y, polyline.z])
        map_points = np.array(map_points)
        return map_points

    @staticmethod
    def scenario_rollouts_from_states(
            scenario: scenario_pb2.Scenario, 
            states: tf.Tensor, object_ids: tf.Tensor,
            ) -> sim_agents_submission_pb2.ScenarioRollouts:
        joint_scenes = []
        for i_rollout in range(states.shape[0]):
            joint_scenes.append(SimAgentsMetric.joint_scene_from_states(states[i_rollout], object_ids))
        
        return sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)


    def update(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """     
        # batch of predictions, 
        # each prediction contains 32 parallel rollouts, 
        # each rollout contains a prediction of agents trajectories 8s into future (16 frames @ 2Hz)

        scenarioid2batchindex = defaultdict(list)
        for i in range(len(predictions['scenario_id'])):
            scenarioid2batchindex[predictions['scenario_id'][i][0]].append(i)
        bs = len(targets['sim_agents'].scenario_id)

        # batch of targets
        # from third_party.functions.forked_pdb import ForkedPdb; ForkedPdb().set_trace()
        waymo_scenario_paths = targets['sim_agents'].pkl_path
        local_to_global_transforms = targets['sim_agents'].local_to_global_transform

        # update part
        for bi in range(bs):            
            # process GT part
            with open(waymo_scenario_paths[bi], 'rb') as f:
                waymo_scenario = pickle.load(f)

            basepath = self.basepath
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            target_agent_idx = int(targets['sim_agents'].agent_idx[bi])
            scene_path = os.path.join(basepath, '{}_{}_rollouts.pkl'.format(waymo_scenario.scenario_id, target_agent_idx))
            
            self.all_saved_scenario_paths.append(scene_path)
            # process pred part
            batch_index_list = scenarioid2batchindex[waymo_scenario.scenario_id]
            batch_mask = torch.zeros(predictions['batch_index'].shape, dtype=torch.bool)
            for i in range(batch_mask.shape[0]):
                batch_mask[i] = predictions['batch_index'][i] in batch_index_list
            parallel_rollouts = torch.cat([predictions['pred_traj'], predictions['pred_head'][..., None]], dim=-1)[batch_mask]
            parallel_rollouts = parallel_rollouts.view(len(batch_index_list), -1, parallel_rollouts.shape[-2], parallel_rollouts.shape[-1])
            pred_agent_ids = torch.tensor(predictions['pred_agent_ids'][batch_index_list[0]][0], dtype=torch.int32)
            result = self._compute_single_frame(parallel_rollouts, waymo_scenario, local_to_global_transforms[bi], pred_agent_ids)
            self.all_results.append(result)
        return
        
    def _compute_single_frame(self, parallel_rollouts, waymo_scenario, local_to_global_transforms, pred_agent_ids):
        logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(waymo_scenario)
        target_agent_ids = submission_specs.get_sim_agent_ids(waymo_scenario)
        logged_trajectories = logged_trajectories.gather_objects_by_id(
            tf.convert_to_tensor(target_agent_ids))
        logged_trajectories = logged_trajectories.slice_time(
            start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
        logged_trajectories_x = logged_trajectories.x.numpy()
        logged_trajectories_y = logged_trajectories.y.numpy()
        logged_trajectories_z = logged_trajectories.z.numpy()
        logged_trajectories_heading = logged_trajectories.heading.numpy()
        logged_trajectories_states = np.stack([
            logged_trajectories_x, 
            logged_trajectories_y, 
            logged_trajectories_z, 
            logged_trajectories_heading],
            axis=-1)
        logged_trajectories_valid = logged_trajectories.valid.numpy()
        map_points = SimAgentsMetric.extract_map_points(waymo_scenario)
        zvalue_regressor = KNeighborsRegressor(n_neighbors=4)
        try:
            zvalue_regressor.fit(map_points[:, :2], map_points[:, 2])
        except:
            pseudo_map_points = np.vstack([
                logged_trajectories_x.ravel(), 
                logged_trajectories_y.ravel(),
                logged_trajectories_z.ravel(),
                ]).T
            zvalue_regressor.fit(pseudo_map_points[:, :2], pseudo_map_points[:, 2])
        # import pdb; pdb.set_trace()
        interpolated_predicted_trajectories = SimAgentsMetric.extract_predicted_trajectories(parallel_rollouts.cpu(), 
                                                                                             target_agent_ids, 
                                                                                             pred_agent_ids, 
                                                                                             local_to_global_transforms, 
                                                                                             zvalue_regressor, 
                                                                                             logged_trajectories_states, 
                                                                                             logged_trajectories_valid)
        simulated_states = tf.convert_to_tensor(interpolated_predicted_trajectories[..., :4])
        scenario_rollouts = SimAgentsMetric.scenario_rollouts_from_states(
            waymo_scenario, simulated_states, logged_trajectories.object_id)
        submission_specs.validate_scenario_rollouts(scenario_rollouts, waymo_scenario)
        config = metrics.load_metrics_config()
        import time
        start = time.time()
        scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
            config, waymo_scenario, scenario_rollouts)
        print("finished eval:", time.time() - start)
        scenario_result = {
            'scenario_id': scenario_metrics.scenario_id,
            'metametric': scenario_metrics.metametric,
            'average_displacement_error': scenario_metrics.average_displacement_error,
            'linear_speed_likelihood': scenario_metrics.linear_speed_likelihood,
            'linear_acceleration_likelihood': scenario_metrics.linear_acceleration_likelihood,
            'angular_speed_likelihood': scenario_metrics.angular_speed_likelihood,
            'angular_acceleration_likelihood': scenario_metrics.angular_acceleration_likelihood,
            'distance_to_nearest_object_likelihood': scenario_metrics.distance_to_nearest_object_likelihood,
            'collision_indication_likelihood': scenario_metrics.collision_indication_likelihood,
            'time_to_collision_likelihood': scenario_metrics.time_to_collision_likelihood,
            'distance_to_road_edge_likelihood': scenario_metrics.distance_to_road_edge_likelihood,
            'offroad_indication_likelihood': scenario_metrics.offroad_indication_likelihood,
            'simulated_collision_rate': scenario_metrics.simulated_collision_rate,
            'simulated_offroad_rate': scenario_metrics.simulated_offroad_rate,
            'min_average_displacement_error': scenario_metrics.min_average_displacement_error,
        }
        return scenario_result
    
    @staticmethod
    def extract_predicted_trajectories(
        parallel_rollouts,
        target_agent_ids,
        pred_agent_ids,
        local_to_global_transforms,
        zvalue_regressor,
        logged_trajectories_states,
        logged_trajectories_valid_flags,
    ):
        """
        Extract and interpolate predicted trajectories for multiple agents across multiple worlds.

        Parameters:
        - parallel_rollouts (list): Rollout data for each world.
        - agent_ids (list): List of agent IDs to process.
        - local_to_global_transforms (np.ndarray): Transformation matrices from local to global coordinates.
        - zvalue_regressor (object): Regressor to compute z-values based on positions.
        - logged_trajectories_states (list): Logged states of trajectories.
        - logged_trajectories_valid_flags (list): Validity flags for logged trajectories.

        Returns:
        - interpolated_predicted_trajectories (np.ndarray): 
            Interpolated predicted trajectories (shape: [N_worlds, N_agents, N_steps, N_dims]).
        """
        N_worlds = parallel_rollouts.shape[0]
        N_agents = len(target_agent_ids)
        N_steps = parallel_rollouts.shape[-2]
        N_dims = 4

        interpolated_predicted_trajectories = np.zeros((N_worlds, N_agents, N_steps, N_dims))
        
        for world_idx in range(N_worlds):

            # Extract raw agent IDs from the tokenized data
            rawid_to_index_map = {int(raw_id): idx for idx, raw_id in enumerate(pred_agent_ids)}
            
            missing_agents = [agent_id for agent_id in target_agent_ids if agent_id not in rawid_to_index_map]
            if len(missing_agents) > 0:
                print(f"Missing agents: {len(missing_agents)}/ {len(agent_ids)}")
            for agent_idx, agent_id in enumerate(target_agent_ids):
                if agent_id not in rawid_to_index_map:
                    # Handle missing agents if necessary
                    continue

                token_index = rawid_to_index_map[agent_id]

                # Extract position and heading information for the agent
                x_positions = parallel_rollouts[world_idx, token_index, :, 0]
                y_positions = parallel_rollouts[world_idx, token_index, :, 1]
                positions_local = np.stack([x_positions, y_positions], axis=-1)
                headings = parallel_rollouts[world_idx, token_index, :, 2] % (2 * np.pi)

                # Apply local to global transformation
                # transformation_matrix = local_to_global_transforms.numpy()
                # positions_global = (transformation_matrix[:2, :2] @ positions_local.T).T + transformation_matrix[:2, -1]

                # Adjust headings based on rotation from the transformation matrix
                # rotation_angle = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
                # adjusted_headings = headings + rotation_angle

                positions_global = positions_local # no need to transform for smart
                adjusted_headings = headings
                # Compute z-values using the regressor
                z_values = zvalue_regressor.predict(positions_global)

                # Normalize z-values based on logged trajectories
                assert logged_trajectories_states[agent_idx].shape[0] == 11, "Logged trajectories must have 11 states."
                logged_z = logged_trajectories_states[agent_idx][10, 2]
                reference_z = z_values[0]
                normalized_z = z_values - reference_z + logged_z

                # Combine position, normalized z, and adjusted heading into predicted values
                predicted_values = np.stack([
                    positions_global[:, 0],
                    positions_global[:, 1],
                    normalized_z,
                    adjusted_headings
                ], axis=-1)
                interpolated_predicted_trajectories[world_idx, agent_idx] = predicted_values
        return interpolated_predicted_trajectories

    
    def compute(self) -> torch.Tensor:
        """
        Computes the metric.

        :return: metric scalar tensor
        """
        # Gather all_results from all nodes
        gathered_results: List[List[Dict[str, float]]] = [ [] for _ in range(dist.get_world_size()) ]
        dist.barrier()
        dist.all_gather_object(gathered_results, self.all_results)
        if dist.get_rank() == 0:
            aggregated: Dict[str, float] = {}
            counts: Dict[str, int] = {}

            # Iterate over results from all nodes
            for node_results in gathered_results:
                for res in node_results:
                    for key, value in res.items():
                        if isinstance(value, str):
                            continue
                        if key in aggregated:
                            aggregated[key] += value
                            counts[key] += 1
                        else:
                            aggregated[key] = value
                            counts[key] = 1
            # Compute the average for each key
            averaged_metrics = {key: aggregated[key] / counts[key] for key in aggregated.keys()}
            # Print the aggregated results
            print("Aggregated Metrics:", averaged_metrics)

            return averaged_metrics
        else:
            return {}         
    
    def to(self, device):
        return self

    def log(self, logger, data, global_step):
        if dist.get_rank() == 0:
            for k, v in data.items():
                # Use the logger's experiment to log the scalar
                logger.experiment.add_scalar(f'sim_agents/{k}', v, global_step=global_step)