import glob
import math
import os
import multiprocessing
from tqdm import tqdm
import pickle
import argparse

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Helper functions for smoothing and outlier handling

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

# Metric class
class SimAgentsMetric:
    def __init__(self, name: str = 'sim_agents_metric'):
        self._name = name
    
    @staticmethod
    def joint_scene_from_states(states: tf.Tensor, object_ids: tf.Tensor) -> sim_agents_submission_pb2.JointScene:
        states = states.numpy()
        simulated_trajectories = []
        for i_object in range(len(object_ids)):
            simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
                center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
                object_id=int(object_ids[i_object])
            ))
        return sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)

    @staticmethod
    def scenario_rollouts_from_states(scenario: scenario_pb2.Scenario, states: tf.Tensor, object_ids: tf.Tensor) -> sim_agents_submission_pb2.ScenarioRollouts:
        joint_scenes = []
        for i_rollout in range(states.shape[0]):
            joint_scenes.append(SimAgentsMetric.joint_scene_from_states(states[i_rollout], object_ids))
        return sim_agents_submission_pb2.ScenarioRollouts(joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)

    @staticmethod
    def linear_interp_and_expand_any_freq(hv_10Hz, h_valid_10Hz, pv_nHz, p_valid_nHz, exp_ratio):
        assert h_valid_10Hz[-1]
        l_final = exp_ratio + 1
        if p_valid_nHz:
            interp_out = np.zeros((l_final, 4))
            interp_out[:, :3] = np.linspace(hv_10Hz[-1, :3], pv_nHz[:3], l_final)
            theta1, theta2 = hv_10Hz[-1, 3], pv_nHz[3]
            V1 = np.array([np.cos(theta1), np.sin(theta1)])
            V2 = np.array([np.cos(theta2), np.sin(theta2)])
            V_interp = np.linspace(V1, V2, l_final)
            V_norm = V_interp / np.linalg.norm(V_interp, axis=1, keepdims=True)
            theta_interp = np.arctan2(V_norm[:, 1], V_norm[:, 0])
            interp_out[:, 3] = theta_interp
            interp_out = interp_out[1:, :]
            new_interp_hf = np.zeros((exp_ratio)).astype(bool)
        else:
            if h_valid_10Hz[-2]:
                v = hv_10Hz[-1, :] - hv_10Hz[-2, :]
            else:
                v = 0
            interp_out = np.zeros((exp_ratio, 4))
            for dt in range(exp_ratio):
                interp_out[dt, :] = hv_10Hz[-1, :] + v * (dt + 1)
            new_interp_hf = np.ones((exp_ratio)).astype(bool)
        return interp_out, new_interp_hf

    @staticmethod
    def extract_map_points(waymo_scenario):
        map_points = []
        for map_point in waymo_scenario.map_features:
            for polyline in map_point.road_line.polyline:
                map_points.append([polyline.x, polyline.y, polyline.z])
        map_points = np.array(map_points)
        return map_points

    @staticmethod
    def compute_single_speed_interp_any_freq_smoothing(path, pred_freq=2, return_metrics=True, query_agent=-1):
        pred_path, output_dir = path
        with open(pred_path, 'rb') as f:
            sim_agents_rollouts = pickle.load(f)
            all_lf_values = sim_agents_rollouts['values']
            all_lf_valid = sim_agents_rollouts['valid']
            waymo_scenario_paths = sim_agents_rollouts['waymo_scenario_paths']
            local_to_global_transforms = sim_agents_rollouts['transform']
        with open(waymo_scenario_paths, 'rb') as f:
            waymo_scenario = pickle.load(f)
        evaluated_sim_agent_ids = tf.convert_to_tensor(
            submission_specs.get_evaluation_sim_agent_ids(waymo_scenario)
        )
        logged_trajectories = trajectory_utils.ObjectTrajectories.from_scenario(waymo_scenario)
        agent_ids = submission_specs.get_sim_agent_ids(waymo_scenario)
        logged_trajectories = logged_trajectories.gather_objects_by_id(
            tf.convert_to_tensor(agent_ids))
        logged_trajectories = logged_trajectories.slice_time(
            start_index=0, end_index=submission_specs.CURRENT_TIME_INDEX + 1)
        logged_trajectories_x = logged_trajectories.x.numpy()
        logged_trajectories_y = logged_trajectories.y.numpy()
        logged_trajectories_z = logged_trajectories.z.numpy()
        logged_trajectories_heading = logged_trajectories.heading.numpy()
        logged_trajectories_states = np.stack([
            logged_trajectories_x, logged_trajectories_y, logged_trajectories_z, logged_trajectories_heading],
            axis=-1)
        logged_trajectories_valid = logged_trajectories.valid.numpy()
        map_points = SimAgentsMetric.extract_map_points(waymo_scenario)
        zvalue_regressor = KNeighborsRegressor(n_neighbors=4)
        try:
            zvalue_regressor.fit(map_points[:, :2], map_points[:, 2])
        except:
            psudo_map_points = np.vstack([
                logged_trajectories_x.ravel(), 
                logged_trajectories_y.ravel(),
                logged_trajectories_z.ravel(),
                ]).T
            zvalue_regressor.fit(psudo_map_points[:, :2], psudo_map_points[:, 2])
        N_worlds, N_agents, N_steps, N_dims = 32, len(agent_ids), 80, 4
        N_pred_frames = 8 * pred_freq + 1
        preds = np.zeros((N_worlds, N_agents, N_steps, N_dims))
        is_interp = np.zeros((N_worlds, N_agents, N_steps)).astype(bool)
        missing_agent = []
        for ri in range(N_worlds):
            missing_agent_ri = []
            for agent_i, _ in enumerate(agent_ids):
                values = all_lf_values[ri][agent_i].numpy()
                values[:, 3] = values[:, 3] % (2 * np.pi)
                valid = all_lf_valid[ri][agent_i].numpy().astype(bool)
                missing_agent_ri.append(~valid[0])
                l2g_mat = local_to_global_transforms.numpy()
                xys = values[:, :2].reshape(-1, 2).T
                xys = xys.astype(l2g_mat.dtype)
                xys = l2g_mat[:2, :2] @ xys + l2g_mat[:2, -1:]
                values[:, :2] = xys.T
                theta = np.arctan2(l2g_mat[1, 0], l2g_mat[0, 0])
                values[:, 3] += theta
                z_values = zvalue_regressor.predict(values[:, :2])
                assert logged_trajectories_states[agent_i].shape[0] == 11
                logz = logged_trajectories_states[agent_i][10][2]
                knn_z = z_values[0]
                values[:, 2] = z_values - knn_z + logz
                values_hf = np.zeros((91, 4))
                values_hf[:11, :] = np.array(logged_trajectories_states[agent_i])
                valid_hf = np.zeros((91)).astype(bool)
                valid_hf[:11] = np.array(logged_trajectories_valid[agent_i])
                is_interp_hf = np.zeros((91)).astype(bool)
                assert 10 % pred_freq == 0
                frame_expand_ratio = 10 // pred_freq
                for i in range(1, N_pred_frames):
                    n_hist = 11 + frame_expand_ratio * (i - 1)
                    hist_values = values_hf[:n_hist, :]
                    hist_valid = valid_hf[:n_hist]
                    pred_value = values[i]
                    pred_valid = valid[i]
                    new_values, new_interp_hf = SimAgentsMetric.linear_interp_and_expand_any_freq(
                        hist_values, hist_valid, pred_value, pred_valid, exp_ratio=frame_expand_ratio)
                    n_future = n_hist + frame_expand_ratio
                    values_hf[n_hist:n_future, :] = new_values
                    valid_hf[n_hist:n_future] = True
                    is_interp_hf[n_hist:n_future] = new_interp_hf
                preds[ri, agent_i] = values_hf[11:, :]
                is_interp[ri, agent_i] = is_interp_hf[11:]
            missing_agent.append(missing_agent_ri)
        missing_agent = np.array(missing_agent)
        if not query_agent < 0:
            missing_agent = np.ones_like(missing_agent)
            missing_agent[:, query_agent] = 0
        missing_agent_rate = missing_agent.sum() / len(missing_agent.reshape(-1))
        smoothing_factors = [9, 9, 1, 9, 9]
        smoothed_trajectory = np.zeros(list(preds.shape[:3]) + [5])
        smoothed_trajectory[..., :3] = preds[..., :3]
        smoothed_trajectory[..., 3] = np.sin(preds[..., 3])
        smoothed_trajectory[..., 4] = np.cos(preds[..., 3])
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                for dim in range(preds.shape[3]):
                    if smoothing_factors[dim] > 1:
                        smoothed_trajectory[i, j, :, dim] = smooth_window(smoothed_trajectory[i, j, :, dim], smoothing_factors[dim])
        smoothed_trajectory[..., 3] = np.arctan2(smoothed_trajectory[..., 3], smoothed_trajectory[..., 4])
        simulated_states = tf.convert_to_tensor(smoothed_trajectory[..., :4])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pkl_out_path = os.path.join(output_dir, pred_path.split('/')[-1])
        with open(pkl_out_path, 'wb') as f:
            pickle.dump(simulated_states, f)
        scenario_rollouts = SimAgentsMetric.scenario_rollouts_from_states(
            waymo_scenario, simulated_states, logged_trajectories.object_id)
        submission_specs.validate_scenario_rollouts(scenario_rollouts, waymo_scenario)
        if not return_metrics:
            return scenario_rollouts
        config = metrics.load_metrics_config()
        scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
            config, waymo_scenario, scenario_rollouts)
        scenario_result = [scenario_metrics.scenario_id, scenario_metrics.metametric,
                           scenario_metrics.average_displacement_error, scenario_metrics.linear_speed_likelihood,
                           scenario_metrics.linear_acceleration_likelihood, scenario_metrics.angular_speed_likelihood,
                           scenario_metrics.angular_acceleration_likelihood, scenario_metrics.distance_to_nearest_object_likelihood,
                           scenario_metrics.collision_indication_likelihood, scenario_metrics.time_to_collision_likelihood,
                           scenario_metrics.distance_to_road_edge_likelihood, scenario_metrics.offroad_indication_likelihood,
                           scenario_metrics.min_average_displacement_error,
                           missing_agent_rate]
        return scenario_result

    @staticmethod
    def pack_shared(shard, shard_suffix):
        shard_srs = []
        for p in shard:
            sr = SimAgentsMetric.compute_single_speed_interp_any_freq_smoothing(pred_path=p, pred_freq=2, return_metrics=False, query_agent=-1)
            shard_srs.append(sr)
    
    @staticmethod
    def pack_shared_wrapper(args):
            return SimAgentsMetric.pack_shared(*args)

    @staticmethod
    def process_with_multiprocessing(all_shards, shard_suffixes, num_processes=16):
        with multiprocessing.Pool(num_processes) as pool:
            results = list(tqdm(pool.imap(SimAgentsMetric.pack_shared_wrapper, zip(all_shards, shard_suffixes)), total=len(shard_suffixes)))
        return results

    def pack_submission(self, pkl_dir, output_dir, shared_size=16, num_processes=32):
        all_preds = glob.glob(os.path.join(pkl_dir, '*.pkl'))
        all_shards = []
        shard_suffixes = []
        N_shard = math.ceil(len(all_preds) / shared_size)
        assert N_shard < 99999, 'too many shards!'
        total_number_str = str(N_shard).zfill(5)
        for shard_i in range(N_shard):
            shard = all_preds[shard_i * shared_size : (shard_i+1) * shared_size]
            index_str = str(shard_i+1).zfill(5)
            shard_suffix = f"-{index_str}-of-{total_number_str}"
            all_shards.append(shard)
            shard_suffixes.append(shard_suffix)
        output_filenames = SimAgentsMetric.process_with_multiprocessing(all_shards, shard_suffixes, num_processes)
        return output_filenames

    def compute(self, pkl_dir, output_dir, num_processes):
        all_preds = sorted(glob.glob(os.path.join(pkl_dir, '*_rollouts.pkl')))
        paths = [(pred, output_dir) for pred in all_preds]
        with multiprocessing.Pool(num_processes) as pool:
            results = list(tqdm(pool.imap(SimAgentsMetric.compute_single_speed_interp_any_freq_smoothing, paths), total=len(paths)))
        numerical_results = [np.array(r[1:]) for r in results]
        numerical_results = np.vstack(numerical_results)
        low_score_mask = numerical_results[:, 0] < 0.25
        low_score_results = [res for res, is_low in zip(all_preds, low_score_mask) if is_low]
        # with open('low-score-results.pkl', 'wb') as f:
        #     pickle.dump(low_score_results, f)
        final_results = numerical_results.mean(axis=0)
        # with open('full-results.pkl', 'wb') as f:
        #     pickle.dump(final_results, f)
        entries = [
            'metametric',  
            'average_displacement_error',
            'linear_speed_likelihood',
            'linear_acceleration_likelihood',
            'angular_speed_likelihood',
            'angular_acceleration_likelihood',
            'distance_to_nearest_object_likelihood',
            'collision_indication_likelihood',
            'time_to_collision_likelihood',
            'distance_to_road_edge_likelihood',
            'offroad_indication_likelihood',
            'min_average_displacement_error',
            'my_missing_agent_ratio'
        ]
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        for entry, v in zip(entries, final_results):
            print('{}: \t{}'.format(entry, v))
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', required=True, help='Path to the directory containing pkl files')
    parser.add_argument('--nprocess', type=int, default=32, help='Number of processes to use')
    parser.add_argument('--output_dir', required=True, help='Directory to save the output files')
    args = parser.parse_args()

    metric = SimAgentsMetric()
    # metric.pack_submission(args.pkl_dir, args.output_dir, num_processes=args.nprocess)
    metric.compute(args.pkl_dir, args.output_dir, args.nprocess)

if __name__ == '__main__':
    main()
