from typing import List
import os

import torch
import torch.nn.functional as F
import numpy as np
import pickle

from torchmetrics import Metric
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

# import matplotlib
# matplotlib.use('Agg')  # Use 'Agg' for non-interactive plots
# import matplotlib.pyplot as plt


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
        
        self.all_scenario_id = []
        self.all_metametric = []

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
        sim_agents_rollouts = predictions['sim_agents_rollouts']  

        bs = len(sim_agents_rollouts)

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
            agent_ids = submission_specs.get_sim_agent_ids(waymo_scenario)
            ego_id = waymo_scenario.sdc_track_index
            ego_id =  waymo_scenario.tracks[ego_id].id
            # process pred part
            parallel_rollouts = sim_agents_rollouts[bi]  # 32 rollouts
            all_lf_values = []
            all_lf_valid = []

            # ##################################
            # for ri, rollout in enumerate(parallel_rollouts):
            #     preds_curr = [agent for agent in rollout if agent.frame_index == 2]
            #     for agent_i, idx in enumerate(agent_ids):
            #         agent_pred = [agent for agent in preds_curr if agent.raw_id == idx]  
            #         if not agent_pred:
            #             import pdb; pdb.set_trace()
            # ##################################

            for ri, rollout in enumerate(parallel_rollouts):
                for agent_i in range(len(rollout)):
                    if rollout[agent_i].is_ego:
                        # ego agent's raw_id is None
                        rollout[agent_i].raw_id = ego_id

                # create a raw_id -> agent token list map
                map_agent = {}
                for agent_tok in rollout:
                    agent_raw_id = agent_tok.raw_id
                    if agent_raw_id in map_agent:
                        map_agent[agent_raw_id].append(agent_tok)
                    else:
                        map_agent[agent_raw_id] = [agent_tok]
                agent_values = []
                agent_valid = []
                # import time
                for agent_i, idx in enumerate(agent_ids):
                    if not idx in map_agent:
                        print('Missing pred for agent: {} {}'.format(idx, waymo_scenario.scenario_id))
                        agent_pred = []
                    else:
                        agent_pred = sorted(map_agent[idx], key=lambda x: x.frame_index)  # sort by timestamp
                        
                    if not [agent.frame_index for agent in agent_pred] == list(range(2, 2 + len(agent_pred))):
                        # print([agent.frame_index for agent in agent_pred])
                        agent_pred_new = []
                        for kk in range(len(agent_pred)):
                            if agent_pred[kk].frame_index == 2 + kk:
                                agent_pred_new.append(agent_pred[kk])
                                
                        agent_pred = agent_pred_new
                    assert [agent.frame_index for agent in agent_pred] == list(range(2, 2 + len(agent_pred)))
                    n_valid_steps = len(agent_pred)

                    # nHz mask and values
                    freq = 2
                    n_frames = 8 * freq + 1
                    values = torch.zeros(n_frames, 6)
                    valid = torch.zeros(n_frames)
                    values[:, 4] = float(agent_pred[0].width) if len(agent_pred) > 0 else 1
                    values[:, 5] = float(agent_pred[0].length) if len(agent_pred) > 0 else 1

                    values[:n_valid_steps, 0] = torch.tensor([agent.x for agent in agent_pred])
                    values[:n_valid_steps, 1] = torch.tensor([agent.y for agent in agent_pred])
                    values[:n_valid_steps, 3] = torch.tensor([agent.heading for agent in agent_pred])
                    valid[:n_valid_steps] = 1
                    
                    if len(agent_pred) > 0:
                        start_index = agent_pred[0].frame_index
                    else:
                        start_index = 0

                    next_x, next_y, next_heading = None, None, None
                    for agent in agent_pred:
                        index = agent.frame_index - start_index
                        traj_valid =  agent.predicted_future_trajectory is not None and  agent.pred_future_prob is not None
                        if next_x is not None: 
                            dist = np.sqrt((next_x - agent.x)**2 + (next_y - agent.y)**2)
                            if dist > 5.0:
                                n_valid_steps = max(index+1, 1)
                                valid[:n_valid_steps] = 1
                                valid[n_valid_steps:] = 0
                                break
                        if traj_valid:
                            traj_pred = agent.predicted_future_trajectory 
                            best_traj_idx = agent.pred_future_prob.argmax() 
                            traj = traj_pred[best_traj_idx] 

                            next_x = agent.x + traj[0,0]
                            next_y = agent.y + traj[0,1]
                            next_heading = traj[0,2]
                        else:
                            next_x, next_y, next_heading = None, None, None


                    if n_valid_steps > 0 and n_valid_steps < n_frames:
                        # print(n_valid_steps, agent_pred[n_valid_steps-1].x, agent_pred[n_valid_steps-1].y)
                        # last_valid_agent_pred = agent_pred[-1]
                        last_valid_agent_pred = agent_pred[n_valid_steps-1]
                        traj_pred = last_valid_agent_pred.predicted_future_trajectory
                        best_traj_idx = last_valid_agent_pred.pred_future_prob.argmax()
                        traj = traj_pred[best_traj_idx]  # (n_frames - 1, 3)
                        traj[:, 0] += last_valid_agent_pred.x
                        traj[:, 1] += last_valid_agent_pred.y
                        # new_heading = (traj[:, 2] + last_valid_agent_pred.heading) % (2 * np.pi)
                        new_heading = traj[:, 2] % (2 * np.pi)

                        def estimate_heading(xx, yy, x0, y0):
                            """
                            xx: np (16,), pred x
                            yy: np (16,), pred y
                            x0: float, current x
                            y0: float, current y
                            """
                            dx = xx - x0
                            dy = yy - y0
                            headings = np.arctan2(dy, dx)
                            return headings
                        
                        # new_heading = estimate_heading(traj[:, 0], traj[:, 1], last_valid_agent_pred.x, last_valid_agent_pred.y)

                        traj[:, 2] = new_heading
                        keep_steps = n_frames - n_valid_steps
                        values[n_valid_steps:, 0] = torch.from_numpy(traj[:keep_steps, 0])
                        values[n_valid_steps:, 1] = torch.from_numpy(traj[:keep_steps, 1])
                        values[n_valid_steps:, 3] = torch.from_numpy(traj[:keep_steps, 2])
                        valid[n_valid_steps:] = 1

                    agent_values.append(values.cpu())
                    agent_valid.append(valid.cpu())
                # import pdb; pdb.set_trace()
                all_lf_values.append(agent_values)
                all_lf_valid.append(agent_valid)

            with open(scene_path, 'wb') as f:
                pickle.dump({'values': all_lf_values, 'valid': all_lf_valid, 
                             'waymo_scenario_paths': waymo_scenario_paths[bi], 'transform': local_to_global_transforms[bi].cpu()}, f)
        return

    
    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the metric.

        :return: metric scalar tensor
        """
        
        
        result = np.array(self.all_metametric).mean()
        
        return result        
    

    def log(self, logger, data):
        pass