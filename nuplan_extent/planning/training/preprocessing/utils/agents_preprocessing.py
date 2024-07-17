from typing import Dict, List

import torch
from nuplan.common.actor_state.tracked_objects import (TrackedObjects)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.features.agents import (
    AgentFeatureIndex)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex, _extract_agent_tensor, _validate_agent_internal_shape)


def sampled_tracked_objects_based_on_object_type(
        tracked_objects: List[TrackedObjects],
        object_types: List[TrackedObjectType]) -> List[torch.Tensor]:
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param tracked_objects: The tracked objects to tensorize.
    :param object_types: List of TrackedObjectType to filter agents by.
    :return: The tensorized objects.
    """
    output: List[torch.Tensor] = []
    track_token_ids: Dict[str, int] = {}
    for _, tracked_objects_per_frame in enumerate(tracked_objects):
        valid_tracked_objects = []
        for object_type in object_types:
            tensorized, track_token_ids = _extract_agent_tensor(
                tracked_objects_per_frame, track_token_ids, object_type
            )
            if torch.count_nonzero(tensorized) > 0:
                valid_tracked_objects.append(tensorized)
        if valid_tracked_objects:
            valid_tracked_objects = torch.cat(valid_tracked_objects, dim=0)
            output.append(valid_tracked_objects)
        else:
            output.append(tensorized)
    return output


def pad_agent_states(agent_trajectories: List[torch.Tensor],
                     reverse: bool,
                     add_mask: bool = False) -> List[torch.Tensor]:
    """
    Pads the agent states with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param agent_trajectories: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :param add_mask: if True, a mask will be added to the output tensor, indicating which agents are valid at each timestep
    :param track_id_idx: index of the track id in the tensor, defaults to 0.
    :param force_validate_internal_shape: whether to validate trajectory shape before padding. Set to False to apply this function on
        custom data.
    :return: A trajectory of extracted states
    """
    if force_validate_internal_shape:
        for traj in agent_trajectories:
            _validate_agent_internal_shape(traj)

    if reverse:
        agent_trajectories = agent_trajectories[::-1]
    if add_mask:
        agent_trajectories_mask = []
    key_frame = agent_trajectories[0]

    id_row_mapping: Dict[int, int] = {}
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val.item())] = idx

    current_state = torch.zeros((key_frame.shape[0], key_frame.shape[1]),
                                dtype=torch.float32)
    for idx, frame in enumerate(agent_trajectories):
        if add_mask:
            current_mask = torch.zeros((key_frame.shape[0], 1),
                                       dtype=torch.float32)
        # Update current frame
        for row_idx in range(frame.shape[0]):
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx].item())]
            current_state[mapped_row, :] = frame[row_idx, :]
            if add_mask:
                current_mask[mapped_row, 0] = 1

        # Save current state
        agent_trajectories[idx] = torch.clone(current_state)
        if add_mask:
            agent_trajectories_mask.append(current_mask)

    if reverse:
        agent_trajectories = agent_trajectories[::-1]
        if add_mask:
            agent_trajectories_mask = agent_trajectories_mask[::-1]
    if add_mask:
        return agent_trajectories, agent_trajectories_mask
    else:
        return agent_trajectories


def pack_agents_tensor_with_mask(
        padded_agents_tensors: List[torch.Tensor], yaw_rates: torch.Tensor,
        avail_mask: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines the local padded agents states and the computed yaw rates into the final output feature tensor.
    :param padded_agents_tensors: The padded agent states for each timestamp.
        Each tensor is of shape <num_agents, len(AgentInternalIndex)> and conforms to the AgentInternalIndex schema.
    :param yaw_rates: The computed yaw rates. The tensor is of shape <num_timestamps, agent>
    :return: The final feature, a tensor of shape [timestamp, num_agents, len(AgentsFeatureIndex)] conforming to the AgentFeatureIndex Schema
    """
    if yaw_rates.shape != (
        len(padded_agents_tensors),
        padded_agents_tensors[0].shape[0],
    ):
        raise ValueError(f"Unexpected yaw_rates tensor shape: {yaw_rates.shape}")

    agents_tensor = torch.zeros(
        (
            len(padded_agents_tensors),
            padded_agents_tensors[0].shape[0],
            AgentFeatureIndex.dim() + 1,
        )
    )

    for i, padded_tensor in enumerate(padded_agents_tensors):
        _validate_agent_internal_shape(padded_tensor)
        agents_tensor[i, :, AgentFeatureIndex.x()] = padded_tensor[
            :, AgentInternalIndex.x()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.y()] = padded_tensor[
            :, AgentInternalIndex.y()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.heading()] = padded_tensor[
            :, AgentInternalIndex.heading()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vx()] = padded_tensor[
            :, AgentInternalIndex.vx()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vy()] = padded_tensor[
            :, AgentInternalIndex.vy()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.yaw_rate()] = yaw_rates[i, :].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.width()] = padded_tensor[
            :, AgentInternalIndex.width()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.length()] = padded_tensor[
            :, AgentInternalIndex.length()
        ].squeeze()
        agents_tensor[i, :, -1] = avail_mask[i].squeeze()

    return agents_tensor


# fmt: off
def build_generic_ego_features_from_tensor(
    ego_feature: torch.Tensor,
    reverse: bool = False,
    force_validate_internal_shape: bool = True,
    include_velocity: bool = True,
    include_acceleration: bool = True,
    other_feature_range: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Build ego vector features. Converts global coordinates to local.

    NOTE: adapted from build_generic_ego_feature_from_tensor from nuplan.planning.training.preprocessing.utils.agents_preprocessing.py

    :param ego_trajectory: a tensor of ego features with shape [num_frames, num_features].
    :param reverse: if True, use the last frame as the key frame. defaults to False.
    :param force_validate_internal_shape: set to False to use custom data, otherwise will do a
        shape validation.
    :param include_velocity: whether to include velocity during this process, defaults to True
    :param include_acceleration: whether to include acceleration during this process, defaults to True
    :param other_feature_range: index range of other features that stays untouched, must be tuple of (start, int).
    :return: processed tensor.
    """
    if force_validate_internal_shape:
        _validate_ego_internal_shape(ego_feature, expected_first_dim=2)

    if reverse:
        anchor_ego_pose = ego_feature[-1, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].squeeze().double()
        anchor_ego_velocity = ego_feature[-1, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]].squeeze().double() if include_velocity else None  # vx, vy, heading
        anchor_ego_acceleration = ego_feature[-1, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]].squeeze().double() if include_acceleration else None  # ax, ay, heading
    else:
        anchor_ego_pose = ego_feature[0, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]].squeeze().double()
        anchor_ego_velocity = ego_feature[0, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]].squeeze().double() if include_velocity else None  # vx, vy, heading
        anchor_ego_acceleration = ego_feature[0, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]].squeeze().double() if include_acceleration else None  # ax, ay, heading

    global_ego_poses = ego_feature[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
    global_ego_velocities = ego_feature[:, [EgoInternalIndex.vx(), EgoInternalIndex.vy(), EgoInternalIndex.heading()]]
    global_ego_acceleration = ego_feature[:, [EgoInternalIndex.ax(), EgoInternalIndex.ay(), EgoInternalIndex.heading()]]

    local_ego_poses = global_state_se2_tensor_to_local(global_ego_poses, anchor_ego_pose, precision=torch.float64)
    if include_velocity:
        local_ego_velocities = global_state_se2_tensor_to_local(global_ego_velocities, anchor_ego_velocity, precision=torch.float64)
    if include_acceleration:
        local_ego_acceleration = global_state_se2_tensor_to_local(global_ego_acceleration, anchor_ego_acceleration, precisoin=torch.float64)

    local_ego_trajectory: torch.Tensor = torch.empty(ego_feature.size(), dtype=torch.float32, device=ego_feature.device)
    local_ego_trajectory[:, EgoInternalIndex.x()] = local_ego_poses[:, 0].float()
    local_ego_trajectory[:, EgoInternalIndex.y()] = local_ego_poses[:, 1].float()
    local_ego_trajectory[:, EgoInternalIndex.heading()] = local_ego_poses[:, 2].float()
    if include_velocity:
        local_ego_trajectory[:, EgoInternalIndex.vx()] = local_ego_velocities[:, 0].float()
        local_ego_trajectory[:, EgoInternalIndex.vy()] = local_ego_velocities[:, 1].float()
    if include_acceleration:
        local_ego_trajectory[:, EgoInternalIndex.ax()] = local_ego_acceleration[:, 0].float()
        local_ego_trajectory[:, EgoInternalIndex.ay()] = local_ego_acceleration[:, 1].float()

    if other_feature_range is not None:
        start, end = other_feature_range
        local_ego_trajectory[:, start:end] = ego_feature[:, start:end].float()
    return local_ego_trajectory
# fmt: on


def filter_agents_tensor(
    agents: List[torch.Tensor],
    reverse: bool = False,
    force_validate_internal_shape: bool = True,
    track_id_idx: int = AgentInternalIndex.track_token(),
) -> List[torch.Tensor]:
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)

    NOTE: adapted filter_agents_tensor from nuplan.planning.training.preprocessing.utils.agents_preprocessing.py

    :param agents: list of agent tensor. Each represents a timestep.
    :param reverse: if True, use the last tensor as the key frame. Defaults to False.
    :param force_validate_internal_shape: whether to validate internal shape before doing the filtering. Set to False to
        use this function on custom data
    :param track_id_idx: track id index, defaults to 0
    """
    target_tensor = agents[-1] if reverse else agents[0]
    for i in range(len(agents)):
        if force_validate_internal_shape:
            _validate_agent_internal_shape(agents[i])

        rows: List[torch.Tensor] = []
        for j in range(agents[i].shape[0]):
            if target_tensor.shape[0] > 0:
                agent_id: float = float(agents[i][j, track_id_idx].item())
                is_in_target_frame: bool = bool(
                    (agent_id == target_tensor[:, track_id_idx]).max().item()
                )
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())

        if len(rows) > 0:
            agents[i] = torch.stack(rows)
        else:
            agents[i] = torch.empty((0, agents[i].shape[1]), dtype=torch.float32)

    return agents


def convert_absolute_quantities_to_relative(
    agent_states: List[torch.Tensor],
    ego_state: torch.Tensor,
    position_only: bool = False,
    force_validate_internal_shape: bool = True,
    custom_agent_index: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """
    Convert global quantities to local coordinate frame.

    Adapted from convert_absolute_quantities_to_relative from nuplan.planning.training.preprocessing.utils.agents_preprocessing.py

    :param agent_states: list of tensors to be converted.
    :param ego_state: a length-3 tensor representing ego_state.
    :param position_only: if False, will also convert velocity vector to local frame, defaults to False.
    :param force_validate_internal_shape: whether to validate shape. Set to False to apply function to custom data
    :param custom_agent_index: a int list with length of 5, specifying the index of x, y, heading, vx and vy of agent data.
    """
    if force_validate_internal_shape:
        _validate_agent_internal_shape(ego_state, expected_first_dim=1)

    ego_pose = torch.tensor([float(ego_state[EgoInternalIndex.x()].item()), float(ego_state[EgoInternalIndex.y()].item()), float(ego_state[EgoInternalIndex.heading()].item())], dtype=torch.float64)  # fmt: skip
    ego_velocity = torch.tensor([float(ego_state[EgoInternalIndex.vx()].item()), float(ego_state[EgoInternalIndex.vy()].item()), float(ego_state[EgoInternalIndex.heading()].item())])  # fmt: skip

    if custom_agent_index is None:
        agent_x_idx = AgentInternalIndex.x()
        agent_y_idx = AgentInternalIndex.y()
        agent_heading_idx = AgentInternalIndex.heading()
        agent_vx_idx = AgentInternalIndex.vx()
        agent_vy_idx = AgentInternalIndex.vy()
    else:
        agent_x_idx = custom_agent_index[0]
        agent_y_idx = custom_agent_index[1]
        agent_heading_idx = custom_agent_index[2]
        agent_vx_idx = custom_agent_index[3]
        agent_vy_idx = custom_agent_index[4]

    for agent_state in agent_states:
        if force_validate_internal_shape:
            _validate_agent_internal_shape(agent_state)

        agent_global_poses = agent_state[:, [agent_x_idx, agent_y_idx, agent_heading_idx]].double()  # fmt: skip
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)  # fmt: skip
        agent_state[:, agent_x_idx] = transformed_poses[:, 0].float()
        agent_state[:, agent_y_idx] = transformed_poses[:, 1].float()
        agent_state[:, agent_heading_idx] = transformed_poses[:, 2].float()
        if not position_only:
            agent_global_velocities = agent_state[:, [agent_vx_idx, agent_vy_idx, agent_heading_idx, ]].double()  # fmt: skip
            transformed_velocities = global_state_se2_tensor_to_local(
                agent_global_velocities, ego_velocity, precision=torch.float64
            )
            agent_state[:, agent_vx_idx] = transformed_velocities[:, 0].float()
            agent_state[:, agent_vy_idx] = transformed_velocities[:, 1].float()

    return agent_states
