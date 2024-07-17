from numba import jit
import numpy as np

@jit(nopython=True)
def get_future_trajectory(agents_np, t, batch_index,
                          cls_idx, agent_trajectory_np):
    num_timesteps, num_agents, _ = agents_np.shape

    future_hs = agents_np[:, :, 3]
    future_xs = agents_np[:, :, 6]
    future_ys = agents_np[:, :, 7]

    for current_id in range(num_agents):
        current_agent = agents_np[t, current_id]

        if current_agent[0] < 0:
            continue

        ego_agent_x = current_agent[6]
        ego_agent_y = current_agent[7]
        ego_agent_h = current_agent[3]

        for future_t in range(num_timesteps):
            future_agents = agents_np[future_t]

            for future_id in range(num_agents):
                future_agent = future_agents[future_id]

                if future_agent[0] < 0:
                    continue

                if int(future_agent[0]) == int(current_agent[0]):
                    dx = future_xs[future_t, future_id] - ego_agent_x
                    dy = future_ys[future_t, future_id] - ego_agent_y
                    dh = future_hs[future_t, future_id]

                    agent_trajectory_np[batch_index, t,
                                        cls_idx + 1, current_id, future_t, 0] = dx
                    agent_trajectory_np[batch_index, t,
                                        cls_idx + 1, current_id, future_t, 1] = dy
                    agent_trajectory_np[batch_index, t,
                                        cls_idx + 1, current_id, future_t, 2] = dh
                    break
    return agent_trajectory_np


@jit(nopython=True)
def isin_numba_set(array1, array2):
    set2 = set(array2)
    result = np.zeros(array1.shape, dtype=np.bool_)
    for i in range(array1.shape[0]):
        result[i] = array1[i] in set2
    return result


@jit(nopython=True)
def find_disappeared_newborn_survived(
        prev_track_ids, cls_idx, valid_track_ids, state_tokens, track_to_token_indexing_table):
    # Find Disappeared IDs
    mask_disappeared = ~isin_numba_set(
        prev_track_ids[cls_idx], valid_track_ids)
    disappeared_ids = prev_track_ids[cls_idx][mask_disappeared]
    disappeared_ids = disappeared_ids[disappeared_ids >= 0]
    disappeared_tokens = track_to_token_indexing_table[disappeared_ids]

    # Find Newborn IDs
    mask_newborn = ~isin_numba_set(valid_track_ids, prev_track_ids[cls_idx])
    newborn_ids = valid_track_ids[mask_newborn]
    newborn_state_tokens = state_tokens[mask_newborn]

    # Find Survived IDs
    mask_survived = isin_numba_set(valid_track_ids, prev_track_ids[cls_idx])
    survived_ids = valid_track_ids[mask_survived]
    survived_id_tokens = track_to_token_indexing_table[survived_ids]
    sorted_index = np.argsort(survived_id_tokens)
    survived_id_tokens = survived_id_tokens[sorted_index]
    survived_state_tokens = state_tokens[mask_survived][sorted_index]
    return disappeared_tokens, newborn_ids, newborn_state_tokens, survived_id_tokens, survived_state_tokens