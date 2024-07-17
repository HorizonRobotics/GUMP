from typing import Callable, List, Tuple, Optional, Dict, Generator, Union

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    get_traffic_light_dict_from_generator,
    _get_layer_coords,
)

def get_traffict_light_vector_list(
    lane_ids,
    baseline_paths_coords,
    traffic_light_connectors
) -> List[Dict[str, Union[str, List[float]]]]:
    """
    Get lane color from lane_connector color dict.
    :param lane_ids: the list of lane_ids which we want the color for
    :param traffic_light_connectors: our dict for traffic light information
    :return: lane_colors array for input lanes.
    """
    # Get a list indicating the color of each lane
    tl_index = 1
    lane_ids = [int(x) for x in lane_ids]
    traffict_light_vector_list = []
    for traffic_light_status, traffic_lane_ids in traffic_light_connectors.items():
        # if TrafficLightStatusType.UNKNOWN != traffic_light_status:
        for traffic_lane_id in traffic_lane_ids:
            if traffic_lane_id in lane_ids:
                lane_index = lane_ids.index(traffic_lane_id)
                lane_coords = baseline_paths_coords[lane_index]
                traffict_light_vector_list.append(
                    {'lane_id': traffic_lane_id,
                     'lane_coords': lane_coords,
                     'traffic_light_status': traffic_light_status,
                     'tl_index': tl_index,}
                )
                tl_index += 1
    return traffict_light_vector_list

def get_vectorized_traffic_light_data(
    ego_state: EgoState,
    map_api: AbstractMap,
    traffic_light_connectors: Dict[TrafficLightStatusType, List[str]],
):
    radius = 50
    baseline_paths_coords = []
    lane_ids = []
    for map_features in ['LANE', 'LANE_CONNECTOR']:
        baseline_paths_coord, lane_id = _get_layer_coords(
            ego_state=ego_state,
            map_api=map_api,
            map_layer_name=SemanticMapLayer[map_features],
            map_layer_geometry='linestring',
            radius=radius,
        )
        baseline_paths_coords += baseline_paths_coord
        lane_ids += lane_id 
        
    # get lane_colors for the two type.
    traffict_light_vector_list = get_traffict_light_vector_list(
        lane_ids,
        baseline_paths_coords,
        traffic_light_connectors)
    return traffict_light_vector_list