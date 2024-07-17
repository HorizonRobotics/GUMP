
import numpy as np
import torch
from typing import Tuple, Type, List
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization, PlannerInput)
from nuplan_extent.planning.training.preprocessing.features.dict_tensor_feature import DictTensorFeature
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder, AbstractModelFeature)
from third_party.mtr.datasets.waymo.waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type
from third_party.mtr.utils import common_utils


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir

class WaymoMapFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for constructing map features in a vector-representation.
    """

    def __init__(self, max_num_map_objects: int, radius: float, center_offset: List[float]) -> None:
        super().__init__()
        # a number of closest polylines for center object
        self._max_num_map_objects = max_num_map_objects
        self._radius = radius
        self._select_map_layers = [
            SemanticMapLayer.BASELINE_PATHS,
            SemanticMapLayer.LANE,
            SemanticMapLayer.EXTENDED_PUDO,
            SemanticMapLayer.STOP_SIGN,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.SPEED_BUMP
        ]
        self._point_sampled_interval = 1
        self._vector_break_dist_thresh = 1.0
        self._num_points_each_polyline = 20
        self._center_offset = center_offset


    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "map_polylines"


    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return DictTensorFeature  # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(
            self, 
            scenario: AbstractScenario,
            center_offset: Tuple[float] = (0,0)) -> DictTensorFeature:
        """Inherited, see superclass."""
        ego_state = scenario.initial_ego_state
        map_api = scenario.map_api
        near_map_objects = map_api.get_proximal_map_objects(
            layers=self._select_map_layers,
            point=ego_state.rear_axle.point,
            radius=self._radius)
        
        polylines = []
        for map_layer, map_objects in near_map_objects.items():
            if map_layer == SemanticMapLayer.LANE:
                for map_object in map_objects:
                    global_type = polyline_type[lane_type[map_object.type]]
                    cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in map_object.map_feature.lane.polyline], axis=0)
                    cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                    cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                    polylines.append(cur_polyline)
            elif map_layer == SemanticMapLayer.BASELINE_PATHS:
                for map_object in map_objects:
                    global_type = polyline_type[road_line_type[map_object.type]]
                    cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in map_object.map_feature.road_line.polyline], axis=0)
                    cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                    cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                    polylines.append(cur_polyline)

            elif map_layer == SemanticMapLayer.EXTENDED_PUDO:
                for map_object in map_objects:
                    global_type = polyline_type[road_edge_type[map_object.type]]
                    cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in map_object.map_feature.road_edge.polyline], axis=0)
                    cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                    cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                    polylines.append(cur_polyline)

            elif map_layer == SemanticMapLayer.STOP_SIGN:
                for map_object in map_objects:
                    point = map_object.map_feature.stop_sign.position
                    global_type = polyline_type['TYPE_STOP_SIGN']
                    cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)
                    polylines.append(cur_polyline)
            
            elif map_layer == SemanticMapLayer.CROSSWALK:
                for map_object in map_objects:
                    global_type = polyline_type['TYPE_CROSSWALK']
                    cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in map_object.map_feature.crosswalk.polygon], axis=0)
                    cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                    cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                    polylines.append(cur_polyline)
            
            elif map_layer == SemanticMapLayer.SPEED_BUMP:
                for map_object in map_objects:
                    global_type = polyline_type['TYPE_SPEED_BUMP']
                    cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in map_object.map_feature.speed_bump.polygon], axis=0)
                    cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
                    cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                    polylines.append(cur_polyline)
            else:
                raise ValueError                    
        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
            print('Empty polylines: ')            


        batch_polylines, batch_polylines_mask = self.generate_batch_polylines_from_map(
            polylines=polylines, point_sampled_interval=self._point_sampled_interval,
            vector_break_dist_thresh=self._vector_break_dist_thresh,
            num_points_each_polyline=self._num_points_each_polyline,
        ) 
        if len(batch_polylines) > self._max_num_map_objects:
            polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
            center_offset_rot = torch.from_numpy(np.array(self._center_offset, dtype=np.float32))[None, :].repeat(1, 1)
            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot.view(1, 1, 2),
                angle=torch.tensor([ego_state.rear_axle.heading]).view(1, 1)
            ).view(1, 2)

            pos_of_map_centers = torch.tensor(ego_state.rear_axle.point.array).view(1,2) + center_offset_rot

            dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (1, num_polylines)
            topk_dist, topk_idxs = dist.topk(k=self._max_num_map_objects, dim=-1, largest=False)
            map_polylines = batch_polylines[topk_idxs]  # (1, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[topk_idxs]  # (1, num_topk_polylines, num_points_each_polyline)
        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(1, 1, 1, 1)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(1, 1, 1)

        center_objects = torch.tensor([ego_state.rear_axle.point.x, ego_state.rear_axle.point.y, 0]).view(1, 3)
        
        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].view(1, -1, 2),
                angle=-torch.tensor([ego_state.rear_axle.heading]).view(1, 1)
            ).view(1, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].view(1, -1, 2),
                angle=-torch.tensor([ego_state.rear_axle.heading]).view(1, 1)
            ).view(1, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (1, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask        
        
        
        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (1, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (1, num_polylines, 3)

        map_polylines = map_polylines.squeeze(0)
        map_polylines_mask = map_polylines_mask.squeeze(0)
        map_polylines_center = map_polylines_center.squeeze(0)

        ret_dict = {}
        ret_dict['map_polylines'] = map_polylines
        ret_dict['map_polylines_mask'] = (map_polylines_mask > 0)
        ret_dict['map_polylines_center'] = map_polylines_center

        return DictTensorFeature(ret_dict)        



    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask        

    @torch.jit.unused
    def get_features_from_simulation(
            self, current_input: PlannerInput,
            initialization: PlannerInitialization) -> DictTensorFeature:
        """Inherited, see superclass."""
        pass        