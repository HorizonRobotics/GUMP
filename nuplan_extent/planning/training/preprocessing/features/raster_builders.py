from abc import ABC, abstractmethod
import cv2
import math
import numpy.typing as npt
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.coords import CoordinateSequence
from typing import Dict, List, NamedTuple, Optional, Tuple
import alf
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, PolygonMapObject, PolylineMapObject
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import get_route_polygon_from_roadblock_ids
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan_extent.planning.scenario_builder.prepared_scenario import (
    NpEgoState, PreparedScenario, PreparedScenarioFeatureBuilder)
from nuplan_extent.planning.training.preprocessing.features.raster_utils import (
    generate_virtual_center,
    _get_proximal_map_objects,
)
from nuplan_extent.planning.training.preprocessing.utils.route_utils import (
    route_roadblock_correction, )
from nuplan_extent.common.geometry.oriented_box import box_to_corners

from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from typing import Set
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import prune_route_by_connectivity


def get_roadblock_from_roadblock_ids(
        map_api: AbstractMap, point: Point2D, radius: float,
        route_roadblock_ids: List[str]) -> List[RoadBlockGraphEdgeMapObject]:
    """
    Extract route polygon from map for route specified by list of roadblock ids. Polygon is represented as collection of
        polygons of roadblocks/roadblock connectors encompassing route.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param route_roadblock_ids: ids of roadblocks/roadblock connectors specifying route.
    :return: A route as sequence of roadblock/roadblock connector polygons.
    """
    roadblocks: List[RoadBlockGraphEdgeMapObject] = []

    # extract roadblocks/connectors within query radius to limit route consideration
    layer_names = [
        SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR
    ]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    roadblock_ids: Set[str] = set()

    for layer_name in layer_names:
        roadblock_ids = roadblock_ids.union(
            {map_object.id
             for map_object in layers[layer_name]})
    # prune route by connected roadblocks within query radius
    route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids,
                                                      roadblock_ids)

    for route_roadblock_id in route_roadblock_ids:
        # roadblock
        roadblock_obj = map_api.get_map_object(route_roadblock_id,
                                               SemanticMapLayer.ROADBLOCK)

        # roadblock connector
        if not roadblock_obj:
            roadblock_obj = map_api.get_map_object(
                route_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)

        if roadblock_obj:
            roadblocks.append(roadblock_obj)

    return roadblocks


def _get_coords_xy(coords: CoordinateSequence) -> npt.NDArray[np.float64]:
    """X and Y arrays from coords.

    :param coords: the coordinate sequence
    :return: 2d coordinates of the vertices of the polygon.
    """
    return np.array(coords)


def _linestring_to_coords(geometry: LaneGraphEdgeMapObject):
    """Get 2d coordinates of the endpoints of line segment string.

    The line segment string is a shapely.geometry.linestring.
    :param geometry: the line segment string.
    :return: 2d coordinates of the endpoints of line segment string.
    """
    return _get_coords_xy(geometry.baseline_path.linestring.coords)


def _polygon_to_coords(geometry: PolygonMapObject):
    """Get 2d coordinates of the vertices of a polygon.

    The polygon is a shapely.geometry.polygon.
    :param geometry: the polygon.
    :return: 2d coordinates of the vertices of the polygon.
    """
    return _get_coords_xy(geometry.polygon.exterior.coords)


def _polyline_to_coords(geometry: PolylineMapObject):
    """Get 2d coordinates of the vertices of a polyline.

    :param geometry: the polyline.
    :return: 2d coordinates of the vertices of the polygon.
    """
    return _get_coords_xy(geometry.linestring.coords)


def get_deduplicated_route_roadblock_ids(
        scenario: AbstractScenario,
        with_route_roadblock_correction: bool = True) -> List[str]:
    """Get the roadblock ids of the route without duplication.

    Sometimes there are artifacts in the route that the adjacent roadblocks are
    repeated. This is a hack to remove the repeated roadblocks.

    :param scenario: scenario
    :param with_route_roadblock_correction: whether to correct the route roadblocks.
        In some scenario, the original route roadblocks have some artifacts. For
        example, the ego car may not be on the route. This option will correct
        the artifacts.
    :return: roadblock ids of the route without duplication
    """
    ids = scenario.get_route_roadblock_ids()
    roadblock_ids = [ids[0]]
    for i in ids:
        if i != roadblock_ids[-1]:
            roadblock_ids.append(i)
    if with_route_roadblock_correction:
        try:
            roadblock_ids = route_roadblock_correction(
                scenario.get_ego_state_at_iteration(0), scenario.map_api,
                roadblock_ids)
        except:
            print('Route roadblock correction failed')
    return roadblock_ids


class ShapeList(NamedTuple):
    """Represents a list of shapes.

    All the coordinates are put into one single array for efficent storage
    and computation. The sizes array is used to index into the coordinates
    """

    # concatenated  coordinates of the points of all the shapes, shape [N, 2]
    coords: npt.NDArray[np.float32]

    # number of points for each shape, shape [M], sizes.sum() == coords.shape[0]
    sizes: npt.NDArray[np.int32]

    @property
    def num(self):
        """The number of shapes in this list."""
        return len(self.sizes)


class PreparedMapObject(NamedTuple):
    coords: npt.NDArray[np.float32]
    speed_limit: Optional[float]


def _get_extended_layer_untransformed_coords(
        center: Point2D, radius: float, offset: float, map_api: AbstractMap,
        map_layer_name: SemanticMapLayer, map_layer_geometry: str,
        prepared_map_objects: PreparedMapObject):
    """Get the untransformed coordinates of of the objects for the given layers.

    :param center: center of the raster in the coordinates of the original map
    :param radius: radius of the raster in meters
    :param offset: offset to be subtracted from the coordinates
    :param map_api: map api
    :param map_layer_name: name of the map layer
    :param map_layer_geometry: geometry type of the map layer, one of ('polygon',
        'linestring'). If it is 'polygon', the coordinates of the vertices of
        the polygon will be returned. If it is 'linestring', the coordinates of
        the baseline paths will be returned.
    :param prepared_map_objects: dict from map object is to the the prepared map
        objects. New map objects will be added to this dict.
    :return: list of ids of the objects for the given layers.
    """
    nearest_vector_map = _get_proximal_map_objects(
        map_api=map_api,
        layers=[map_layer_name],
        point=center,
        radius=radius,
    )

    geometry = nearest_vector_map[map_layer_name]

    if map_layer_name in [SemanticMapLayer['BOUNDARIES']]:
        obj_to_coords_func = _polyline_to_coords
    elif map_layer_geometry == 'polygon':
        obj_to_coords_func = _polygon_to_coords
    elif map_layer_geometry == 'linestring':
        obj_to_coords_func = _linestring_to_coords
    else:
        raise RuntimeError(
            f'Layer geometry {map_layer_geometry} type not supported')
    for geom in geometry:
        if geom.id in prepared_map_objects:
            continue
        coords = obj_to_coords_func(geom)
        coords = (coords - offset).astype(np.float32)
        speed_limit = getattr(geom, 'speed_limit_mps', None)
        prepared_map_objects[geom.id] = PreparedMapObject(coords, speed_limit)

    geom_ids = [geom.id for geom in geometry]

    return geom_ids


# This matrix transforms the ego coordinates to raster coordinates.
# The x-axis of the raster coordinate system corresponds to the y-axis of
# the ego coordinate (i.e. the left direction of the ego vehicle). The y-axis of
# the raster coordinate corresponds to the negative x-axis of the ego coordinate
# (i.e the backward direction of the ego vehicle).
_map_align_transform = np.diag([1., -1., 1.]) @ R.from_euler(
    'z', 90, degrees=True).as_matrix()


def get_global_to_local_transform(pose: StateSE2) -> npt.NDArray[np.float32]:
    """The the transformation matrix from global to local coordinates.

    :param pose: the pose of the local coordinate system in the global coordinate system.
    :return: inverse of the 3x3 2D transformation matrix
    """
    c = math.cos(pose.heading)
    s = math.sin(pose.heading)
    x = pose.x
    y = pose.y
    return np.array([
        [c, s, -c * x - s * y],
        [-s, c, s * x - c * y],
        [0, 0, 1],
    ])


def get_local_to_global_transform(pose: StateSE2) -> npt.NDArray[np.float32]:
    """The the transformation matrix from global to local coordinates.

    :param pose: the pose of the local coordinate system in the global coordinate system.
    :return: inverse of the 3x3 2D transformation matrix
    """
    c = math.cos(pose.heading)
    s = math.sin(pose.heading)
    x = pose.x
    y = pose.y
    return np.array([
        [c, -s, x],
        [s, c, y],
        [0, 0, 1],
    ])


def transform_to_pixel_coords(untransformed_coords: npt.NDArray[np.float32],
                              center: StateSE2, radius: float, image_size: int,
                              bit_shift: int):
    """Transform global coordinates to pixel coordinates.

    Note that the resulted coordinates are the actual pixel locations multiplied
    by 2**bit_shift. bit_shift is for cv2.fillPoly/cv2.polylines to work with
    float coordinates.

    :param untransformed_coords: global coordinates
    :param center: the pose of the local coordinate system in the global
        coordinate system.
    :param radius: radius of the raster in meters
    :param image_size: size of the raster image in pixels
    :param bit_shift: bit shift for converting float to int
    """

    if len(untransformed_coords) == 0:
        return []
    resolution = (2 * radius) / image_size
    scale = 2**bit_shift / resolution
    global_transform = get_global_to_local_transform(center)
    # By default the map is right-oriented, this makes it top-oriented.
    transform = _map_align_transform @ global_transform
    mat = (transform[:2, :2].T * scale).astype(np.float32)
    vec = transform[:2, 2] + radius
    # Previously, the raster is generated with a different _map_align_transform
    # without multiplying diag(1,-1, 1). And the result is flipped vertically.
    # (See raster_utils.get_roadmap_raster for example).
    # The current way achieves the flip directly by changing _map_align_transform.
    # However, this results in a one pixel vertical shift. The following line
    # is to compensate for this shift so that the generated raster is compatible
    # with the previously trainied model.
    vec[1] -= resolution
    vec = (vec * scale).astype(np.float32)
    object_coords = (untransformed_coords @ mat + vec).astype(np.int64)
    return object_coords


def draw_polygon_image(polygons: ShapeList, colors: List[float],
                       image_size: int, center: StateSE2,
                       radius: float) -> npt.NDArray[np.float32]:
    """Draw polygons on the raster.

    :param polygons: polygons to be drawn. The coordinates are in the global
        frame.
    :param colors: colors of the polygons
    :param image_size: size of the raster image in pixels
    :param center: the pose of the local coordinate system in the global
        coordinate system.
    :param radius: radius of the raster in meters
    """
    raster: npt.NDArray[np.float32] = np.zeros((image_size, image_size),
                                               dtype=np.float32)
    if polygons.coords.size == 0:
        return raster

    bit_shift = 10
    coords = transform_to_pixel_coords(polygons.coords, center, radius,
                                       image_size, bit_shift)
    start = 0
    for color, size in zip(colors, polygons.sizes):
        cv2.fillPoly(
            raster,
            coords[start:start + size][None],
            color=color,
            shift=bit_shift,
            lineType=cv2.LINE_AA)
        start += size
    return raster


def draw_polyline_image(polylines: ShapeList, colors: List[float],
                        thickness: int, image_size: int, center: StateSE2,
                        radius: float) -> npt.NDArray[np.float32]:
    """Draw polylines on the raster.

    :param polylines: polygons to be drawn. The coordinates are in the global
        frame.
    :param colors: colors of the polygons
    :param image_size: size of the raster image in pixels
    :param center: the pose of the local coordinate system in the global
        coordinate system.
    :param radius: radius of the raster in meters
    """
    raster: npt.NDArray[np.float32] = np.zeros((image_size, image_size),
                                               dtype=np.float32)
    if polylines.coords.size == 0:
        return raster
    bit_shift = 10
    coords = transform_to_pixel_coords(polylines.coords, center, radius,
                                       image_size, bit_shift)
    start = 0
    for color, size in zip(colors, polylines.sizes):
        cv2.polylines(
            raster, [coords[start:start + size]],
            isClosed=False,
            color=color,
            shift=bit_shift,
            thickness=thickness,
            lineType=cv2.LINE_AA)
        start += size
    return raster


def _to_shape_list(coords: List[npt.NDArray], offset: npt.NDArray
                   ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Convert a list of coordinates to a ShapeList.

    :param coords: a list of coordinates, each element is a numpy array of shape [N, 2]
    :param offset: offset to be subtracted from the coordinates
    """
    if len(coords) == 0:
        return ShapeList(
            np.empty((0, 2), dtype=np.float32), np.empty((0, ),
                                                         dtype=np.int32))
    sizes = np.array([len(c) for c in coords], dtype=np.int32)
    coords = np.concatenate(coords, axis=0)
    coords = (coords - offset).astype(np.float32)
    return ShapeList(coords, sizes)


def _concat_shape_list(shape_list: List[ShapeList]) -> ShapeList:
    """Concatenate a list of ShapeList into one ShapeList."""
    coords = np.concatenate([s.coords for s in shape_list], axis=0)
    sizes = np.concatenate([s.sizes for s in shape_list], axis=0)
    return ShapeList(coords, sizes)


class RasterBuilderBase(PreparedScenarioFeatureBuilder):
    """The base class for raster builders.

    :param image_size: size of the raster image in pixels
    :param radius: radius of the raster in meters
    :param ego_longitudinal_offset: the center of the raster is `longitudinal_offset * radius`
        meters in front of the rear axle of ego vehicle. 0 means the center is
        at the rear axle.
    """

    def __init__(self, image_size: int, radius: float,
                 longitudinal_offset: float):
        super().__init__()
        self._image_size = image_size
        self._radius = radius
        self._longitudinal_offset = longitudinal_offset
        self._cache_enabled = True
        # see set_cache_parameter for explanations about the following two parameters
        self._cache_grid_step = self._radius / 2
        self._cache_radius = self._radius * 2.0

    def set_cache_parameter(self, cache_grid_step: float,
                            cache_radius: float) -> None:
        """Setup cache parameters.

        The map features are cached based on the discretized center of the raster.
        If two raster has same discretized center, they will share the same cache.
        The discretized center is calculated as `(round(x / cache_grid_step), round(y / cache_grid_step))`.
        In order for the same cached feature to cover different raster at different
        centers, the cache radius should be larger than the radius of the raster.
        The default cache_grid_step is radius / 2, and the default cache_radius is
        radius * 1.5.

        :param cache_grid_step: grid step for the cache
        :param cache_radius: radius for the cache
        """
        self._cache_grid_step = cache_grid_step
        self._cache_radius = cache_radius

    def calc_raster_center(self, ego_state: NpEgoState) -> StateSE2:
        """Calculate the center of the raster.

        The center of the raster is `longitudinal_offset * radius` meters in front
        of the rear axle of ego_state.

        :param ego_state: ego state
        """
        return generate_virtual_center(
            StateSE2(*ego_state.rear_axle_pose.tolist()), self._radius,
            self._longitudinal_offset)

    def calc_cache_key(self, center: Point2D) -> Tuple[int, int]:
        """Calculate cache key.

        The cache key is the discretized center of the raster and it is used
        for retrieving the map objects from the cache.

        :param center: center of the raster in the coordinates of prepared scenario
        :return: cache key for retrieving the map objects in the cache.
        """
        step = self._cache_grid_step
        grid_x = int(math.floor(center.x / step + 0.5))
        grid_y = int(math.floor(center.y / step + 0.5))
        return grid_x, grid_y

    def calc_cache_key_and_center(self, center: Point2D, offset: npt.NDArray[2]
                                  ) -> Tuple[Tuple[int, int], Point2D]:
        """Calculate cache key and map center for getting map objects from the map.

        :param center: center of the raster in the coordinates of the prepared scenario
        :param offset: offset for shifting the coordinates for the prepared scenario
        :return:
            - cache key for storing/retrieving the map objects in the cache.
            - the center for retrieving map objects (in the original map coordinate)
                The caller should use the returned center and self._cache_radius to
                get map objects from the map.
        """
        step = self._cache_grid_step
        grid_x = int(math.floor(center.x / step + 0.5))
        grid_y = int(math.floor(center.y / step + 0.5))
        # Avoid numpy types since Point2D members are float
        x0 = grid_x * step + float(offset[0])
        y0 = grid_y * step + float(offset[1])
        return (grid_x, grid_y), Point2D(x0, y0)

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        pass


class MapObjectRasterBuilder(RasterBuilderBase):
    """The base class for raster builders that use map objects.

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    :param map_features: list of map features to be used
    """

    def __init__(self, image_size: int, radius: float,
                 longitudinal_offset: float, map_features: List[str]):
        super().__init__(image_size, radius, longitudinal_offset)
        self._map_features = map_features

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        map_api = scenario.map_api
        for iteration in iterations:
            ego_state = prepared_scenario.get_ego_state_at_iteration(iteration)
            center = self.calc_raster_center(ego_state)
            self.prepare_scenario_at_center(map_api, prepared_scenario, center)

    def prepare_scenario_at_center(self, map_api: AbstractMap,
                                   prepared_scenario: PreparedScenario,
                                   center: Point2D) -> None:
        """Prepare map objects at the given center.

        :param map_api: map api
        :param prepared_scenario: prepared scenario
        :param center: center of the map object in the coordinates of prepared scenario
        """
        for feature_name in self._map_features:
            self.prepare_map_objects(map_api, prepared_scenario, center,
                                     feature_name, 'polygon')

    def prepare_map_objects(self, map_api: AbstractMap,
                            scenario: PreparedScenario, center: StateSE2,
                            feature_name: str, geometry: str) -> None:
        """Prepare map object polygons/polylines at the given center.

        Also prepare features for  speed limit.

        Two features are prepared:
        - map_object_{geometry}_{feature_name} (Dict[str, PreparedMapObject]):
            polygons/linestring of the map objects. Each polygon/linestring is a list
            of 2d coordinates, with origin at `scenario.get_offset()`.
        - map_object_{geometry}_{feature_name}_ids (List[str]):
            the ids of the object around `center`.

        :param map_api: map api
        :param scenario: prepared scenario
        :param center: center of the map object
        :param feature_name: feature name
        :param geometry: geometry type of the map object, either "polygon" or "linestring"
        """
        geometry_feature_name = f'map_object_{geometry}_{feature_name}'
        geometry_feature_ids_name = f'map_object_{geometry}_{feature_name}_ids'
        offset = scenario.get_offset()
        cache_key, center = self.calc_cache_key_and_center(center, offset)
        if scenario.feature_exists_at_center(geometry_feature_ids_name,
                                             cache_key):
            return
        if scenario.feature_exists(geometry_feature_name):
            prepared_map_objects = scenario.get_feature(geometry_feature_name)
        else:
            prepared_map_objects = {}
            scenario.add_feature(geometry_feature_name, prepared_map_objects)
        obj_ids = _get_extended_layer_untransformed_coords(
            center, self._cache_radius, offset, map_api,
            SemanticMapLayer[feature_name], geometry, prepared_map_objects)
        scenario.add_feature_at_center(geometry_feature_ids_name, obj_ids,
                                       cache_key)


class EgoRasterBuilder(RasterBuilderBase):
    """The ego layer of the raster

    It draws the extent of the ego car at the origin.

    Cached Features: None

    Note that this ego layer is slightly different from raster_utils.get_ego_raster()
    because of using cv2.fillPoly instead of cv2.rectangle.

    TODO: get ego_width/ego_front_length/ego_rear_length from EgoState.car_footprint.vehicle_parameters

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    :param ego_width: width of the ego car
    :param ego_front_length: front length (rear-axle to front) of the ego car
    :param ego_rear_length: rear length (rear-axle to back) of the ego car
    """

    def __init__(self, image_size: int, radius: float,
                 longitudinal_offset: float, ego_width: float,
                 ego_front_length: float, ego_rear_length: float):
        super().__init__(image_size, radius, longitudinal_offset)
        self._ego_width = ego_width
        self._ego_front_length = ego_front_length
        self._ego_rear_length = ego_rear_length

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        x = self._radius * (1 - self._longitudinal_offset * 2)
        x0 = x - self._ego_rear_length
        x1 = x + self._ego_front_length
        y0 = -self._ego_width / 2
        y1 = self._ego_width / 2
        center = StateSE2(self._radius, 0, 0)
        polygons = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]],
                            dtype=np.float32)
        polygons = ShapeList(polygons, np.array([4], dtype=np.int32))
        return draw_polygon_image(
            polygons=polygons,
            colors=[1.0],
            image_size=self._image_size,
            center=center,
            radius=self._radius)


def _get_prepared_coords(scenario: PreparedScenario, feature_name: str,
                         cache_key: Tuple[int, int]):
    """Get the prepared coordinates of the map objects from PreparedScenario.

    :param scenario: prepared scenario
    :param feature_name: feature name
    :param cache_key: cache key
    :return:
        - coordinates of the map objects
        - ids of the map objects
        - dict from map object id to the prepared map objects
    """
    prepared_objects = scenario.get_feature(feature_name)
    obj_ids = scenario.get_feature_at_center(feature_name + '_ids', cache_key)
    return (prepared_objects[id].coords
            for id in obj_ids), obj_ids, prepared_objects


class RoadmapRasterBuilder(MapObjectRasterBuilder):
    """Map layer of the raster by converting vector map to raster map.

    Cached Features:
        map_object_polygons_{feature_name} (ShapeList): polygons of the map objects.

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    :param map_features: list of map features to be used
    """

    def __init__(self, image_size: int, radius: float,
                 longitudinal_offset: float,
                 map_feature_to_color_dict: Dict[str, float]):
        super().__init__(image_size, radius, longitudinal_offset,
                         list(map_feature_to_color_dict.keys()))
        self._map_feature_to_color_dict = map_feature_to_color_dict

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        all_polygons = []
        colors = []
        center = self.calc_raster_center(ego_state)
        cache_key = self.calc_cache_key(center)
        all_coords = []
        for feature_name, color in self._map_feature_to_color_dict.items():
            coords, obj_ids, _ = _get_prepared_coords(
                scenario, 'map_object_polygon_' + feature_name, cache_key)
            all_coords.extend(coords)
            colors.extend([color] * len(obj_ids))
        all_polygons = ShapeList(
            coords=np.concatenate(all_coords, axis=0),
            sizes=np.array([len(c) for c in all_coords]))
        return draw_polygon_image(all_polygons, colors, self._image_size,
                                  center, self._radius)


class SpeedLimitRasterBuilder(MapObjectRasterBuilder):
    """Raster representing the speed limit information of the map

    The speed is encoded as different colors for different map objects.

    Cached Features:
        map_object_polygons_{feature_name} (ShapeList): polygons of the map objects.
        map_object_speed_limit_{feature_name} (List[None|float]): speed limit of the map objects.

    feature_name is either "LANE" or "LANE_CONNECTOR".

    Note that speed information is not always available for all map objects. Those
    speed limits will be None.

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    :param none_speed_limit: the speed limit for None speed limit.
    :param max_speed_normalizer: the speed used to normalize the speed limit.
        The normalized speed limit is `speed_limit / max_speed_normalizer`.
    """

    def __init__(self,
                 image_size: int,
                 radius: float,
                 longitudinal_offset: float,
                 none_speed_limit: float = 0,
                 max_speed_normalizer: float = 16.0):
        super().__init__(image_size, radius, longitudinal_offset,
                         ["LANE", "LANE_CONNECTOR"])
        self._max_speed_normalizer = max_speed_normalizer
        self._none_speed_limit = none_speed_limit

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        all_coords = []
        colors = []
        center = self.calc_raster_center(ego_state)
        cache_key = self.calc_cache_key(center)
        for feature_name in self._map_features:
            coords, obj_ids, prepared_objects = _get_prepared_coords(
                scenario, 'map_object_polygon_' + feature_name, cache_key)
            all_coords.extend(coords)
            speed_limits = [(prepared_objects[id].speed_limit
                             or self._none_speed_limit) for id in obj_ids]
            colors.extend([
                speed_limit / self._max_speed_normalizer
                for speed_limit in speed_limits
            ])
        all_polygons = ShapeList(
            coords=np.concatenate(all_coords, axis=0),
            sizes=np.array([len(c) for c in all_coords]))
        return draw_polygon_image(all_polygons, colors, self._image_size,
                                  center, self._radius)


class RouteRasterBuilder(RasterBuilderBase):
    """Raster representing the navigation route.

    Cached Features:
        route_roadblock_polygons (ShapeList): polygons of the roadblocks on the route.

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    :param with_route_roadblock_correction: whether to correct the route roadblocks.
        In some scenario, the original route roadblocks have some artifacts. For
        example, the ego car may not be on the route. This option will correct
        the artifacts.
    """

    def __init__(self,
                 image_size: int,
                 radius: float,
                 longitudinal_offset: float,
                 with_route_roadblock_correction: float = True):
        super().__init__(image_size, radius, longitudinal_offset)
        self._with_route_roadblock_correction = with_route_roadblock_correction

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        map_api = scenario.map_api
        route_roadblock_ids: Tuple[str] = get_deduplicated_route_roadblock_ids(
            scenario, self._with_route_roadblock_correction)
        offset = prepared_scenario.get_offset()
        for iteration in iterations:
            ego_state = prepared_scenario.get_ego_state_at_iteration(iteration)
            center = self.calc_raster_center(ego_state)
            cache_key, center = self.calc_cache_key_and_center(center, offset)
            if prepared_scenario.feature_exists_at_center(
                    'route_roadblock_polygons', cache_key):
                continue
            block_polygons = get_route_polygon_from_roadblock_ids(
                map_api, center, self._cache_radius,
                route_roadblock_ids).to_vector()
            block_polygons = _to_shape_list(block_polygons, offset)
            prepared_scenario.add_feature_at_center('route_roadblock_polygons',
                                                    block_polygons, cache_key)

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        center = self.calc_raster_center(ego_state)
        cache_key = self.calc_cache_key(center)
        polygons = scenario.get_feature_at_center('route_roadblock_polygons',
                                                  cache_key)
        return draw_polygon_image(polygons, [1.0] * polygons.num,
                                  self._image_size, center, self._radius)


class DrivableAreaRasterBuilder(MapObjectRasterBuilder):
    """Raster representing the drivable area.

    Cached features:
        map_object_polygons_DRIVABLE_AREA (ShapeList): polygons of the drivable area.

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    """

    def __init__(self, image_size: int, radius: float,
                 longitudinal_offset: float):
        super().__init__(image_size, radius, longitudinal_offset,
                         ['DRIVABLE_AREA'])

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        center = self.calc_raster_center(ego_state)
        cache_key = self.calc_cache_key(center)
        coords = list(
            _get_prepared_coords(scenario, 'map_object_polygon_DRIVABLE_AREA',
                                 cache_key)[0])
        polygons = ShapeList(
            coords=np.concatenate(coords, axis=0),
            sizes=np.array([len(c) for c in coords]))
        colors = [1.0] * polygons.num
        return draw_polygon_image(polygons, colors, self._image_size, center,
                                  self._radius)


class BaselinePathsRasterBuilder(MapObjectRasterBuilder):
    """Raster representing the baseline paths.

    Cached features:
        map_object_linestring_{feature_name} (ShapeList): polylines of the baseline
            paths, where feature_name is either "LANE" or "LANE_CONNECTOR".

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    :param line_thickness: thickness of the baseline paths in pixels.
    """

    def __init__(self, image_size: int, radius: float,
                 longitudinal_offset: float, line_thickness: int):
        super().__init__(image_size, radius, longitudinal_offset,
                         ['LANE', 'LANE_CONNECTOR'])
        self._line_thickness = line_thickness

    def prepare_scenario_at_center(self, map_api: AbstractMap,
                                   prepared_scenario: PreparedScenario,
                                   center: Point2D) -> None:
        for feature_name in self._map_features:
            self.prepare_map_objects(map_api, prepared_scenario, center,
                                     feature_name, 'linestring')

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        all_coords = []
        center = self.calc_raster_center(ego_state)
        cache_key = self.calc_cache_key(center)
        for feature_name in self._map_features:
            all_coords.extend(
                _get_prepared_coords(scenario,
                                     'map_object_linestring_' + feature_name,
                                     cache_key)[0])
        if len(all_coords) == 0:
            return np.zeros((self._image_size, self._image_size), dtype=np.float32)
        all_polylines = ShapeList(
            coords=np.concatenate(all_coords, axis=0),
            sizes=np.array([len(c) for c in all_coords]))
        colors = [1.] * len(all_polylines.sizes)
        return draw_polyline_image(all_polylines, colors, self._line_thickness,
                                   self._image_size, center, self._radius)


class EgoSpeedRasterBuilder(RasterBuilderBase):
    """Raster representing with normlized speed of ego car.

    It is raster with all the pixels with a same value equal to the normalized
    speed of the ego car.

    The speed is obtained from EgoState.agent.velocity.

    Cached Features: None

    Note that the speed from EgoState is slightly different from that calculated
    based on the current and previous location of the ego vehicle, which is used
    by `raster_utils.get_speed_raster`.

    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the virtual center
    :param max_speed_normalizer: the speed used to normalize the speed limit.
        The normalized speed limit is `speed / max_speed_normalizer`.
    """

    def __init__(self,
                 image_size: int,
                 radius: float,
                 longitudinal_offset: float,
                 max_speed_normalizer: float = 16.0):
        super().__init__(image_size, radius, longitudinal_offset)
        self._max_speed_normalizer = max_speed_normalizer

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        return np.full((self._image_size, self._image_size),
                       (ego_state.vel_lon**2 + ego_state.vel_lat**2)**0.5 /
                       self._max_speed_normalizer,
                       dtype=np.float32)


class TrafficLightRasterBuilder(RasterBuilderBase):
    """
    A raster builder that creates a raster representation for traffic lights in the environment.
    
    This builder generates an image where traffic lights and their states are represented. Currently, it initializes an empty raster image with no traffic lights represented, serving as a placeholder for future implementations where traffic lights will be visually encoded based on their state and relevance to the ego vehicle.
    
    Parameters:
    - image_size (int): The height and width of the square raster image in pixels.
    - radius (float): The observation radius around the ego vehicle to include in the raster, in meters. Traffic lights outside this radius are not included.
    - longitudinal_offset (float): The offset from the center of the ego vehicle towards the front of the vehicle where the observation radius begins, in meters.
    """

    def __init__(self,
                 image_size: int,
                 radius: float,
                 longitudinal_offset: float):
        super().__init__(image_size, radius, longitudinal_offset)

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        return np.full((self._image_size, self._image_size),
                       0.0,
                       dtype=np.float32)


@alf.configurable(whitelist=["relative_overlay"])
class PastCurrentAgentsRasterBuilder(RasterBuilderBase):
    """Raster representing the past and current agents.

    Cached Features:
        tracked_object_boxes: Dict[int, np.ndarray[n, 5]): boxes of the tracked
            object for each iteration. Key is iteration. Each box is reparesented
            by 5 numbers: center_x, center_y, heading, half_length, half_width.
        ego_state: Dict[int, NpEgoState]: ego states for iterations before 0,
            up to `past_time_horizon`.
    :param image_size: size of the raster image
    :param radius: radius of the raster
    :param longitudinal_offset: longitudinal offset of the raster
    :param past_time_horizon: [s] time horizon of past agents
    :param past_num_steps: number of past steps to sample
    :param relative_overlay: If False, each agent is represented a series of boxes
        corresponding to their past and current pose. If True, each agent is represented
        as a series of boxes, where each box represents the pose of the agent
        relative to the pose of the ego vehicle at the corresponding iteration.
    """

    def __init__(self,
                 image_size: int,
                 radius: float,
                 longitudinal_offset: float,
                 past_time_horizon: float,
                 past_num_steps: int,
                 relative_overlay: bool = False):
        super().__init__(image_size, radius, longitudinal_offset)
        self._past_time_horizon = past_time_horizon
        self._past_num_steps = past_num_steps
        self._relative_overlay = relative_overlay
        self._key_name = "tracked_object_boxes"
        self._agent_type = ['vehicle', 'pedestrian', 'bicycle']

    def prepare_scenario(self, scenario: AbstractScenario,
                         prepared_scenario: PreparedScenario,
                         iterations: range) -> None:
        offset = prepared_scenario.get_offset()
        interval = scenario.database_interval

        if self._past_time_horizon > 0:
            history_window = int(self._past_time_horizon / interval)
            assert history_window % self._past_num_steps == 0, (
                f"history window {history_window} must be divisible by "
                f"past_num_steps {self._past_num_steps}")
            history_step = history_window // self._past_num_steps
        else:
            history_window = 0
            history_step = 1

        # 1. get all the history iteration and current iteration for iterations
        all_iterations = set()
        for iteration in iterations:
            all_iterations.update(
                range(iteration - history_window, iteration + 1, history_step))
        all_iterations = sorted(all_iterations)

        # 2. find the earliest unprepared iteration
        earliest_unprepared_iteration = None
        for iteration in all_iterations:
            if not prepared_scenario.feature_exists_at_iteration(
                    self._key_name, iteration):
                earliest_unprepared_iteration = iteration
                break

        # 3. Get past tracked objects
        if earliest_unprepared_iteration is not None and earliest_unprepared_iteration < 0:
            num_samples = -earliest_unprepared_iteration
            if num_samples == 1 and isinstance(scenario, NuPlanScenario):
                # 0.5 * iterval is due to an artifact of NuPlanScenario.get_past_tracked_object()
                # that when num_samples=1, it actually gets a sample at one
                # iteration before the given iteration. This is caused by
                # scenario_utils.sample_indices_with_time_horizon()
                # See nuplan issue https://github.com/motional/nuplan-devkit/issues/348
                # TODO: remove this hack when the issue is fixed.
                time_horizon = 0.5 * interval
            else:
                time_horizon = num_samples * interval
            past_detections = list(
                scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=time_horizon,
                    num_samples=num_samples))
            assert len(past_detections) == num_samples
            past_ego_states = list(
                scenario.get_ego_past_trajectory(
                    iteration=0,
                    time_horizon=time_horizon,
                    num_samples=num_samples))

        # 4. prepare polygons for all iterations
        for iteration in all_iterations:
            if prepared_scenario.feature_exists_at_iteration(
                    self._key_name, iteration):
                continue
            if iteration < 0:
                detections = past_detections[iteration]
                past_ego_state = past_ego_states[iteration]
                prepared_scenario.add_ego_state_at_iteration(
                    past_ego_state, iteration)
            else:
                detections = scenario.get_tracked_objects_at_iteration(
                    iteration)
            polygons = self._get_detection_boxes(detections, offset)
            prepared_scenario.add_feature_at_iteration(self._key_name,
                                                       polygons, iteration)

    def _get_detection_boxes(self, detections: DetectionsTracks,
                             offset) -> npt.NDArray[np.float32]:
        if len(detections.tracked_objects) == 0:
            return np.empty((0, 5), dtype=np.float32)
        x0 = float(offset[0])
        y0 = float(offset[1])

        def _extract(box: OrientedBox) -> np.ndarray:
            center = box.center
            return (center.x - x0, center.y - y0, center.heading,
                    box.half_length, box.half_width)
        def _in_type(tracked_object_type) -> bool:
            return tracked_object_type.fullname in self._agent_type
        boxes = [_extract(o.box) for o in detections.tracked_objects if _in_type(o.tracked_object_type)]
        return np.array(boxes, dtype=np.float32)

    def get_features_from_prepared_scenario(
            self, scenario: PreparedScenario, iteration: int,
            ego_state: NpEgoState) -> npt.NDArray[np.float32]:
        all_boxes = []
        all_colors = []
        step_interval = int(self._past_time_horizon /
                            scenario.database_interval / self._past_num_steps)
        for i in range(self._past_num_steps + 1):
            iter = iteration - (self._past_num_steps - i) * step_interval
            boxes = scenario.get_feature_at_iteration(self._key_name,
                                                      iter)
            if boxes.size > 0:
                if self._relative_overlay:
                    prev_ego_state = scenario.get_ego_state_at_iteration(iter)
                    boxes = self._adjust_boxes_relative_position(
                        boxes, prev_ego_state, ego_state)
                all_boxes.append(boxes)
                all_colors.extend(
                    [(i + 1) / (self._past_num_steps + 1)] * boxes.shape[0])

        if len(all_boxes) > 0:
            all_boxes = np.concatenate(all_boxes, axis=0)
            coords = box_to_corners(all_boxes).reshape(-1, 2)
        else:
            coords = np.empty((0, 2), dtype=np.float32)
        center = self.calc_raster_center(ego_state)
        return draw_polygon_image(
            ShapeList(coords, np.array([4] * len(all_boxes), dtype=np.int32)),
            all_colors, self._image_size, center, self._radius)

    def _adjust_boxes_relative_position(self, boxes, prev_ego_state,
                                        ego_state):
        boxes = boxes.copy()
        current_center = self.calc_raster_center(ego_state)
        prev_center = self.calc_raster_center(prev_ego_state)
        trans1 = get_global_to_local_transform(prev_center)
        trans2 = get_local_to_global_transform(current_center)
        trans = (trans2 @ trans1).astype(np.float32)
        boxes[:, :2] = boxes[:, :2] @ trans[:2, :2].T + trans[:2, 2]
        dh = current_center.heading - prev_center.heading
        boxes[:, 2] += dh

        return boxes

@alf.configurable(whitelist=["relative_overlay"])
class StaticAgentsRasterBuilder(PastCurrentAgentsRasterBuilder):
    """
    A raster builder that creates a raster representation for static agents in the environment.
    
    This builder generates an image where static agents (like parked cones, barriers, etc.) are represented. Currently, it initializes an empty raster image with no agents represented, serving as a placeholder for future implementations where static agents will be visually encoded.
    
    Parameters:
    - image_size (int): The height and width of the square raster image in pixels.
    - radius (float): The observation radius around the ego vehicle to include in the raster, in meters. Agents outside this radius are not included.
    - longitudinal_offset (float): The offset from the center of the ego vehicle towards the front of the vehicle where the observation radius begins, in meters.
    """
    def __init__(self, image_size: int, radius: float, longitudinal_offset: float, past_time_horizon: float, past_num_steps: int, relative_overlay: bool = False):
        super().__init__(image_size, radius, longitudinal_offset, past_time_horizon, past_num_steps, relative_overlay)
        self._agent_type = ['traffic_cone', 'barrier', 'czone_sign', 'generic_object']
