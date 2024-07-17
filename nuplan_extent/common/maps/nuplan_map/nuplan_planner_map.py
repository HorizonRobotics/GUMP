from typing import List, Union

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.imapsdb import IMapsDB
from nuplan_extent.common.maps.nuplan_map.route_planner.global_lane_route_planner import \
    GlobalLaneRoutePlanner


class NuPlanPlannerMap(NuPlanMap):
    def __init__(self, maps_db: IMapsDB, map_name: str) -> None:
        """This function initializes an instance of a class that inherits from NuPlanMap, which support the planner.

        Args:
            maps_db (IMapsDB): MapsDB instance.
            map_name (str): Name of the map.
        """
        super().__init__(maps_db, map_name)
        self.global_route_planner = GlobalLaneRoutePlanner(self)

    def path_search(self, origin: Union[StateSE2, Point2D],
                    destination: Union[StateSE2, Point2D]) -> List[Point2D]:
        """
        This function return the navigation path searched from lane graph
        Args:
            origin (Point2D): 2D global position of origin
            destination (Point2D): 2D global position of destination
        Return:
            List of Point2D: navigation trajectory
        """
        return self.global_route_planner.path_search(origin, destination)

    def block_path_search(self, origin: Union[StateSE2, Point2D],
                          destination: Union[StateSE2, Point2D]
                          ) -> List[RoadBlockGraphEdgeMapObject]:
        """This function return the navigation block path searched from lane graph

        Args:
            origin (Union[StateSE2, Point2D]): 2D global position of origin
            destination (Union[StateSE2, Point2D]): 2D global position of destination

        Returns:
            List[RoadBlockGraphEdgeMapObject]: List of passed roadblocks
        """
        return self.global_route_planner.block_path_search(origin, destination)
