from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple, Type

from nuplan.common.maps.abstract_map_factory import AbstractMapFactory
from nuplan_extent.common.maps.wod_map.wod_map import WodMap


class WodMapFactory(AbstractMapFactory):
    """
    Factory creating maps from an pickle file
    """

    def __init__(self, data_root: str):
        """
        :param data_root: map data root
        """
        self._data_root = data_root

    def __reduce__(self) -> Tuple[Type[WodMapFactory], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return self.__class__, (self._maps_db, )

    def build_map_from_name(self, scenario_id: str) -> WodMap:
        """
        Builds a map interface given a map name.
        Examples of names: 'sg-one-north', 'us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood'
        :param map_name: Name of the map.
        :return: The constructed map interface
        """
        return WodMap(self._data_root, scenario_id)


@lru_cache(maxsize=32)
def get_maps_api(data_root: str, scenario_id: str) -> WodMap:
    return WodMap(data_root, scenario_id)
