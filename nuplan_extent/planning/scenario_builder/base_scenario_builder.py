import logging
import pickle
from typing import Dict, List

import tqdm

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import \
    AbstractScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool

logger = logging.getLogger(__name__)


class BaseScenarioBuilder(AbstractScenarioBuilder):
    """Builder class for constructing Base scenarios for training and simulation."""

    @staticmethod
    def load_cached_data_queues(cache_path: str):
        """Load data queues cache."""
        with open(cache_path, 'rb') as handle:
            data_queues = pickle.load(handle)
        return data_queues

    @staticmethod
    def cache_data_queue(
            data_queues: List[Dict],
            cache_path: str,
    ):
        """Cache data queues to local."""
        with open(cache_path, 'wb') as handle:
            pickle.dump(data_queues, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        if getattr(self, 'indices', None) is not None:
            return len(self.indices)
        else:
            return len(self.data_queues)

    def get_scenarios(self, scenario_filter: ScenarioFilter,
                      worker: WorkerPool) -> List[AbstractScenario]:
        """
        Retrieve filtered scenarios from the database.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: A list of scenarios.

        Currently not using filters(no need) and workers(not necessary).
        """
        logger.info("Creating scenes...")
        scenes = []
        if self.use_cache:
            self.data_queues = self.data_queues[self.start_idx::self.
                                                load_interval]

            if self.skip_training:
                train_queues = []
                val_queues = []
                for data_queue in self.data_queues:
                    if data_queue['split'] == 'train':
                        train_queues.append(data_queue)
                    else:
                        val_queues.append(data_queue)
                self.data_queues = train_queues[::200] + val_queues

            for idx in tqdm.tqdm(range(len(self.data_queues))):
                data_queue = self.data_queues[idx]
                scenes.append(self._convert_to_scene(data_queue))
            del self.data_queues
        else:
            data_queues = []
            for idx in tqdm.tqdm(range(0, len(self), self.load_interval)):
                data_queue = self._prepare_data_queue(idx)
                data_queues.append(data_queue)
                scenes.append(self._convert_to_scene(data_queue))
            self.cache_data_queue(data_queues, self.cache_path)
            exit()
        return scenes

    def get_map_factory(self):
        """
        Get a map factory instance.
        Currently no map factory for nuscenes.
        """
