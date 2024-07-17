import logging
import os
from typing import List, Optional, Type, cast

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import \
    AbstractScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool
from nuplan_extent.common.maps.wod_map.map_factory import WodMapFactory
from nuplan_extent.planning.scenario_builder.wod_db.wod_scenario import \
    WodScenario

logger = logging.getLogger(__name__)


class WodScenarioBuilder(AbstractScenarioBuilder):
    """Builder class for constructing Waymo Open Dataset scenarios for training and simulation."""

    def __init__(
            self,
            data_root: str,
            training_token_list_path: Optional[str] = None,
            validation_token_list_path: Optional[str] = None,
            force_scenario_creation: bool = False,
            db_files: Optional[str] = None,
            subsample_ratio: Optional[int] = 1,
            start_index: Optional[int] = 0,
            **kwargs,
    ):
        self.data_root = data_root
        self._db_files = db_files
        self.training_token_list_path = training_token_list_path
        self.validation_token_list_path = validation_token_list_path
        self.subsample_ratio = subsample_ratio
        self.start_index = start_index

    @classmethod
    def get_scenario_type(cls) -> Type[AbstractScenario]:
        """Inherited. See superclass."""
        return cast(Type[AbstractScenario], WodScenario)

    def get_map_factory(self) -> WodMapFactory:
        """Inherited. See superclass."""
        return WodMapFactory(self.data_root)

    def get_scenarios(self, scenario_filter: ScenarioFilter,
                      worker: WorkerPool) -> List[AbstractScenario]:
        """
        Retrieve filtered scenarios from the database.
        :param scenario_filter: Structure that contains scenario filtering instructions.
        :param worker: Worker pool for concurrent scenario processing.
        :return: A list of scenarios.

        Currently not using filters(no need) and workers(not necessary).
        """

        if self.training_token_list_path is None:
            self.training_token_list_path = os.path.join(
                self.data_root, 'raw_tokens', 'training_token_list.txt')
        if self.validation_token_list_path is None:
            self.validation_token_list_path = os.path.join(
                self.data_root, 'raw_tokens', 'validation_token_list.txt')
        with open(self.training_token_list_path, 'r') as f:
            self.training_token_list = [
                line.strip() for line in f if line.strip()
            ]
            self.training_token_list = self.training_token_list[::self.subsample_ratio]
        with open(self.validation_token_list_path, 'r') as f:
            self.validation_token_list = [
                line.strip() for line in f if line.strip()
            ]
            self.validation_token_list = self.validation_token_list[self.start_index::self.subsample_ratio]
        scenarios = []
        for token in self.training_token_list:
            scenario_id, agent_idx = token.split('_')
            scenarios.append(
                WodScenario(self.data_root, "training", scenario_id,
                            agent_idx))
        for token in self.validation_token_list:
            scenario_id, agent_idx = token.split('_')

            if "interactive" in self.validation_token_list_path:
                scenarios.append(
                    WodScenario(self.data_root, "validation_interactive",
                                scenario_id, agent_idx))
            elif "testing" in self.validation_token_list_path:
                scenarios.append(
                    WodScenario(self.data_root, "testing", scenario_id,
                                agent_idx))
            else:
                scenarios.append(
                    WodScenario(self.data_root, "validation", scenario_id,
                                agent_idx))
        return scenarios
