from typing import List

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool


class WodSplitter(AbstractSplitter):
    """
    Splitter that splits database to lists of samples for each of train/val/test
    sets based on the split attribute of the scenario.
    """

    def __init__(self, training_token_list_path: str,
                 validation_token_list_path: str,
                 subsample_ratio: int = 1,
                 start_index: int = 0) -> None:
        """
        Initializes the class.

        """
        self.start_index = start_index
        with open(training_token_list_path, 'r') as f:
            self.training_token_list = [
                line.strip() for line in f if line.strip()
            ][self.start_index::subsample_ratio]
        with open(validation_token_list_path, 'r') as f:
            self.validation_token_list = [
                line.strip() for line in f if line.strip()
            ][self.start_index::subsample_ratio]
        self.training_token_list = set(self.training_token_list)
        self.validation_token_list = set(self.validation_token_list)

    def get_train_samples(self, scenarios: List[AbstractScenario],
                          worker: WorkerPool) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return [
            scenario for scenario in scenarios
            if scenario.token in self.training_token_list
        ]

    def get_val_samples(self, scenarios: List[AbstractScenario],
                        worker: WorkerPool) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return [
            scenario for scenario in scenarios
            if scenario.token in self.validation_token_list
        ]

    def get_test_samples(self, scenarios: List[AbstractScenario],
                         worker: WorkerPool) -> List[AbstractScenario]:
        """Inherited, see superclass."""
        return [
            scenario for scenario in scenarios
            if scenario.token in self.validation_token_list
        ]
