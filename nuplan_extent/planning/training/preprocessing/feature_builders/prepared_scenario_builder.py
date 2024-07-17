from typing import List, Optional, Type
from hydra.utils import instantiate

from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from omegaconf import DictConfig

from nuplan_extent.planning.training.preprocessing.features.raster_builders import PreparedScenarioFeatureBuilder
from nuplan_extent.planning.scenario_builder.prepared_scenario import PreparedScenario, ScenarioPreparer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class PreparedScenarioBuilder(AbstractFeatureBuilder):
    """This class builds the prepared scenario from a scenario.

    get_features_from_scenario() returns a PreparedScenario object so that the
    NuPlan's caching pipeline can cache the prepared scenario.

    :param iteration_step: the step size of the iterations to be included in the
        prepared scenario.
    :param builders: a list of feature builders that will be used to build
        features for the prepared scenario.
    :param model_cfg: the model configuration that contains the feature builders
    """

    def __init__(self,
                 iteration_step: int,
                 builders: List[ScenarioPreparer] = [],
                 model_cfg: Optional[DictConfig] = None):
        super().__init__()
        self._iteration_step = iteration_step
        self._builders = []
        for builder in builders:
            assert isinstance(builder, ScenarioPreparer)
            self._builders.append(builder)
        if model_cfg is not None:
            for cfg in model_cfg.feature_builders:
                builder = instantiate(cfg)
                assert isinstance(builder, PreparedScenarioFeatureBuilder)
                self._builders.append(builder)

            for cfg in model_cfg.target_builders:
                assert isinstance(builder, PreparedScenarioFeatureBuilder)
                self._builders.append(builder)

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "prepared_scenario"

    @classmethod
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        return PreparedScenario

    def get_features_from_scenario(
            self, scenario: AbstractScenario) -> PreparedScenario:
        prepared_scenario = PreparedScenario()
        iterations = range(0, scenario.get_number_of_iterations(),
                           self._iteration_step)
        prepared_scenario.prepare_scenario(scenario, iterations)
        for builder in self._builders:
            builder.prepare_scenario(scenario, prepared_scenario, iterations)
        return prepared_scenario

    def get_features_from_simulation(
            self,
            current_input: PlannerInput,
            initialization: PlannerInitialization,
    ) -> PreparedScenario:
        raise NotImplementedError()


class PreparedScenarioModelWrapper(TorchModuleWrapper):
    """This class wraps the prepared scenario builder into a TorchModuleWrapper.

    NuPlan's caching pipeline obtains the feature/target builders from a model
    of type TorchModuleWrapper to cache the prepared. So we need a dummy model
    that wraps the prepared scenario builder.

    :param prepared_scenario_builder: the prepared scenario builder
    :param future_trajectory_sampling: the future trajectory sampling parameters
    """

    def __init__(self, prepared_scenario_builder: PreparedScenarioBuilder,
                 future_trajectory_sampling: TrajectorySampling):
        super().__init__(
            feature_builders=[prepared_scenario_builder],
            target_builders=[],
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self._prepared_scenario_builder = prepared_scenario_builder

    def forward(self, input_features: FeaturesType) -> TargetsType:
        return {}
