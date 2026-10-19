import numpy as np
from typing import Dict, List, Any, Literal
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
)
from brachyutils.planning.optimization.optim_cath.moo import MOO
from torch import cuda

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.adapter.registry import Generators
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationStep
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from ax.core.utils import get_pending_observation_features

from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.adapter.registry import Generators

class MOO_Ax(MOO):
    def __init__(
        self,
        catheter_table_optim: CatheterTableOptim_Gurobi,
        parameter_space: Dict[str, np.typing.ArrayLike],
        max_workers: int = 16,
        normalize = False,
        anchor_dvh_metric: str = None,
        sampler_name_id:Literal["fast", "quality", "qNEHVI"]="qNEHVI",
        slack_space_shrinkage: float = None,
        slack_objective_threshold: float | Dict[str, float] = 0,
        device: Literal["cpu", "cuda"] = "cpu",
        ):
        r"""
        ### Purpose:
        - Configure Ax as the backend for multi-objective optimization.

        - For the shared optimization inputs, DVH goals, parameter-space format,
        and inherited behavior, see the `MOO` documentation in `moo.py`.

        - The instance exposes the Ax `Client` as `tuner`, along with the
        configured sampler and objective slack values.

        ### Inputs:
        The following inputs are specific to the Ax backend:
        - `sampler_name_id`: Ax generation strategy to use: `"fast"`,
            `"quality"`, or `"qNEHVI"` (default).
        - `device`: `"cpu"` or `"cuda"`; selects the device for Ax's
            BoTorch optimization when using `"qNEHVI"`.
        """
        self.tuner: Client
        super().__init__(
            catheter_table_optim= catheter_table_optim,
            parameter_space= parameter_space,
            max_workers= max_workers,
            normalize= normalize,
            anchor_dvh_metric=anchor_dvh_metric,
            slack_space_shrinkage=slack_space_shrinkage,
            slack_objective_threshold=slack_objective_threshold
            )
        self.sampler_name_id = sampler_name_id
        self.device = device
        self._ax_parameters = None
        self._ax_constraints = None
        self._ax_objectives = None

        if self.device == "cuda":
            if not cuda.is_available:
                self.device = "cpu"
        # fill out the ax-specific attributes
        self.set_tuner()

    def _parameter_space_to_ax(self):
        r"""
        ### Purpose:
        - To convert the brachy MOO parameters to ax parameters stored at
        `self._ax_parameters`
        """
        self._ax_parameters = []
        for param_space in self.parameter_space:
            ax_param = RangeParameterConfig(
                name=param_space,
                bounds=self.parameter_space[param_space],
                parameter_type="float",
            )
            self._ax_parameters.append(ax_param)

    def set_tuner(self):
        self._parameter_space_to_ax()
        # # instantiate a new tuner object. in ax it's called Client
        self.tuner = Client()
        self.tuner.configure_experiment(parameters = self._ax_parameters)
        # # build the objective string
        ax_objectives = []
        outcome_constraints = []
        for dvh_name, direction in self.directions.items():
            if direction == "minimize":
                ax_objectives.append(
                    f"-{_clean_dvh_names(dvh_name)}"
                )
                outcome_constraints.append(
                    f"{_clean_dvh_names(dvh_name)} <= {
                        self.dvh_metric_goals[dvh_name][1]
                        *(1+self.slack_objective_threshold.get(dvh_name, 0))}"
                )
            else:
                ax_objectives.append(
                    f"{_clean_dvh_names(dvh_name)}"
                )
                outcome_constraints.append(
                    f"{_clean_dvh_names(dvh_name)} >= {
                        self.dvh_metric_goals[dvh_name][1]
                        *(1-self.slack_objective_threshold.get(dvh_name, 0))}"
                )
        ax_objectives = ", ".join(ax_objectives)
        self._ax_objectives = ax_objectives
        self._ax_constraints = outcome_constraints

        self.tuner.configure_optimization(
            objective=ax_objectives,
            outcome_constraints=outcome_constraints
        )

    def objectives(self, parameters, sampler_name_id):
        observed_objects = super().objectives(parameters, sampler_name_id)
        old_columns = observed_objects.columns
        new_columns = []
        for col in old_columns:
            new_columns.append(_clean_dvh_names(col))
        observed_objects.columns = new_columns
        return observed_objects

    def run_random_sampling(
        self,
        n_samples,
        use_for_warmup: bool = True
        ):
        self.tuner.configure_generation_strategy(
            method="random_search",
            initialization_budget=n_samples,)
        trials = self.tuner.get_next_trials(max_trials=n_samples)
        param_trials = self.get_parameters_from_trials(trials=trials)
        objectives = self.objectives(parameters=param_trials, sampler_name_id="random_search")
        if self.slack_space_shrinkage is not None:
            acceptable_trials = self.shrink_parameter_space(return_acceptable_trials=True)
            self.set_tuner()
            true_indices = acceptable_trials.index[acceptable_trials["acceptable"]].tolist()
            new_trials = {}
            for i, true_indx in enumerate(true_indices):
                new_trials[i]=trials[true_indx]
            trials = new_trials
        if use_for_warmup:
            self.attach_objectives_to_trials(trials=trials, observed_objectives=objectives)
        else:
            self.set_tuner()

    def run_trials(self, n_trials: int, batch_size: int = 1):
        if self.sampler_name_id == "qNEHVI":
            nodes = [
                GenerationStep(
                generator=Generators.SOBOL,
                num_trials=1,),
                GenerationStep(
                generator=Generators.BOTORCH_MODULAR,
                num_trials=-1,
                model_kwargs={
                    "botorch_acqf_class": qLogNoisyExpectedHypervolumeImprovement,
                    "torch_device": self.device,
                    },)
                ]
            self.tuner.set_generation_strategy(
                GenerationStrategy(nodes=nodes, name=self.sampler_name_id))
        else:
            self.tuner.configure_generation_strategy(
                method=self.sampler_name_id,
                initialization_budget=1,
                allow_exceeding_initialization_budget=False,
                use_existing_trials_for_initialization=True,
                simplify_parameter_changes=True,
                torch_device=self.device)

        for _ in range(n_trials):
            trials = self.tuner.get_next_trials(max_trials=batch_size)
            param_trials = self.get_parameters_from_trials(trials=trials)
            objectives = self.objectives(parameters=param_trials, sampler_name_id=self.sampler_name_id)
            self.attach_objectives_to_trials(trials=trials, observed_objectives=objectives)

    def get_parameters_from_trials(self, trials: List[Any]) -> pd.DataFrame:
        """
        ### Purpose:
        - To extract the plan optimization parameters (penalty weights) from Ax trial
        objects.

        ### Inputs:
        trials := 

        ### Outputs:
        - parameters_df := a dataframe with the name of the parameters as columns and
        their values in the rows.
        """
        return pd.DataFrame(trials).T

    def attach_objectives_to_trials(self, trials, observed_objectives):
        for trial, (_, row) in zip(trials.items(), observed_objectives.iterrows()):
            if not trial[0] in self.tuner._experiment.trials:
                self.tuner.attach_trial(trial[1])
            self.tuner.complete_trial(trial_index=trial[0], raw_data=row)

def _clean_dvh_names(dvh_name:str) -> str:
    r"""
    ### Purpose:
    - remove %, (, ) characters from the dvh names and replace them with _.
    so V150%(CTV) -> V150_CTV
    """
    dvh_name = dvh_name.replace("%", "")
    dvh_name = dvh_name.replace("(", "_")
    dvh_name = dvh_name.replace(")", "")
    return dvh_name
