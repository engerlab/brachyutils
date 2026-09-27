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

class MOO_Ax(MOO):
    def __init__(
        self,
        catheter_table_optim: CatheterTableOptim_Gurobi,
        parameter_space: Dict[str, np.typing.ArrayLike],
        max_workers: int = 16,
        normalize = False,
        sampler_name_id:Literal["fast", "quality"]="quality",
        ):
        self.tuner: Client
        super().__init__(
            catheter_table_optim= catheter_table_optim,
            parameter_space= parameter_space,
            max_workers= max_workers,
            normalize= normalize
            )
        self.sampler_name_id = sampler_name_id
        self._ax_parameters = None

        # fill out the ax-specific attributes
        self._parameter_space_to_ax()
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
                    f"{_clean_dvh_names(dvh_name)} <= {self.dvh_metric_goals[dvh_name][1]}"
                )
            else:
                ax_objectives.append(
                    f"{_clean_dvh_names(dvh_name)}"
                )
                outcome_constraints.append(
                    f"{_clean_dvh_names(dvh_name)} >= {self.dvh_metric_goals[dvh_name][1]}"
                )
        ax_objectives = ", ".join(ax_objectives)
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

    def run_warmups(self, n_warmups):
        self.tuner.configure_generation_strategy(
            method="random_search",
            initialization_budget=n_warmups,)
        trials = self.tuner.get_next_trials(max_trials=n_warmups)
        param_trials = self.get_parameters_from_trials(trials=trials)
        objectives = self.objectives(parameters=param_trials, sampler_name_id="random_search")
        self.attach_objectives_to_trials(trials=trials, observed_objectives=objectives)

    def run_trials(self, n_trials: int, batch_size: int = 1):
        self.tuner.configure_generation_strategy(
            method=self.sampler_name_id,
            # simplify_parameter_changes=True,
            torch_device="cuda" if cuda.is_available() else "cpu")

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
    