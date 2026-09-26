import numpy as np
from typing import Dict, List, Any
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
)
from brachyutils.planning.optimization.optim_cath.moo import (
    MOO, evaluate_parameters, are_acceptable, get_hyper_volume)

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

class MOO_Ax(MOO):
    def __init__(
        self,
        catheter_table_optim: CatheterTableOptim_Gurobi,
        parameter_space: Dict[str, np.typing.ArrayLike],
        max_workers: int = 16,
        normalize = False,
        ):
        self.tuner: Client
        super().__init__(
            catheter_table_optim= catheter_table_optim,
            parameter_space= parameter_space,
            max_workers= max_workers,
            normalize= normalize
            )
        self._ax_parameters = None

        # fill out the ax-specific attributes
        self._parameter_space_to_ax()
        self.set_tuner()

    def objectives(
        self,
        trials: List[Any],
        sampler_name_id:str):
        trial_params = self.get_params_from_ax_trials(trials)
        dvh_metrics_data = evaluate_parameters(
            trial_params,
            self.catheter_table_optim,
            max_workers=self.max_workers,
        )

        acceptable_trials = are_acceptable(dvh_metrics_data, self.dvh_metric_goals)
        hv_trials = get_hyper_volume(dvh_metrics_data, self.dvh_metric_goals)
        sampler_df = pd.Series(
            [sampler_name_id for _ in range(len(trial_params))],
            name="sampler_name_id").to_frame()
        self.trial_data = pd.concat([
            self.trial_data,
            pd.concat([
                trial_params, dvh_metrics_data,
                acceptable_trials, hv_trials, sampler_df], axis=1)
        ], axis=0)
        self.trial_data.reset_index(drop=True, inplace=True)

        # # Attach the observed DVH metrics to each trial for the constraints_func to use.
        # # XXX do this for AX!
        for trial, (_, row) in zip(trials, dvh_metrics_data.iterrows()):
            trial.set_user_attr("dvh_metrics", row.to_dict())

        objectives = dvh_metrics_data[list(self.dvh_metric_goals.keys())].values.tolist()
        return objectives

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
        for dvh_name, direction in self.directions.items():
            if direction == "minimize":
                ax_objectives.append(
                    f"-{_clean_dvh_names(dvh_name)}"
                )
            else:
                ax_objectives.append(
                    f"{_clean_dvh_names(dvh_name)}"
                )
        ax_objectives = ", ".join(ax_objectives)
        self.tuner.configure_optimization(
            objective=ax_objectives
        )

    def run_warmups(self, n_warmups):
        self.tuner.configure_generation_strategy(
            method="random_search",
            initialization_budget=n_warmups,)
        trials = self.tuner.get_next_trials(max_trials=n_warmups)
        objectives = self.objectives(trials=trials, sampler_name_id="random_search")

    def run_trials(self, n_trials):
        pass ### TODO

    def get_params_from_ax_trials(trials: List[Any]) -> pd.DataFrame:
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
        pass

    def attach_objectives_to_trials(self, trials, observed_objectives):
        pass ### TODO

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
    