import numpy as np
from typing import Dict, List, Literal
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

    def objectives(self, parameters):
        pass

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

    def run_warmups(self, n_warmups, multi_proc = True):
        pass
    def run_trials(self, n_trials):
        pass

def _clean_dvh_names(dvh_name:str) -> str:
    r"""
    ### Purpose:
    - remove %, (, ) characters from the dvh names and replace them with _.
    so V150%(CTV) -> V150_CTV
    """
    dvh_name = dvh_name.replace("%", "_")
    dvh_name = dvh_name.replace("(", "")
    dvh_name = dvh_name.replace(")", "")
    return dvh_name
    