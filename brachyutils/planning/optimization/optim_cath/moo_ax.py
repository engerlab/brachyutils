import numpy as np
from typing import Dict, List, Literal
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
)
from brachyutils.planning.optimization.optim_cath.moo import (
    MOO, evaluate_parameters, are_acceptable, get_hyper_volume)

class MOO_Ax(MOO):
    def __init__(
        self,
        catheter_table_optim: CatheterTableOptim_Gurobi,
        parameter_space: Dict[str, np.typing.ArrayLike],
        max_workers: int = 16,
        normalize = False,
        ):

        super().__init__(
            catheter_table_optim= catheter_table_optim,
            parameter_space= parameter_space,
            max_workers= max_workers,
            normalize= normalize
            )
        self.set_tuner()

    def set_tuner(self):
        pass
        