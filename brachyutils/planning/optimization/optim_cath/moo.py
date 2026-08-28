import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod

from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
    update_penalty_weights_and_voxel_goals,
    )

from brachyutils.planning.optimization.optim_gurobi import (
    get_optimized_dwelltimes_from_model
)
class MOO(ABC):
    @abstractmethod
    def __init__(
        self,
        catheter_table_optim: CatheterTableOptim_Gurobi,
        parameter_space: Dict[str, np.typing.ArrayLike],
        batch_size: int):
        r"""
        ### Purpose:
        - The multi-objective optimization class performs hyper-parameter tuning
        to yield clinically acceptable treatment plans with Pareto optimal DVHs.
        The definition of clinically acceptale is set by the dvh metric goals inside
        the BrachyPlan of the `catheter_table_optim`.
        ### Inputs:
        - `catheter_table_optim` := An optimization object with a BrachyPlan and
        a Gurobi model. The plan shold have DVH metrics goal loaded.
        - `parameter_space` := A dictionary mapping the names of the parameters to be
        optimized to their range [min, max]. The names of the parameters are 
        some attributes of the Optimization_Config class followed by the name of
        that structure in paranthesis. For example:
            {
                penalty_weight_linear(CTV) : [1, 500]
            }
        - `batch_size` := Number of parallel trials. This can change in the future. 
        """        
        self.catheter_table_optim = catheter_table_optim
        self.parameter_space = parameter_space
        self.batch_size = batch_size
        # # Attributes to be filled out
        self.dvh_metric_goals = None

        self.validate_init()

    @abstractmethod
    def validate_init(self):
        r"""
        ### Purpose:
        - To ensure `self.catheter_table_optim` and `parameter_space` contain
        the correct information.
        """
        self.dvh_metric_goals = self.catheter_table_optim.plan.dvh_metric_goals
        if (
            self.dvh_metric_goals is None
            or len(self.dvh_metric_goals) == 0
            or isinstance(self.dvh_metric_goals, list)):
            raise ValueError("The DVH metric goal dictionary is essential for \
multi-objective optimization. please provide it to the optimization object.") 

        for key in self.parameter_space.keys():
            structure_name = key.split("(")[-1].split(")")[0]
            parameter_name = key.split("(")[0]

            structure_found = False
            parameter_found = False
            for optim_config in self.catheter_table_optim.plan.optimization_config_dict.values():
                if optim_config.structure_name == structure_name:
                    structure_found = True
                if optim_config.to_dict().get(parameter_name, None) is not None:
                    parameter_found = True

            if not structure_found:
                raise ValueError(f"The structure: {structure_name} was not found in the \
 optimization config dict of the plan")            
            if not parameter_found:
                raise ValueError(f"The parameter: {parameter_name} was not found in the \
 as a valid optimization parameter. Please see `Optimization_Config.to_dict()`")