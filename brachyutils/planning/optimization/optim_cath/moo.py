import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod
import pandas as pd

from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
    update_penalty_weights_and_voxel_goals,
    )

from brachyutils.planning.optimization.optim_gurobi import (
    get_optimized_dwelltimes_from_model
)
class MOO(ABC):
    _valid_parameter_names = [
        "dose_voxel_goal",
        "penalty_weight_linear",
        "penalty_weight_quadratic",
        "penalty_weight_hotspot",
        "hotspot_threshold",
        "penalty_weight_uniformity",
        "penalty_weight_variance_time",
    ]

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
        self.dvh_metric_goals: Dict[str, float] = None
        self.tuner: Any = None
        self.trial_data:pd.DataFrame = None
        # # Fill out the attributes
        self.validate_init()

    @abstractmethod
    def validate_init(self):
        r"""
        ### Purpose:
        - To ensure `self.catheter_table_optim` and `parameter_space` contain
        the correct information.
        
        ### Inputs:
        None := Expects the following to be filled already:
        - `self.catheter_table_optim`
        - `self.parameter_space`
        
        ### Outputs:
        None := Fills out the following attributes:
        - `self.dvh_metric_goals` := maps the DVH names {metric_name(structure_name)}
        to their clinically desired values.
        - `self.trial_data`: pd.DataFrame := A master dataframe containing the result of
        all the trials. The columns are parameter names from the keys of 
        `self.parameter_space` and the dvh metric names from the keys of 
        `self.dvh_metric_goals`.  
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
                if parameter_name in MOO._valid_parameter_names:
                    parameter_found = True

            if not structure_found:
                raise ValueError(f"The structure: {structure_name} was not found in the \
 optimization config dict of the plan")            
            if not parameter_found:
                raise ValueError(f"The parameter: {parameter_name} was not found in the \
 as a valid optimization parameter. Please see `Optimization_Config.to_dict()`")

        # Now build the columns of the 
        self.trial_data = pd.DataFrame(
            columns=(
                list(self.parameter_space.keys())
                +list(self.dvh_metric_goals.keys())))

    @abstractmethod
    def evaluate(self, parameters: np.DataFrame) -> pd.DataFrame:
        r"""
        Evaluates the parameters and returns the observed dvh metrics 
        corresponding to those parameters.
        """
        pass

    @abstractmethod
    def set_tuner(self):
        r"""
        ### Purpose:
        Builds the Tuner object that will recommend the next batch of parameter
        queries to be evaluated.
        """
        