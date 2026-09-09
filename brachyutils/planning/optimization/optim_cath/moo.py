import numpy as np
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
    update_penalty_weights_and_voxel_goals,
    )

from brachyutils.planning.optimization.optim_gurobi import (
    get_optimized_dwelltimes_from_model
)

import optuna
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        ):
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
        """        
        self.catheter_table_optim = catheter_table_optim
        self.parameter_space = parameter_space
        # # Attributes to be filled out
        self.dvh_metric_goals: Dict[str, List[str, float]] = None
        self.tuner: Any = None
        self.trial_data:pd.DataFrame = None
        # # Fill out the attributes
        self.validate_init()

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
        self.dvh_metric_goals = {}
        for value in self.catheter_table_optim.plan.dvh_metric_goals.values():
            for dvh in value["dvh_metric_goals"]:
                if value["dvh_metric_goals"].get(dvh, None) is not None:
                    self.dvh_metric_goals[dvh] = value["dvh_metric_goals"][dvh]
        if (
            self.dvh_metric_goals is None
            or len(self.dvh_metric_goals) == 0
            or isinstance(self.dvh_metric_goals, list)):
            raise ValueError("The DVH metric goal dictionary is essential for \
multi-objective optimization. please provide it to the optimization object.") 
        for key, value in self.dvh_metric_goals.items():
            wrong_value = False
            if not isinstance(value, list):
                wrong_value = True
            elif len(value) != 2:
                wrong_value = True
            elif value[0] not in ["==", "<=", ">="]:
                wrong_value = True
            elif not isinstance(value[1], (int, float)):
                wrong_value = True
            if wrong_value:
                raise ValueError(f"The value of the DVH metric goal: {key} \
should be a list of string operation (one of ['==', '<=', '>=']) and float \
see `BrachyStructure.set_dvh_metric_goals()` for more details.")

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
    def evaluate(self, parameters: pd.DataFrame) -> pd.DataFrame:
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
        pass

    @abstractmethod
    def run_warmups(
        self,
        n_warmups: int,
        multi_proc: bool = True,
        ):
        r"""
        ### Purpose:
        - To run random sampling for `n_warmups` number of warmup trials.
        All the warmup trials will be randomly sampled from the parameter space and
        evaluated either sequentially or in parallel. The results will be stored in 
        `self.trial_data`. The tuner will be built after the warmup trials are completed.

        ### Inputs:
        - n_warmups: int := The number of warmup trials to run.
        
        ### Outputs:
        None := Fills out the following attributes:
        - `self.trial_data`: pd.DataFrame := A master dataframe containing the result of
        all the trials. The columns are parameter names from the keys of 
        `self.parameter_space` and the dvh metric names from the keys of 
        `self.dvh_metric_goals`.
        """
        pass

    @abstractmethod
    def run_trials(
        self,
        n_trials: int,
        ):
        r"""
        ### Purpose:
        - To run the multi-objective optimization for `n_trials` number of trials.
        The tuner will recommend the next batch of parameters to be evaluated.
        The evaluation will be done by the `evaluate()` method. The results will be
        stored in `self.trial_data`.
        ### Inputs:
        - n_trials: int := The number of trials to run.
        
        ### Outputs:
        None := Fills out the following attributes:
        - `self.trial_data`: pd.DataFrame := A master dataframe containing the result of
        all the trials. The columns are parameter names from the keys of 
        `self.parameter_space` and the dvh metric names from the keys of 
        `self.dvh_metric_goals`. 
        """
        pass

class MOO_Optuna(MOO):
    def __init__(
        self,
        catheter_table_optim: CatheterTableOptim_Gurobi,
        parameter_space: Dict[str, np.typing.ArrayLike],
        ):
        super().__init__(
            catheter_table_optim=catheter_table_optim,
            parameter_space=parameter_space,
            )
        self._parameter_distributions = None
        self._parameter_space_to_distributions()

    def _parameter_space_to_distributions(self):
        r"""
        ### Purpose:
        - To convert the parameter space to distributions that can be used by Optuna.
        The distributions are stored in `self._parameter_distributions` as a dictionary
        mapping the parameter names to their distributions.
        """
        self._parameter_distributions = {}
        for key, value in self.parameter_space.items():
            if len(value) != 2:
                raise ValueError(f"The parameter space for {key} should be a list of two values [min, max]")
            self._parameter_distributions[key] = optuna.distributions.FloatDistribution(
                low=value[0], high=value[1])

    def set_tuner(self):
        directions = self._get_directions_from_dvh_metric_goals()
        sampler = optuna.samplers.NSGAIISampler() # you can control the sampler here.
        study = optuna.create_study(
            directions = list(directions.values()),
            study_name = f"MOO_{self.catheter_table_optim.plan.phantom.pth_image.stem}",
            sampler = sampler,
        )
        self.tuner = study

    def run_warmups(
        self,
        n_warmups: int,
        multi_proc: bool = True,
        ):
        r"""
        ### Purpose:
        - To run random sampling for `n_warmups` number of warmup trials.
        All the warmup trials will be randomly sampled from the parameter space and
        evaluated either sequentially or in parallel. The results will be stored in 
        `self.trial_data`. The tuner will be built after the warmup trials are completed.

        ### Inputs:
        - n_warmups: int := The number of warmup trials to run.
        
        ### Outputs:
        None := Fills out the following attributes:
        - `self.trial_data`: pd.DataFrame := A master dataframe containing the result of
        all the trials. The columns are parameter names from the keys of 
        `self.parameter_space` and the dvh metric names from the keys of 
        `self.dvh_metric_goals`.
        """
        original_sampler = self.tuner.sampler
        # replace the sample with random sampler for warmup trials
        random_sampler = optuna.samplers.RandomSampler(seed=1)
        self.tuner.sampler = random_sampler
        # TODO 2: check if trials.params are in the right format
        trials = [self.tuner.ask() for _ in range(n_warmups)]
        if multi_proc:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.evaluate, pd.DataFrame([trial.params])) 
                    for trial in trials
                ]
                for future in as_completed(futures):
                    trial = futures[future]
                    values = future.result()
                    # TODO 2: check if the values are in the right format
                    self.tuner.tell(trial, values)
        else:
            for trial in trials:
                values = self.evaluate(pd.DataFrame([trial.params]))
                self.tuner.tell(trial, values)
        # restore the original sampler
        self.tuner.sampler = original_sampler

    def _get_directions_from_dvh_metric_goals(self):
        r"""
        ### Purpose:
        - To get the directions of optimization for each DVH metric goal.
        The direction is either "minimize" or "maximize" depending on the 
        operation in the dvh_metric_goals. For example, if the operation is "<=",
        then the direction is "minimize". If the operation is ">=", then the direction
        is "maximize". If the operation is "==", then the direction is "minimize".
        ### Inputs:
        None := Expects self.dvh_metric_goals to be filled out.
        
        ### Outputs:
        directions: Dict[str, str] := A dictionary of directions for each DVH metric goal.
        The order of the directions corresponds to the order of the keys in 
        self.dvh_metric_goals.
        """
        directions = {}
        for key, value in self.dvh_metric_goals.items():
            if value[0] == "<=":
                directions[key] = "minimize"
            elif value[0] == ">=":
                directions[key] = "maximize"
            elif value[0] == "==":
                directions[key] = "minimize"
            else:
                raise ValueError(f"The operation: {value[0]} \
for DVH metric goal: {key} is not valid. Please use one of ['==', '<=', '>=']")
        return directions

    def evaluate(self, parameters: pd.DataFrame) -> pd.DataFrame:
        # TODO 1: Implement this function as a stand alone function for the MOO class.
        return NotImplementedError("The evaluation is not implemented yet. \
Please use the `MOO_Optuna` class to implement the evaluation.")

    def run_trials(self, n_trials: int):
        return NotImplementedError("The run_trials is not implemented yet. \
Please use the `MOO_Optuna` class to implement the run_trials.")