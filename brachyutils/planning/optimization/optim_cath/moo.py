import numpy as np
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
    update_penalty_weights_and_voxel_goals,
)
from brachyutils.planning.optimization.optim_gurobi import (
    get_optimized_dwelltimes_from_model,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from botorch.utils.multi_objective import Hypervolume
import torch

def _update_optimization_configs_with_parameters(
    parameters: pd.DataFrame,
    optimization_configs: List[Any],
) -> None:
    r"""
    ### Purpose:
    - To update the optimization config of each structure with the parameters from the dataframe.
    """
    if parameters.shape[0] != 1:
        raise ValueError("The parameters dataframe should have only one row.")
    for key, value in parameters.iloc[0].items():
        structure_name = key.split("(")[-1].split(")")[0]
        parameter_name = key.split("(")[0]
        for optim_config in optimization_configs:
            if optim_config.structure_name == structure_name:
                setattr(optim_config, parameter_name, value)

def evaluate_parameters(
    parameters: pd.DataFrame,
    optim_obj: CatheterTableOptim_Gurobi,
    max_workers: int = 16,
    normalize: bool = False,
) -> pd.DataFrame:
    r"""
    ### Purpose:
    - Evaluates the parameters (i.e. penalty weights and target dose) and
    returns the observed dvh metrics corresponding to those parameters.

    ### Inputs:
    - `parameters`: pd.DataFrame := A dataframe with the parameters to be evaluated.
    The columns are the parameter names and the rows are the different parameter sets to be evaluated.
    - `optim_obj`: CatheterTableOptim_Gurobi := The optimization object that will be used to evaluate the parameters.
    - `max_workers`: int := The maximum number of workers to use for parallel evaluation.
    If max_workers is 1, the evaluation will be done sequentially.
    - `normalize`: If True, the dvh metric values are devided by 100.

    ### Outputs:
    - `dvh_metrics_data`: pd.DataFrame := A dataframe with the observed dvh metrics
    corresponding to the evaluated parameters. The columns are the dvh metric names and
    the rows are the different parameter sets that were evaluated.
    """
    model_list = []
    optimization_configs = list(optim_obj.plan.optimization_config_dict.values())
    for row in range(parameters.shape[0]):
        model = optim_obj.model.copy()
        _update_optimization_configs_with_parameters(
            parameters.iloc[row:row + 1],
            optimization_configs)
        update_penalty_weights_and_voxel_goals(
            model,
            optimization_configs)
        model_list.append(model)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(get_optimized_dwelltimes_from_model, model)
                for model in model_list]
            dvh_metrics_list = []
            for future in as_completed(futures):
                dwell_time_dict = future.result()[0]
                optim_obj.plan.catheter_table.set_dwelltimes_by_names(
                    dwell_time_dict)
                dvh_metrics = optim_obj.plan.get_dvh_metrics()
                if normalize:
                    for key in dvh_metrics:
                        dvh_metrics[key] = dvh_metrics[key]/100
                dvh_metrics_list.append(dvh_metrics)
    else:
        dvh_metrics_list = []
        for model in model_list:
            dwell_time_dict = get_optimized_dwelltimes_from_model(model)[0]
            optim_obj.plan.catheter_table.set_dwelltimes_by_names(
                dwell_time_dict)
            dvh_metrics = optim_obj.plan.get_dvh_metrics()
            if normalize:
                for key in dvh_metrics:
                    dvh_metrics[key] = dvh_metrics[key]/100
            dvh_metrics_list.append(dvh_metrics)

    return pd.DataFrame(dvh_metrics_list)

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
        max_workers: int = 16,
        normalize: bool = False,
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
            penalty_weight_linear(CTV) : [1, 500]
        - `normalize` := If true dvh_metric_goals would be normalized.
        Be sure that the DVH metrics are in percentage form (defualt is percentage).

        """
        self.catheter_table_optim = catheter_table_optim
        self.parameter_space = parameter_space
        self.max_workers = max_workers
        self.normalize = normalize
        # # Attributes to be filled out
        self.dvh_metric_goals: Dict[str, List] = None
        self.tuner: Any = None
        self.trial_data: pd.DataFrame = None
        self.directions = None
        # # Fill out the attributes
        self.validate_init()
        self.build_constraints_func()
        self.get_directions_from_dvh_metric_goals()

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
        to their clinically desired values. Metrics with a `None` goal (i.e. not
        clinically constrained/tracked) are skipped entirely and never appear here.
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
                    if self.normalize:
                        self.dvh_metric_goals[dvh] = [
                            self.dvh_metric_goals[dvh][0],
                            self.dvh_metric_goals[dvh][1]/100,
                        ]

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
            elif value[0] not in ["<=", ">="]:
                wrong_value = True
            elif not isinstance(value[1], (int, float)):
                wrong_value = True
            if wrong_value:
                raise ValueError(f"The value of the DVH metric goal: {key} \
should be a list of string operation (one of ['<=', '>=']) and float \
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

        # # Now build the columns of the master trial dataframe
        self.trial_data = pd.DataFrame(
            columns=(
                list(self.parameter_space.keys())
                + list(self.dvh_metric_goals.keys())
                + ["sampler_name_id", "acceptable", "hypervolume"]))

    @abstractmethod
    def objectives(self, parameters: pd.DataFrame) -> pd.DataFrame:
        r"""
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

    def get_directions_from_dvh_metric_goals(self):
        r"""
        ### Purpose:
        - To get the directions of optimization for each DVH metric goal.
        The direction is either "minimize" or "maximize" depending on the
        operation in the dvh_metric_goals. For example, if the operation is "<=",
        then the direction is "minimize". If the operation is ">=", then the direction
        is "maximize".
        ### Inputs:
        None := Expects self.dvh_metric_goals to be filled out.

        ### Outputs:
        -None:= sets self.directions: Dict[str, str] := A dictionary of directions for each DVH metric goal.
        The order of the directions corresponds to the order of the keys in
        self.dvh_metric_goals.
        """
        directions = {}
        for key, value in self.dvh_metric_goals.items():
            if value[0] == "<=":
                directions[key] = "minimize"
            elif value[0] == ">=":
                directions[key] = "maximize"
            else:
                raise ValueError(f"The operation: {value[0]} \
for DVH metric goal: {key} is not valid. Please use one of ['<=', '>=']")
        self.directions = directions

def are_acceptable(
    dvh_metrics: pd.DataFrame,
    dvh_metric_goals: Dict
    ) -> pd.DataFrame:
    r"""
    ### Purpose:
    - To assess if the provided dvh metrics are acceptable according to
    the dvh metric goals

    ### Inputs:
    - dvh_metrics := DVH metric names and their observed values
    - dvh_metric_goals := The operation and goal for each DVH metrics
    Ex. D95%(CTV): [">=", 100]    
    
    ### Outputs:
    - results_df: pd.DataFrame := a data frame containing boolean values
    for each row of the dvh_metrics. 
    """
    results_df = pd.DataFrame(columns=["acceptable"])
    for i, row in dvh_metrics.iterrows():    
        acceptable = True
        for key, value in dvh_metric_goals.items():
            observed_value = row.get(key, None)
            if observed_value is None:
                acceptable = False
            else:
                if value[0] == "<=":
                    if not observed_value <= value[1]:
                        acceptable = False
                        break
                if value[0] == ">=":
                    if not observed_value >= value[1]:
                        acceptable = False
                        break
        results_df.loc[i] = acceptable
    return results_df

def get_hyper_volume(
    dvh_metrics: pd.DataFrame, 
    dvh_metric_goals: dict[str, list],
    return_series: bool = True
):
    """
    ### Purpose:
    - Computes hypervolume for valid points and a negative penalty score
    for points violating at least one DVH requirement.

    ### Inputs
        - dvh_metrics: pd.DataFrame with candidate solutions as rows.
        - dvh_metric_goals: dict mapping metric -> [operator ('>=' or '<='), threshold].
        - return_series: If True, returns a pd.Series with scores for each individual row.
        If False, returns a single float (joint HV of valid set, 
        or worst negative violation if no point is valid).
    """
    ordered_keys = list(dvh_metric_goals.keys())
    
    # 1. Parse operators and reference points
    signs = []
    ref_values = []
    raw_thresholds = []
    for key in ordered_keys:
        op, threshold = dvh_metric_goals[key]
        raw_thresholds.append(float(threshold))
        if op == ">=":
            signs.append(1.0)
            ref_values.append(float(threshold))
        elif op == "<=":
            signs.append(-1.0)
            ref_values.append(-float(threshold))
        else:
            raise ValueError(f"Unsupported operation '{op}' for '{key}'.")

    sign_tensor = torch.tensor(signs, dtype=torch.double)
    ref_point = torch.tensor(ref_values, dtype=torch.double)
    scale = torch.tensor([max(abs(t), 1.0) for t in raw_thresholds], dtype=torch.double)

    # 2. Align to BoTorch maximization format (Y >= ref_point is feasible)
    metric_matrix = dvh_metrics[ordered_keys].to_numpy(dtype=np.float64)
    Y = torch.tensor(metric_matrix, dtype=torch.double) * sign_tensor

    # 3. Evaluate each solution row-by-row
    hv_calculator = Hypervolume(ref_point=ref_point)
    scores = []

    for i in range(Y.shape[0]):
        y_i = Y[i]
        diff = y_i - ref_point  # diff >= 0 means goal is satisfied
        
        if (diff >= 0).all():
            # Solution meets all goals: compute positive individual hypervolume
            val = float(hv_calculator.compute(y_i.unsqueeze(0)))
            scores.append(val)
        else:
            # Solution violates at least one goal: compute negative relative violation
            violations = torch.clamp(-diff, min=0.0)
            rel_violation = (violations / scale).sum().item()
            scores.append(-rel_violation)

    score_series = pd.Series(scores, index=dvh_metrics.index, name="hypervolume")

    if return_series:
        return score_series.to_frame()

    # If aggregated: return joint hypervolume of valid Pareto front,
    # or the best (least negative) violation score if none are valid
    valid_mask = Y >= ref_point
    all_valid_idx = valid_mask.all(dim=-1)
    
    if all_valid_idx.any():
        return float(hv_calculator.compute(Y[all_valid_idx]))
    else:
        return float(max(scores))