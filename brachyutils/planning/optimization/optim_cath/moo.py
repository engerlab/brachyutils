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
    anchor_dvh_metric: Dict[str, float] = None
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
    - `anchor_dvh_metric` := The dwell times are scaled to match the desired value for 
    the dvh metric provided. Only one anchor can be provided. 
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
                if anchor_dvh_metric is not None:
                    dvh_metrics = _anchor_plan_to_dvh_metric(
                        plan=optim_obj.plan,
                        anchor_dvh_metric=anchor_dvh_metric,
                        observed_dvh_metrics=dvh_metrics,)
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
            if anchor_dvh_metric is not None:
                dvh_metrics = _anchor_plan_to_dvh_metric(
                    plan=optim_obj.plan,
                    anchor_dvh_metric=anchor_dvh_metric,
                    observed_dvh_metrics=dvh_metrics,)
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
        scale_dwelltimes_by_metric: str = None
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
        - `normalize` := If true dvh_metric_goals would be normalized from 100% to 1.
        Be sure that the DVH metrics are in percentage form (defualt is percentage).

        """
        self.catheter_table_optim = catheter_table_optim
        self.parameter_space = parameter_space
        self.max_workers = max_workers
        self.normalize = normalize
        self.scale_dwelltimes_by_metric = scale_dwelltimes_by_metric
        # # Attributes to be filled out
        self.dvh_metric_goals: Dict[str, List] = None
        self.tuner: Any = None
        self.trial_data: pd.DataFrame = None
        self.directions = None
        # # Fill out the attributes
        self.validate_init()
        self.set_directions_from_dvh_metric_goals()

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

    def objectives(
        self,
        parameters:pd.DataFrame,
        sampler_name_id: str) -> pd.DataFrame:
        r"""
        ### Purpose:
        - To evaluate the objectives for the parameters generated in each trial.
        The objectives are the DVH metrics in the order set by the keys in self.dvh_metric_goals.
        - All parameters passed here are evaluated together in a single, batched call to
        `evaluate_parameters()`, which parallelizes the underlying Gurobi solves across
        `self.max_workers` threads.
        - Also calculates the hyper-volume and whetheter each parameter lead to acceptable
        dvh metrics or not.
        - Lastly, `self.trial_data` is updated with all the information:
            - parameter values
            - dvh metrics observed
            - hyper volume
            - acceptability

        ### Inputs:
        - parameters := A dataframe of parameters to be evaluated. The columns are the names
        of the parameters while the rows are parameter values for each trial.
        - sampler_name_id: str := The name of the sampler being used. This is used to
        keep track of which sampler was used for each trial in the `self.trial_data` dataframe.

        ### Outputs:
        - A dataframe with the dvh metrics specified by the keys in self.dvh_metric_goals.
        """
        if self.scale_dwelltimes_by_metric is not None:
            anchor_dvh_metric = {
                self.scale_dwelltimes_by_metric: self.dvh_metric_goals[
                    self.scale_dwelltimes_by_metric][1]
            }
        else:
            anchor_dvh_metric=None
        dvh_metrics_data = evaluate_parameters(
            parameters,
            self.catheter_table_optim,
            max_workers=self.max_workers,
            anchor_dvh_metric=anchor_dvh_metric,
        )
        acceptable_trials = are_acceptable(dvh_metrics_data, self.dvh_metric_goals)
        hv_trials = get_hyper_volume(dvh_metrics_data, self.dvh_metric_goals)
        sampler_df = pd.Series(
            [sampler_name_id for _ in range(len(parameters))],
            name="sampler_name_id").to_frame()
        self.trial_data = pd.concat([
            self.trial_data,
            pd.concat([
                parameters.reset_index(drop=True), dvh_metrics_data,
                acceptable_trials, hv_trials, sampler_df], axis=1)
        ], axis=0)
        self.trial_data.reset_index(drop=True, inplace=True)

        return dvh_metrics_data[list(self.dvh_metric_goals.keys())]

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
        max_workers: int = 16,
    ):
        r"""
        ### Purpose:
        - To run random sampling for `n_warmups` number of warmup trials.
        All the warmup trials will be randomly sampled from the parameter space and
        evaluated either sequentially or in parallel.

        ### Inputs:
        - n_warmups: int := The number of warmup trials to run.
        - max_workers: int := The number of threads to be used for parallelized
        evaluation of the penalty weights.

        ### Outputs:
        None := Fills out the following attributes:
        - `self.trial_data`: pd.DataFrame := A master dataframe containing the result of
        all the trials. The columns are parameter names from the keys of
        `self.parameter_space` and the dvh metric names from the keys of
        `self.dvh_metric_goals`.
        """
        pass

    @abstractmethod
    def run_trials(self, n_trials: int, batch_size: int = 1):
        r"""
        ### Purpose:
        - To run the multi-objective optimization for `n_trials` outer iterations.
        On each iteration, `batch_size` trials are asked for at once and evaluated
        together in a single batched call to `objectives()` (which itself parallelizes
        the Gurobi solves via `self.max_workers`), then each trial is told back to the
        study individually. The total number of trials evaluated across the whole run
        is `n_trials * batch_size`.
        - When `batch_size == 1` this reduces to the original sequential ask/evaluate/tell
        loop.

        ### Inputs:
        - n_trials: int := The number of outer iterations (batches) to run.
        - batch_size: int := The number of trials to ask for and evaluate together on
        each iteration. Total trials evaluated = n_trials * batch_size.

        ### Outputs:
        None := Fills out the following attributes:
        - `self.trial_data`: pd.DataFrame := A master dataframe containing the result of
        all the trials. The columns are parameter names from the keys of
        `self.parameter_space` and the dvh metric names from the keys of
        `self.dvh_metric_goals`.
        """
        pass

    def set_directions_from_dvh_metric_goals(self):
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

    @abstractmethod
    def get_parameters_from_trials(
        self,
        trials: List[Any]) -> pd.DataFrame:
        r"""
        ### Purpose:
        - To extract the plan optimization parameters (penalty weights, target dose etc.) from the 
        trial objects generated by the Tuner.
        The trial objects are specific to the underlying MOO package used.

        ### Inputs:
        trials := A list of trial objects generated by self.tuner

        ### Outputs:
        - parameters_df := a dataframe with the name of the parameters as columns and
        their values in the rows.
        """
        pass
    
    @abstractmethod
    def attach_objectives_to_trials(
        self,
        trials: List[Any],
        observed_objectives:pd.DataFrame
        ):
        r"""
        ### Purpose:
        - To update the trial objects and self.tuner with the result of parameter evaluations, i.e.
        observed dvh metrics
        """
        pass

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

def _anchor_plan_to_dvh_metric(
    plan,
    anchor_dvh_metric: dict,
    observed_dvh_metrics: dict,
    ):
    r"""
    ### Purpose:
    - plan:BrachyPlan := the whose dwell times are to be scaled to match the anchor DVH metric.
    
    ### Inputs:
    - `anchor_dvh_metric`: dict := a dictionary containing the name of the DVH metric to be used
    for anchoring as well its desired value.
    - `observed_dvh_metrics`: dict := The dictinoray of the observed dvh metrics from the plan.
    
    ### Outputs:
    - scaled_dvh_metrics: dict := The dictionary of the dvh metrics from the scaled dvh dwell times.
    """
    if len(anchor_dvh_metric) != 1:
        raise ValueError("Only one dvh metric can be used for anchoring")

    dvh_name, dvh_value = list(anchor_dvh_metric.items())[0]
    observed_dvh_value = observed_dvh_metrics[dvh_name]
    scaling_factor = dvh_value / observed_dvh_value

    plan.catheter_table.scale_dwelltimes_by(scaling_factor)
    return plan.get_dvh_metrics()