import numpy as np
from typing import Dict, Any, List, Literal, Optional
from abc import ABC, abstractmethod
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
    update_penalty_weights_and_voxel_goals,
)
from brachyutils.planning.optimization.optim_gurobi import (
    get_optimized_dwelltimes_from_model,
)
import optuna
import optunahub
from concurrent.futures import ThreadPoolExecutor, as_completed

def _dominates_min(a: np.ndarray, b: np.ndarray) -> bool:
    r"""
    ### Purpose:
    - Pareto dominance check in a minimization convention: `a` dominates `b` if `a`
    is no worse than `b` in every dimension and strictly better in at least one.
    """
    return bool(np.all(a <= b) and np.any(a < b))


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
) -> pd.DataFrame:
    r"""
    ### Purpose:
    - Evaluates the parameters (i.e. penalty weights and target dose) and
    returns the observed dvh metrics corresponding to those parameters.

    ### Inputs:
    - parameters: pd.DataFrame := A dataframe with the parameters to be evaluated.
    The columns are the parameter names and the rows are the different parameter sets to be evaluated.
    - optim_obj: CatheterTableOptim_Gurobi := The optimization object that will be used to evaluate the parameters.
    - max_workers: int := The maximum number of workers to use for parallel evaluation.
    If max_workers is 1, the evaluation will be done sequentially.

    ### Outputs:
    - dvh_metrics_data: pd.DataFrame := A dataframe with the observed dvh metrics
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
                dvh_metrics_list.append(dvh_metrics)
    else:
        dvh_metrics_list = []
        for model in model_list:
            dwell_time_dict = get_optimized_dwelltimes_from_model(model)[0]
            optim_obj.plan.catheter_table.set_dwelltimes_by_names(
                dwell_time_dict)
            dvh_metrics = optim_obj.plan.get_dvh_metrics()
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
        """
        self.catheter_table_optim = catheter_table_optim
        self.parameter_space = parameter_space
        self.max_workers = max_workers
        # # Attributes to be filled out
        self.dvh_metric_goals: Dict[str, List] = None
        self.tuner: Any = None
        self.trial_data: pd.DataFrame = None
        # # Fill out the attributes
        self.validate_init()
        self.build_constraints_func()

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
                + list(self.dvh_metric_goals.keys())))

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

    def get_convergence_stats(
        self,
        reference_point: Optional[Dict[str, float]] = None,
        reference_margin: float = 0.1,
    ) -> Dict[str, Any]:
        r"""
        ### Purpose:
        - To report whether the trials recorded in `self.trial_data` are "getting better",
        using three complementary diagnostics computed purely from `self.trial_data` and
        `self.dvh_metric_goals`:

        1. Acceptance: whether each trial satisfies every DVH metric goal, plus the overall
        and running (cumulative) acceptance rate.
        2. Per-metric running best: the best value seen so far for each DVH metric
        individually (min for `<=` goals, max for `>=` goals), tracked trial-by-trial.
        3. Hypervolume: the dominated hypervolume of the feasible Pareto front, tracked
        trial-by-trial. This is the standard scalar convergence indicator for
        multi-objective optimization; it should trend upward (or plateau once converged)
        as better, more balanced trade-offs are discovered.

        ### Inputs:
        - reference_point: Optional[Dict[str, float]] := An explicit reference point for
        the hypervolume computation, mapping DVH metric name to a value that is *worse*
        than any trial should reasonably achieve (e.g. `{"D2cc(RECTUM)": 75}` for a
        `<=66` goal). Any metric not supplied here falls back to the automatic default
        below. If None entirely, all metrics use the automatic default.
        - reference_margin: float := Fractional margin used to build the automatic
        reference point when not supplied: `threshold * (1 + margin)` for `<=` goals
        and `threshold * (1 - margin)` for `>=` goals (or `threshold +/- margin` if the
        threshold is 0). Defaults to 0.1 (10%).

        ### Outputs:
        stats: Dict[str, Any] := A dictionary with:
        - "per_trial": pd.DataFrame := One row per trial (same order as `self.trial_data`),
        with columns "is_feasible", "cumulative_acceptance_rate", "hypervolume", and
        "running_best_{metric}" for every key in `self.dvh_metric_goals`.
        - "summary": Dict[str, Any] := {"n_trials", "n_feasible", "acceptance_rate",
        "final_hypervolume", "reference_point"}.
        """
        if self.trial_data is None or len(self.trial_data) == 0:
            raise ValueError("self.trial_data is empty. Run some trials before \
calling get_convergence_stats().")

        metrics = list(self.dvh_metric_goals.keys())
        data = self.trial_data.reset_index(drop=True)
        n = len(data)

        # # 1. Per-trial feasibility w.r.t. every DVH metric goal.
        is_feasible = np.ones(n, dtype=bool)
        for metric in metrics:
            operation, threshold = self.dvh_metric_goals[metric]
            observed = data[metric].to_numpy(dtype=float)
            if operation == "<=":
                satisfied = observed <= threshold
            else:  # ">="
                satisfied = observed >= threshold
            is_feasible &= satisfied
        cumulative_acceptance_rate = np.cumsum(is_feasible) / np.arange(1, n + 1)

        # # 2. Per-metric running best (independent of overall feasibility, tracks
        # # each metric's own trajectory across all trials).
        running_best = {}
        for metric in metrics:
            operation, _ = self.dvh_metric_goals[metric]
            observed = data[metric].to_numpy(dtype=float)
            if operation == "<=":
                running_best[f"running_best_{metric}"] = np.minimum.accumulate(observed)
            else:  # ">="
                running_best[f"running_best_{metric}"] = np.maximum.accumulate(observed)

        # # 3. Hypervolume of the feasible Pareto front, tracked trial-by-trial.
        signs = np.array([
            1.0 if self.dvh_metric_goals[metric][0] == "<=" else -1.0
            for metric in metrics
        ])
        loss_matrix = data[metrics].to_numpy(dtype=float) * signs

        resolved_reference = {}
        for metric in metrics:
            operation, threshold = self.dvh_metric_goals[metric]
            if reference_point is not None and metric in reference_point:
                resolved_reference[metric] = reference_point[metric]
            elif operation == "<=":
                resolved_reference[metric] = (
                    threshold * (1 + reference_margin) if threshold != 0
                    else threshold + reference_margin)
            else:  # ">="
                resolved_reference[metric] = (
                    threshold * (1 - reference_margin) if threshold != 0
                    else threshold - reference_margin)
        reference_loss = np.array([
            resolved_reference[metric] * sign
            for metric, sign in zip(metrics, signs)
        ])

        try:
            from optuna._hypervolume import compute_hypervolume
        except ImportError:
            compute_hypervolume = None

        pareto_front_loss: List[np.ndarray] = []
        current_hypervolume = 0.0
        hypervolume_history = []
        for i in range(n):
            if not is_feasible[i]:
                hypervolume_history.append(current_hypervolume)
                continue
            point = loss_matrix[i]
            if any(_dominates_min(f, point) for f in pareto_front_loss):
                hypervolume_history.append(current_hypervolume)
                continue
            pareto_front_loss = [
                f for f in pareto_front_loss if not _dominates_min(point, f)
            ] + [point]
            valid_points = np.array([
                f for f in pareto_front_loss if np.all(f <= reference_loss)
            ])
            if valid_points.size > 0 and compute_hypervolume is not None:
                current_hypervolume = compute_hypervolume(valid_points, reference_loss)
            hypervolume_history.append(current_hypervolume)

        per_trial = pd.DataFrame({
            "is_feasible": is_feasible,
            "cumulative_acceptance_rate": cumulative_acceptance_rate,
            "hypervolume": hypervolume_history,
            **running_best,
        })

        summary = {
            "n_trials": n,
            "n_feasible": int(is_feasible.sum()),
            "acceptance_rate": float(is_feasible.mean()),
            "final_hypervolume": hypervolume_history[-1] if hypervolume_history else 0.0,
            "reference_point": resolved_reference,
        }

        return {"per_trial": per_trial, "summary": summary}

class MOO_Optuna(MOO):
    # # Samplers that accept a `constraints_func` kwarg for constrained optimization.
    _constraint_capable_samplers = {
        "AutoSampler",
        "NSGAIISampler",
        "TPESampler",
        "GPSampler",
        "BoTorchSampler",
    }

    def __init__(
        self,
        catheter_table_optim: CatheterTableOptim_Gurobi,
        parameter_space: Dict[str, np.typing.ArrayLike],
        max_workers: int = 16,
        sampler_name_id: Literal["AutoSampler",
            "NSGAIISampler", "TPESampler", "GPSampler",
            "NSGAIIISampler", "BoTorchSampler"] = "AutoSampler",
    ):
        super().__init__(
            catheter_table_optim=catheter_table_optim,
            parameter_space=parameter_space,
            max_workers=max_workers,
        )
        self.sampler_name_id = sampler_name_id
        self._parameter_distributions = None
        self.constraints_func = None
        self._parameter_space_to_distributions()
        self.build_constraints_func()
        self.set_tuner()

    def build_constraints_func(self):
        r"""
        ### Purpose:
        - To build, once, the constraint function that Optuna's samplers will use to
        determine trial feasibility. Every entry in `self.dvh_metric_goals` is treated
        as BOTH a multi-objective direction (see `_get_directions_from_dvh_metric_goals`)
        AND a hard clinical constraint: the plan must, ideally, satisfy the goal, and
        Optuna's constrained samplers will steer the search towards the feasible region.

        Only `"<="` and `">="` operators are supported (see `validate_init`).
        Optuna's convention is: a constraint value <= 0 means feasible, > 0 means violated.
        - `<=` goal (e.g. D2cc(RECTUM) <= 66): constraint = observed - threshold
        - `>=` goal (e.g. D95%(CTV) >= 95): constraint = threshold - observed

        The resulting function expects the observed DVH metrics for a trial to be stored
        under `trial.user_attrs["dvh_metrics"]` (a dict keyed by DVH metric name). This is
        populated inside `objectives()` right before the trial is told to the study, so the
        constraints function only ever reads already-computed values, it never re-evaluates
        anything.

        ### Inputs:
        None := Expects `self.dvh_metric_goals` to already be filled out by `validate_init`.

        ### Outputs:
        None := Fills out the following attribute:
        - `self.constraints_func`: Callable[[optuna.trial.FrozenTrial], List[float]]
        """
        constraint_specs = [
            (key, operation, threshold)
            for key, (operation, threshold) in self.dvh_metric_goals.items()
        ]

        def constraints_func(trial: "optuna.trial.FrozenTrial") -> List[float]:
            dvh_metrics = trial.user_attrs["dvh_metrics"]
            constraint_values = []
            for key, operation, threshold in constraint_specs:
                observed = dvh_metrics[key]
                if operation == "<=":
                    constraint_values.append(observed - threshold)
                elif operation == ">=":
                    constraint_values.append(threshold - observed)
                else:
                    raise ValueError(f"Unsupported operator {operation} for \
constraint {key}. Only '<=' and '>=' are supported.")
            return constraint_values

        self.constraints_func = constraints_func

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
        r"""
        ### Purpose:
        - To build the tuner object that will recommend the next batch of parameter
        queries to be evaluated. The tuner is built using the Optuna library and the
        sampler specified in the constructor. The tuner is stored in `self.tuner`.

        Constraints (built once in `__init__` via `build_constraints_func`) are passed
        into the sampler here, for every sampler that supports Optuna's
        `constraints_func` convention. Samplers that don't support constraints (e.g.
        `NSGAIIISampler`, or a future sampler not in `_constraint_capable_samplers`)
        fall back to unconstrained multi-objective optimization with a warning, since
        passing an unsupported kwarg would raise a `TypeError`.

        ### Inputs:
        None := Expects the following to be filled already:
        - `self.catheter_table_optim`
        - `self.parameter_space`
        - `self.dvh_metric_goals`
        - `self.constraints_func`
        - `self.sampler_name_id`
        """
        directions = self._get_directions_from_dvh_metric_goals()
        use_constraints = self.sampler_name_id in self._constraint_capable_samplers

        if not use_constraints:
            print(f"Warning: sampler '{self.sampler_name_id}' does not support \
`constraints_func`. Falling back to unconstrained multi-objective optimization. \
The DVH metric goals will still be tracked as objectives, but infeasible trials \
will not be pruned/penalized by the sampler.")

        constraints_kwargs = {"constraints_func": self.constraints_func} if use_constraints else {}

        if self.sampler_name_id == "AutoSampler":
            auto_sampler_module = optunahub.load_module(
                package="samplers/auto_sampler")
            sampler_obj = auto_sampler_module.AutoSampler(**constraints_kwargs)
        elif self.sampler_name_id == "NSGAIISampler":
            sampler_obj = optuna.samplers.NSGAIISampler(**constraints_kwargs)
        elif self.sampler_name_id == "TPESampler":
            sampler_obj = optuna.samplers.TPESampler(**constraints_kwargs)
        elif self.sampler_name_id == "GPSampler":
            sampler_obj = optuna.samplers.GPSampler(**constraints_kwargs)
        elif self.sampler_name_id == "NSGAIIISampler":
            sampler_obj = optuna.samplers.NSGAIIISampler()
        elif self.sampler_name_id == "BoTorchSampler":
            sampler_obj = optuna.samplers.BoTorchSampler(**constraints_kwargs)
        else:
            raise ValueError(f"Unknown sampler_name_id: {self.sampler_name_id}")

        study = optuna.create_study(
            directions=list(directions.values()),
            study_name=f"MOO_{self.catheter_table_optim.plan.phantom.pth_image.stem}",
            sampler=sampler_obj,
        )
        self.tuner = study

    def run_warmups(self, batch_size: int):
        r"""
        ### Purpose:
        - To run random sampling for `batch_size` number of warmup trials.
        All the warmup trials will be randomly sampled from the parameter space and
        evaluated either sequentially or in parallel. The results will be stored in
        `self.trial_data`.

        ### Inputs:
        - batch_size: int := The number of warmup trials to run.

        ### Outputs:
        None := Fills out the following attributes:
        - `self.trial_data`: pd.DataFrame := A master dataframe containing the result of
        all the trials. The columns are parameter names from the keys of
        `self.parameter_space` and the dvh metric names from the keys of
        `self.dvh_metric_goals`.
        """
        original_sampler = self.tuner.sampler
        # replace the sampler with a random sampler for warmup trials
        random_sampler = optuna.samplers.RandomSampler(seed=1)
        self.tuner.sampler = random_sampler

        trials = [
            self.tuner.ask(self._parameter_distributions)
            for _ in range(batch_size)]

        objectives = self.objectives(trials)
        for trial, objective in zip(trials, objectives):
            self.tuner.tell(trial.number, objective)

        # restore the original sampler
        self.tuner.sampler = original_sampler

    def _get_directions_from_dvh_metric_goals(self):
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
            else:
                raise ValueError(f"The operation: {value[0]} \
for DVH metric goal: {key} is not valid. Please use one of ['<=', '>=']")
        return directions

    def objectives(self, trials: List[optuna.trial.Trial]) -> list:
        r"""
        ### Purpose:
        - To evaluate the objectives for each trial. The objectives are the DVH metrics
        corresponding to the parameters in the trial. The order of the objectives
        corresponds to the order of the keys in self.dvh_metric_goals.
        - All trials passed in are evaluated together in a single, batched call to
        `evaluate_parameters`, which parallelizes the underlying Gurobi solves across
        `self.max_workers` threads. This is what powers batch mode: `run_trials` decides
        how many trials to ask for at once, and this method evaluates them all together.
        - As a side effect, this also stores the observed DVH metrics on each trial via
        `trial.set_user_attr("dvh_metrics", ...)`, so that `self.constraints_func` (built
        once in `__init__`) can read them back after `tell()` without re-evaluating anything.

        ### Inputs:
        - trials: List[optuna.trial.Trial] := A list of trials to be evaluated.

        ### Outputs:
        objectives: list := A list of lists of objectives for each trial. The order
        of the objectives corresponds to the order of the keys in self.dvh_metric_goals.
        """
        trial_params = pd.DataFrame([trial.params for trial in trials])
        dvh_metrics_data = evaluate_parameters(
            trial_params,
            self.catheter_table_optim,
            max_workers=self.max_workers,
        )

        self.trial_data = pd.concat([
            self.trial_data,
            pd.concat([trial_params, dvh_metrics_data], axis=1)
        ], axis=0)
        self.trial_data.reset_index(drop=True, inplace=True)

        # # Attach the observed DVH metrics to each trial for the constraints_func to use.
        for trial, (_, row) in zip(trials, dvh_metrics_data.iterrows()):
            trial.set_user_attr("dvh_metrics", row.to_dict())

        objectives = dvh_metrics_data[list(self.dvh_metric_goals.keys())].values.tolist()
        return objectives

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
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        for _ in range(n_trials):
            trials = [
                self.tuner.ask(self._parameter_distributions)
                for _ in range(batch_size)]
            objectives = self.objectives(trials)
            for trial, objective in zip(trials, objectives):
                self.tuner.tell(trial.number, objective)
