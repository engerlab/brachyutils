import numpy as np
from typing import Dict, List, Literal
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
)
import optuna
import optunahub

from brachyutils.planning.optimization.optim_cath.moo import (
    MOO, evaluate_parameters, are_acceptable, get_hyper_volume)

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
        use_constraints: bool = False,
    ):
        super().__init__(
            catheter_table_optim=catheter_table_optim,
            parameter_space=parameter_space,
            max_workers=max_workers,
        )
        self.sampler_name_id = sampler_name_id
        self.use_constraints = use_constraints
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
        as BOTH a multi-objective direction (see `get_directions_from_dvh_metric_goals`)
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
        use_constraints = self.sampler_name_id in self._constraint_capable_samplers
        use_constraints = self.use_constraints and use_constraints
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
            directions=list(self.directions.values()),
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

        objectives = self.objectives(trials, "RandomSampler")
        for trial, objective in zip(trials, objectives):
            self.tuner.tell(trial.number, objective)

        # restore the original sampler
        self.tuner.sampler = original_sampler

    def objectives(
        self,
        trials: List[optuna.trial.Trial],
        sampler_name_id: str) -> list:
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
        - sampler_name_id: str := The name of the sampler being used. This is used to
        keep track of which sampler was used for each trial in the `self.trial_data` dataframe.

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
            objectives = self.objectives(trials, self.sampler_name_id)
            for trial, objective in zip(trials, objectives):
                self.tuner.tell(trial.number, objective)
