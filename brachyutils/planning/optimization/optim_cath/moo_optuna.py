import numpy as np
from typing import Dict, List, Literal
import pandas as pd
from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
)
import optuna
import optunahub

from brachyutils.planning.optimization.optim_cath.moo import MOO

class MOO_Optuna(MOO):
    # # Samplers that accept a `_optuna_constraints` kwarg for constrained optimization.
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
        anchor_dvh_metric:str = None,
        sampler_name_id: Literal["AutoSampler",
            "NSGAIISampler", "TPESampler", "GPSampler",
            "NSGAIIISampler", "BoTorchSampler"] = "AutoSampler",
        normalize:bool = False,
        slack_space_shrinkage: float = None,
        slack_objective_threshold: float | Dict[str, float] = 0,
        use_constraints: bool = True,
        ):
        super().__init__(
            catheter_table_optim=catheter_table_optim,
            parameter_space=parameter_space,
            max_workers=max_workers,
            normalize=normalize,
            anchor_dvh_metric=anchor_dvh_metric,
            slack_space_shrinkage=slack_space_shrinkage,
            slack_objective_threshold=slack_objective_threshold,
        )
        self.sampler_name_id = sampler_name_id
        self.use_constraints = use_constraints
        self._optuna_parameters = None
        self._optuna_constraints = None
        self._optuna_objectives = None
        self.set_tuner()

    def _build_optuna_constraints(self):
        r"""
        ### Purpose:
        - To build, once, the constraint function that Optuna's samplers will use to
        determine trial feasibility. Every entry in `self.dvh_metric_goals` is treated
        as BOTH a multi-objective direction (see `set_directions_from_dvh_metric_goals`)
        AND a clinical constraint: the plan must, ideally, satisfy the goal, and
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
        - `self._optuna_constraints`: Callable[[optuna.trial.FrozenTrial], List[float]]
        """
        constraint_specs = [
            (key, operation, threshold)
            for key, (operation, threshold) in self.dvh_metric_goals.items()
        ]

        def _optuna_constraints(trial: "optuna.trial.FrozenTrial") -> List[float]:
            dvh_metrics = trial.user_attrs["dvh_metrics"]
            constraint_values = []
            for key, operation, threshold in constraint_specs:
                observed = dvh_metrics[key]
                if operation == "<=":
                    constraint_values.append(observed - threshold*(
                        1 + self.slack_objective_threshold.get(key, 0)
                    ))
                elif operation == ">=":
                    constraint_values.append(threshold*(
                        1 - self.slack_objective_threshold.get(key, 0)
                    ) - observed)
                else:
                    raise ValueError(f"Unsupported operator {operation} for \
constraint {key}. Only '<=' and '>=' are supported.")
            return constraint_values

        self._optuna_constraints = _optuna_constraints

    def _parameter_space_to_optuna(self):
        r"""
        ### Purpose:
        - To convert the parameter space to distributions that can be used by Optuna.
        The distributions are stored in `self._optuna_parameters` as a dictionary
        mapping the parameter names to their distributions.
        """
        self._optuna_parameters = {}
        for key, value in self.parameter_space.items():
            if len(value) != 2:
                raise ValueError(f"The parameter space for {key} should be a list of two values [min, max]")
            self._optuna_parameters[key] = optuna.distributions.FloatDistribution(
                low=value[0], high=value[1])

    def set_tuner(self):
        r"""
        ### Purpose:
        - To build the tuner object that will recommend the next batch of parameter
        queries to be evaluated. The tuner is built using the Optuna library and the
        sampler specified in the constructor. The tuner is stored in `self.tuner`.

        Constraints (built once in `__init__` via `build__optuna_constraints`) are passed
        into the sampler here, for every sampler that supports Optuna's
        `_optuna_constraints` convention. Samplers that don't support constraints (e.g.
        `NSGAIIISampler`, or a future sampler not in `_constraint_capable_samplers`)
        fall back to unconstrained multi-objective optimization with a warning, since
        passing an unsupported kwarg would raise a `TypeError`.

        ### Inputs:
        None := Expects the following to be filled already:
        - `self.catheter_table_optim`
        - `self.parameter_space`
        - `self.dvh_metric_goals`
        - `self._optuna_constraints`
        - `self.sampler_name_id`
        """
        self._parameter_space_to_optuna()
        self._build_optuna_constraints()
        use_constraints = self.sampler_name_id in self._constraint_capable_samplers
        use_constraints = self.use_constraints and use_constraints
        if not use_constraints:
            print(f"Warning: sampler '{self.sampler_name_id}' does not support \
`_optuna_constraints`. Falling back to unconstrained multi-objective optimization. \
The DVH metric goals will still be tracked as objectives, but infeasible trials \
will not be pruned/penalized by the sampler.")

        constraints_kwargs = {
            "_optuna_constraints": self._optuna_constraints} if use_constraints else {}
        # TODO 1: test this
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

    def run_random_sampling(
        self,
        n_samples: int,
        use_for_warmup: bool = False
        ):
        r"""
        ### Purpose:
        - To run random sampling for `n_samples` number of samples.
        All the samples will be randomly sampled from the parameter space and
        evaluated either sequentially or in parallel. The results will be stored in
        `self.trial_data`.

        ### Inputs:
        - n_samples: int := The number of samples to generate and evaluate.
        - use_for_warmup: bool := If True, the random samples will be used to warm up
        the tuner. If False, the random samples will just be stored to self.trial_data.

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
            self.tuner.ask(self._optuna_parameters)
            for _ in range(n_samples)]
        trial_params = self.get_parameters_from_trials(trials=trials)
        observed_objectives = self.objectives(trial_params, "RandomSampler")
        if use_for_warmup:
            self.attach_objectives_to_trials(
                trials=trials,
                observed_objectives=observed_objectives)
        else:
            self.set_tuner()
        # restore the original sampler
        self.tuner.sampler = original_sampler

    def get_parameters_from_trials(
        self,
        trials: List[optuna.trial.Trial],
        ) -> pd.DataFrame:
        return pd.DataFrame([trial.params for trial in trials])

    def attach_objectives_to_trials(
        self,
        trials: List[optuna.trial.Trial],
        observed_objectives: pd.DataFrame,
        ):
        r"""
        ### Purpose:
        - To store the observed DVH metrics on each trial via
        `trial.set_user_attr("dvh_metrics", ...)`, so that `self._optuna_constraints` (built
        once in `__init__`) can read them back after `tell()` without re-evaluating anything.
        """
        for trial, (_, row) in zip(trials, observed_objectives.iterrows()):
            trial.set_user_attr("dvh_metrics", row.to_dict())
            # TODO 1: test this
            self.tuner.tell(trial, row.tolist())
        # for trial, objective in zip(trials, observed_objectives.values.tolist()):
        #     self.tuner.tell(trial.number, objective)

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
                self.tuner.ask(self._optuna_parameters)
                for _ in range(batch_size)]
            trial_params = self.get_parameters_from_trials(trials=trials)
            observed_objectives = self.objectives(trial_params, self.sampler_name_id)
            self.attach_objectives_to_trials(
                trials=trials,
                observed_objectives=observed_objectives)