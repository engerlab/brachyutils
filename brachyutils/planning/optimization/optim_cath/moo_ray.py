import ray
from ray import tune
from ray.tune import TuneConfig, RunConfig, Tuner
import numpy as np
from typing import Dict, Any

from brachyutils.planning.optimization.optim_cath.dosimetric_gurobi import (
    CatheterTableOptim_Gurobi,
    update_penalty_weights_and_voxel_goals,
    )

from brachyutils.planning.optimization.optim_gurobi import (
    get_optimized_dwelltimes_from_model
)
class MOO_Ray():
    def __init__(self, catheter_table_optim:CatheterTableOptim_Gurobi):
        ### TODO: Complete it later
        self.param_space:dict = None
        self.algorithm:str = None
        self.trainable = tune.with_resources(
            {}
        )

def dvh_eval(
    parameters: np.ndarray,
    model: Any,
    plan:Any) -> Dict[str, float]:
    r"""
    ### Purpose:
    - Given a set of parameters (penalty weight and target doses),
    optimize the model, update the 
    """
    pass

def compute_objectives(
    dvh_dict: Dict[str, float]) -> Dict[str, float]:
    r"""
    measures how good are a set of DVH metrics. Ray expects all the objectives
    to be minimized. so for some DVH metrics that we need maximizing, we need to
    megate them.
    """
    to_maximize = {}
    to_minimize = {}
    obj = {
        key: -1 * val for key, val in to_maximize.items()
    } | to_minimize
    return obj

def run_trial(w: np.ndarray) -> Dict[str, float]:
    r"""
    evaluates the weights and calculates an objective based on them.
    """
    dvh = dvh_eval(w)
    objectives = compute_objectives(dvh)
    metrics = {
        **objectives,
        **dvh,
    }
    return metrics

def train_fn(config:dict):
    r"""
    runs a trial and reports the outcome to Ray
    """
    
    # 1. Reconstruct w from config
    # Suppose config has keys like "w_0", "w_1", ... or structured as needed.
    # Example: config = {"w_0": ..., "w_1": ..., ...}
    w_keys = [k for k in config if k.startswith("w_")]
    w_keys.sort()  # ensure consistent ordering
    w = np.array([config[k] for k in w_keys])

    # 2. Run black-box evaluation
    metrics = run_trial(w)

    # 3. Report all metrics to Ray Tune
    tune.report(**metrics)

num_weights = 5
# figure out what type of distribution the weights should be!
search_space = {
    f"w_{i}": tune.loguniform(0.1, 10.0)
    for i in range(num_weights)
}

ray.init()
trainable = tune.with_resources(
    trainable=train_fn, resources= {"cpu": 12, "gpu": 1})

run_config = RunConfig(name="penalty_weight_tuning")
tune_config = TuneConfig(num_samples=10, search_alg=None)

tuner = Tuner(
    trainable=trainable,
    param_space=search_space,
    tune_config=tune_config,
    run_config=run_config,
)

results = tuner.fit()

# # pareto_analysis: probably gotta use Ax, Nevergrad or Optuna for Pareto surface
# # analysis.

# # Here is for search algorithm

# AX
from ray.tune.search.ax import AXSearch
search_alg = AXSearch(
    metric=["obj_D95_HRV", "obj_V20_Lung"],  # or similar, depending on AX version
    mode=["min", "min"],
)

# Optuna
from ray.tune.search.optuna import OptunaSearch
search_alg = OptunaSearch(
    metric=["obj_D95_HRV", "obj_V20_Lung"],
    mode=["min", "min"],
)

# Nevergrad
from ray.tune.search.nevergrad import NevergradSearch
search_alg = NevergradSearch(
    metric=["obj_D95_HRV", "obj_V20_Lung"],
    mode=["min", "min"],
)

# Grid (no special search alg needed; use tune.grid_search in param_space)
search_space = {
    "w_0": tune.grid_search([0.1, 0.5, 1.0]),
    "w_1": tune.grid_search([0.1, 0.5, 1.0]),
    # ...
}
search_alg = None