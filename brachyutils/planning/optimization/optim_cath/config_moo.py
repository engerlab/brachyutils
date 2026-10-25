from pydantic import BaseModel, ConfigDict
import numpy as np
from typing import Dict, Literal

class Config_MOO(BaseModel):
    r"""
    ### Purpose:
    - To specifiy the parameters for multi objective optimization (MOO).

    ### Inputs:
    - `moo_class`:= The name of the class to be used for MOO. The options are
    `MOO_Ax`, `MOO_Optuna`.
    - `sampler_name_id`:= The sampler to be used inside the MOO class. The options
    are class specific. For `MOO_Ax`, the options are: "qNEHVI", "fast", "quality".
    For `MOO_Optuna`, the options are "AutoSampler", "NSGAIISampler", "TPESampler",
    "GPSampler", "BoTorchSampler", "NSGAIIISampler",...
    - `parameter_space` := A dictionary mapping the names of the parameters to be
    optimized to their range [min, max]. The names of the parameters are
    some attributes of the Optimization_Config class followed by the name of
    that structure in paranthesis. For example: 
    penalty_weight_linear(CTV) : [1, 500]
    - `max_workers` := The number of threads for your machine for cuncurrently optimizating
    the optimization model in parallel.
    - `device`: cpu or cuda depending on the sampler. this has only been tested for qEHVI
    - `anchor_dvh_metric`: Dict[str, float] := Provide a DVH metric name to be used for
    anchoring the plan, i.e. the dwell times will be scaled so that the desired value for
    this metric is met. Example: {"V100%(CTV)": 95} will scale the dwell times so that V100%(CTV) = 95.
    - `slack_space_shrinkage`: float := Based on random sampling, the parameter space
    is shrunk to focus on the region of parameters that yield clinically acceptable plans. 
    This value is slack on relaxing the DVH metric goals. For example, if the DVH metric 
    goal is [">=", 100] and the slack is 0.1, then the parameter space will be shrunk to
    focus on the region of parameters that yield plans with DVH metric >= 90.
    - `slack_objective_threshold`: Relative slack for each DVH outcome
    constraint. A float applies to all metrics; a dictionary sets slack
    by DVH metric name (unspecified metrics use zero slack). Values must
    be in [-1, 1]. Positive slack relaxes constraints and negative slack
    tightens them. For a goal of `">= 100"`, slack `0.1` sets the Ax
    constraint to `>= 90`; for a goal of `"<= 100"`, it sets it to
    `<= 110`.
    - `n_random_samples`:= The number of samples to generate randomly and evaluate.
    - `use_random_samples_in_warmup`:= If True, the random samples will be used to warm up
    the tuner. If False, the random samples will just be stored to self.trial_data.
    - `n_tuning_trials`: int := The number of outer iterations (batches) to run.
    - `trial_batch_size`:= The number of trials to ask for and evaluate together on
    each iteration. Total trials evaluated = n_trials * batch_size.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # validate_assignment=True,
    )
    moo_class: Literal["MOO_Ax", "MOO_Optuna"]
    sampler_name_id: Literal[
        "qNEHVI", "fast", "quality",
        "AutoSampler",
        "NSGAIISampler",
        "TPESampler",
        "GPSampler",
        "BoTorchSampler",
        "NSGAIIISampler",
        ]
    parameter_space: Dict[str, np.typing.ArrayLike]
    max_workers: int = 16
    device:Literal["cpu", "cuda"] = "cpu"
    anchor_dvh_metric: Dict[str, float] = None
    slack_space_shrinkage: float = None
    slack_objective_threshold: float | Dict[str, float] = 0
    n_random_samples: int
    use_random_samples_in_warmup:bool = True
    n_tuning_trials:int
    trial_batch_size:int
