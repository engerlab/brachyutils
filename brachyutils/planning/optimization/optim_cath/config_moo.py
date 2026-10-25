from pydantic import BaseModel, ConfigDict
import numpy as np
from typing import Dict, Literal

class Config_MOO(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # validate_assignment=True,
    )
    Moo_class: Literal["MOO_Ax", "MOO_Optuna"]
    sampler_name_id: Literal["qNEHVI", ""]
    parameter_space: Dict[str, np.typing.ArrayLike]
    max_workers: int = 16
    anchor_dvh_metric: Dict[str, float] = None
    slack_space_shrinkage: float = None
    slack_objective_threshold: float | Dict[str, float] = 0
    n_random_samples: int
    use_random_samples_in_warmup:bool = True
    n_trials:int
    trial_batch_size:int
