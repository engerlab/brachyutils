from typing import List
from copy import deepcopy
import warnings
import time
import numpy as np
from pathlib import Path
from brachyutils.types import BrachyPlan
from brachyutils.planning.optimization.optim_utils import (
    DwellTimeOptimizer_ABC, BrachyDwellTime_ABC, crop_mask_resample_dose_rate_map
)

class DwellTime_ORTools(BrachyDwellTime_ABC):
    r"""
    ### Purpose:
    - A class to represent a DwellTimeVariable in the dwell time optimization problem using OR-Tools.
    See `BrachyDwellTime_ABC` for more details on the attributes and methods.
    """
    def build_backend_variable(self, model):
        r"""
        See `BrachyDwellTime_ABC.build_backend_variable` for more details.
        """
        pass
    
    def set_bounds(self, *, lower_bound: float | None = None, upper_bound: float | None = None) -> None:
        r"""
        See `BrachyDwellTime_ABC.set_bounds` for details.
        """
        pass

    def __init__(self, model, **data):
        r"""
        ### Purpose:
        - Initialize the DwellTimeVariable
        ### Inputs:
        - model: The OR-Tools model to which this variable will be added.
        - data: Dictionary containing the attributes of the DwellTimeVariable.
        """
        super().__init__(**data)
        self.build_backend_variable(model)