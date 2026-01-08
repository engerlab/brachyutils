from typing import Dict, List, Optional
import tqdm
import time
from copy import deepcopy
import warnings
import time
from multiprocessing import Pool
import os 
from pathlib import Path

from gurobipy import Model, Var, GRB, MVar, Env, QuadExpr, LinExpr
import numpy as np
import pandas as pd

from opentps.core.processing.imageProcessing.sitkImageProcessing import image3DToSITK
from brachyutils.types import BrachyPlan
from brachyutils.planning.optimization.optim_utils import (
    BrachyDwellTimeOptim, BrachyDwellTime, resample_crop_the_mask_or_contour_to_optimGrid,
    compute_dose_rate_matrices, Optimization_Config
)
from brachyutils.planning.optimization.optim_gurobi import DwellTime_Gurobi
import multiprocessing as mp
from functools import partial

# likley to be factored out later
from brachyutils.geometry.catheter_utils.catheter_table import Catheter 

class CatheterVar_Gurobi():
    r"""
    ### Purpose:
    - a class representing a catheter variable to be used in Gurobi optimization models    
    ### Attributes:
    - `name`: str := the name of the catheter variable. usually its index in the CatheterTable class.
    - `dwelltime_variables`: List[DwellTime_Gurobi] := a list of dwell time variables associated with this catheter.
    - `_model_variable`: Var := the Gurobi variable representing this catheter in the optimization model.
    
    """
    def __init__(
        self,
        catheter: Catheter,
        model: Model,
        lower_dwelltime: Optional[float] | Dict[str:float] = 0.0,
        upper_dwelltime: Optional[float] | Dict[str:float] = 100.0,
        dose_rate_dict: Optional[Dict[str, np.ndarray]] = None,
        ):
        r"""
        ### Purpose:
        - a class representing a catheter variable to be used in Gurobi optimization models
        ### Inputs:
        - `name`: str := the name of the catheter variable. usually its index in the CatheterTable class.
        - `dwelltime_variables`: List[DwellTime_Gurobi] := a list of dwell time variables associated with this catheter.
        - `model`: Model := the Gurobi model to which the variables will be added.
        - `lower_dwelltime`: Optional[float] | Dict[str:float] := the lower bound(s) for the dwell time variables.
        - `upper_dwelltime`: Optional[float] | Dict[str:float] := the upper bound(s) for the dwell time variables.
        - `dose_rate_dict`: Optional[Dict[str, np.ndarray]] := the dose rate matrices for the dwell positions in this catheter.
        """
        self.name: str = f"catheter_{catheter.index+1}"
        self.dwelltime_variables: List[DwellTime_Gurobi] = []
        self.dose_rate_dict = dose_rate_dict
        self.build_backend_variable(model=model)
        for dwell in catheter.dwells:
            self.dwelltime_variables.append(
                DwellTime_Gurobi(
                    model = model,
                    name = f"{self.name}_dwell_{dwell.index+1}",
                    dwell_time = dwell.time,
                    lower_bound = lower_dwelltime,
                    upper_bound = upper_dwelltime,
                    coordinates = dwell.position,
                    # dose_rate_map = dwell.dose_rate_map,
                )
            )
    def build_backend_variable(self, model: Model):
        r"""
        ### Purpose:
        - builds the backend Gurobi variables this catheter, which will be used 
        to activate (set to 1) or deactivate the dwell times (set to 0).
        ### Args:
        - `model`: Model := the Gurobi model to which the variables will be added.
        """
        if not isinstance(model, Model):
            raise TypeError("model must be a Gurobi Model instance.")
        self._model_variable: Var = model.addVar(
            vtype=GRB.BINARY,
            name=self.name,
            lb=0,
            ub=1
        )
    