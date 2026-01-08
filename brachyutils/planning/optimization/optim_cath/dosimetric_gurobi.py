from typing import List, Optional
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
from scipy.sparse import csr_matrix

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

class Catheter_Gurobi():    
    def __init__(
        self,
        catheter: Catheter,
        model: Model,
        ):
        r"""
        ### Purpose:
        - a class representing a catheter variable to be used in Gurobi optimization models
        ### Attributes:
        - `name`: str := the name of the catheter variable. usually its index in the CatheterTable class.
        - `dwelltime_variables`: List[DwellTime_Gurobi] := a list of dwell time variables associated with this catheter.
        """
        self.name: str = f"catheter_{catheter.index+1}"
        self.dwelltime_variables: List[DwellTime_Gurobi] = []
        self.build_backend_variable(model=model)
        for dwell in catheter.dwells:
            self.dwelltime_variables.append(
                DwellTime_Gurobi(
                    model = model,
                    name = f"{self.name}_dwell_{dwell.index+1}",
                    dwell_time = dwell.dwell_time,
                    lower_bound = dwell.lower_bound,
                    upper_bound = dwell.upper_bound,
                    coordinates = dwell.coordinates,
                    dose_rate_map = dwell.dose_rate_map,
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
    