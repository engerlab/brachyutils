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
from brachyutils.geometry.catheter_utils.catheter_table import Catheter, CatheterTable
 
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
        lower_dwelltime: Optional[float] | Dict[str, float] = 0.0,
        upper_dwelltime: Optional[float] | Dict[str, float] = 100.0,
        dose_rates_dict: Optional[Dict[str, np.ndarray]] = None,
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
        self.dose_rates_dict = dose_rates_dict
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
                    dose_rate_map = self.dose_rates_dict[dwell.index] if self.dose_rates_dict is not None else None,
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


class CatheterTableOptim_Gurobi():
    r"""
    ### Purpose:
    - a class to optimize the catheter table using Gurobi.
    ### Attributes:
    XXX: tbd
    """
    def __init__(
        self,
        plan: BrachyPlan,
        roi_margin_mm: float = 5.0,
        multi_processing: bool = False,
        ):
        r"""
        ### Purpose:
        - An catheter table optimization object for Gurobi solver. 
        ### Inputs:
        - `plan`: BrachyPlan := the brachytherapy plan to be optimized.
        - `roi_margin_mm`: float := margin in mm to add around the ROIs when resampling to the optimization grid.
        """
        # # Initialize the attributes to their default values
        self.plan: BrachyPlan = plan
        self.solver = "gurobi"
        self.model = None
        self.catheter_vars: List[CatheterVar_Gurobi] = []
        self.roi_bounds: List[List[float]] = None
        self.roi_margin_mm: float = roi_margin_mm if isinstance(roi_margin_mm, list) else [roi_margin_mm] * 3
        self.solution_found: bool = False
        self.solve_time: float = 0.0
        self.multi_processing = multi_processing
        
        # attributes for later developement XXX
        # self.target_constraints_coords = []
        # self.hotspot_constraints_coords = []
        # self.hotspot_threshold = None
        # self.structure_weights_d = {}

        # start buliding this optimization object
        self.model = self.initialize_model(self.solver)
        self.catheter_vars = self.set_catheter_variables(
            plan=self.plan,
            model=self.model,
            )
        self.roi_bounds = self.get_optimization_roi_bounds(
            plan=self.plan,
            roi_margin_mm=self.roi_margin_mm,
        )
        self.set_penalty_function_and_constraints(
            plan=self.plan,
            catheter_vars=self.catheter_vars,
            model=self.model,
            multi_processing=self.multi_processing,
        )
        
    def initialize_model(self, solver: str, pth_logfile) -> Model:
        r"""
        ### Purpose:
        - initializes the Gurobi optimization model and set the log paths.
        ### Inputs:
        - `solver`: str := the solver to be used. only "gurobi" is supported.
        - `pth_logfile`: Optional[Path] := the path to the logfile. if
        None, default is temp_data/gurobi_model.log.
        ### Returns:
        - Model := the initialized Gurobi model.
        """
        if solver.lower() != "gurobi":
            raise ValueError("Only Gurobi solver is supported in this class.")
        if pth_logfile is None:
            pth_logfile = Path("temp_data/gurobi_model.log").resolve()
        pth_logfile.parent.mkdir(parents=True, exist_ok=True)
        model = Model("CatheterTable_Optimization")
        model.setParam("LogFile", str(pth_logfile))
        return model

    def set_catheter_variables(
        plan: BrachyPlan,
        model: Model,
        ) -> List[CatheterVar_Gurobi]:
        catheter_vars = []
        for catheter in plan.catheter_table:
            # get the dose rate matrices for each catheter
            dose_rates_dict = plan.get_dose_rate_matrices_for_catheter(catheter.index)
            catheter_vars.append(
                CatheterVar_Gurobi(
                catheter=catheter,
                model=model,
                dose_rates_dict=dose_rates_dict,
                )
            )
        model.update()

    def get_optimization_roi_bounds():
        pass

    def set_penalty_function_and_constraints():
        pass