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
    BrachyDwellTimeOptim, BrachyDwellTime, get_optimization_roi_bounds, resample_crop_the_mask_or_contour_to_optimGrid,
    compute_dose_rate_matrices, Optimization_Config
)
from brachyutils.planning.optimization.optim_gurobi import DwellTime_Gurobi
import multiprocessing as mp
from functools import partial

# likley to be factored out later
from brachyutils.geometry.catheter_utils.catheter_table import Catheter, CatheterTable
from itertools import chain

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
        dose_rates: Optional[List[np.ndarray]] = None,
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
        - `dose_rates`: Optional[List[np.ndarray]] := the dose rate matrices for all the dwell positions in this catheter.
        """
        self._model_variable: Var = None
        self.name: str = f"catheter_{catheter.index+1}"
        self.dwelltime_variables: List[DwellTime_Gurobi] = []
        self.dose_rates = dose_rates
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
                    dose_rate_map = self.dose_rates[dwell.index] if self.dose_rates is not None else None,
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
    
    def __iter__(self):
        for dwell_var in self.dwelltime_variables:
            yield dwell_var

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
        self.roi_bounds = get_optimization_roi_bounds(
            plan=self.plan,
            dwellTimeVariables=list(chain.from_iterable(self.catheter_vars)),
            roi_margin_mm=self.roi_margin_mm,
        )
        self.set_penalty_function_and_constraints(
            plan=self.plan,
            catheter_vars=self.catheter_vars,
            model=self.model,
            multi_processing=self.multi_processing,
        )

    def initialize_model(self, solver: str, pth_logfile:str=None) -> Model:
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
        self,
        plan: BrachyPlan,
        model: Model,
        ) -> List[CatheterVar_Gurobi]:
        catheter_vars = []
        for catheter in plan.catheter_table:
            # get the dose rate matrices for each catheter
            dose_rates = plan.get_dose_rate_matrices_for_catheter(catheter.index)
            catheter_vars.append(
                CatheterVar_Gurobi(
                catheter=catheter,
                model=model,
                dose_rates=dose_rates,
                )
            )
        model.update()
        return catheter_vars
    
    def set_penalty_function_and_constraints(
        self,
        plan: BrachyPlan,
        catheter_vars: List[CatheterVar_Gurobi],
        model: Model,
        multi_processing: bool = False,):
        r"""
        ### Purpose:
        - sets the penalty function and constraints for the optimization model.
        ### Inputs:
        - `plan`: BrachyPlan := the brachytherapy plan to be optimized.
        - `catheter_vars`: List[CatheterVar_Gurobi] := the catheter variables to be used in the optimization.
        - `model`: Model := the Gurobi model to which the variables will be added.
        - `multi_processing`: bool := whether to use multi-processing for dose rate matrix computations.
        """
        if not plan.structure_list:
            raise ValueError("Plan does not contain any structures.")

        penalty_terms = {
        "linear": 0,
        "quadratic": 0,
        "hotspot": 0,
        "uniformity": 0
        }
        for structure in plan.structure_list:
            if structure.optimization_config is None:
                continue
            if "hotspot_estimator:" in structure.name.lower():
                continue
            structure_mask = structure.mask
            optim_spacing = structure.optimization_config.spacing_mm
            min_dose = structure.optimization_config.min_dose
            max_dose = structure.optimization_config.max_dose

            # XXX convert these to gurobi variable and build a constraint for each
            # this will get rid of sebbers hacky way dictionaries for MOBO
            target_dose = structure.optimization_config.dose_voxel_goal
            linear_weight = structure.optimization_config.penalty_weight_linear
            quadratic_weight = structure.optimization_config.penalty_weight_quadratic
            uniformity_weight = structure.optimization_config.penalty_weight_uniformity
            hotspot_threshold = structure.optimization_config.hotspot_threshold
            hotspot_weight = structure.optimization_config.penalty_weight_hotspot
            penalty_weight_variance_time = structure.optimization_config.penalty_weight_variance_time

            structure_mask = resample_crop_the_mask_or_contour_to_optimGrid(
                structure_mask=structure_mask,
                template_dose_obj=plan.combined_dose,
                optim_spacing=optim_spacing,
                roi_bounds=self.roi_bounds
                )

            for cath_var in catheter_vars:
                # Build dose rate matrix and dwell time vector for this structure
                dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
                    cath_var,
                    plan,
                    structure.name,
                    structure_mask,
                    optim_spacing,
                    self.roi_bounds,
                    max_workers=16,
                    shift_origin=True,
                    multi_processing=multi_processing
                )
                if not dose_rate_matrices:
                    continue

                t_MVar = MVar(dwell_vars)
                A_sparse = np.column_stack(dose_rate_matrices)
                num_dose_points = A_sparse.shape[0]
                if num_dose_points == 0:
                    continue
                # XXX conver to gurobi variable and set it using a constraint
                target_dose_vec = np.full((num_dose_points,), target_dose)
                
                if structure.target_volume:
                    if linear_weight > 0 or quadratic_weight > 0:
                        x_slack = model.addMVar(
                            shape=num_dose_points,
                            lb=0.0,
                            ub=target_dose - min_dose,
                            name=f"dose_slack_{structure.name}"
                            )
                        model.addConstr(
                            A_sparse @ (cath_var._model_variable * t_MVar) + x_slack >= target_dose_vec,
                            name=f"dose_target_{structure.name}"
                            )
                    if linear_weight > 0:
                        # XXX conver to gurobi variable and set it using a constraint
                        linear_weight_vec = np.full(num_dose_points, linear_weight / num_dose_points)
                        penalty_terms["linear"] += linear_weight_vec @ x_slack

                    if quadratic_weight > 0:
                        # XXX conver to gurobi variable and set it using a constraint
                        quadratic_weight_vec = np.full(num_dose_points, quadratic_weight / num_dose_points)
                        penalty_terms["quadratic"] += quadratic_weight_vec @ (x_slack * x_slack)

                    if uniformity_weight > 0:
                        # XXX conver to gurobi variable and set it using a constraint
                        y_uniform = model.addMVar(
                            shape=num_dose_points,
                            lb=-GRB.INFINITY,
                            ub=target_dose - min_dose,
                            name=f"uniform_slack_{structure.name}"
                        )
                        # Uniformity constraints: A @ dwell_times + y_uniform == target_dose
                        model.addConstr(
                            A_sparse @ (cath_var._model_variable * t_MVar) + y_uniform == target_dose_vec,
                            name=f"dose_uniform_{structure.name}"
                        )

