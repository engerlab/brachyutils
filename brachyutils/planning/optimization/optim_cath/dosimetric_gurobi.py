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
from brachyutils.planning.optimization.optim_gurobi import (
    DwellTime_Gurobi, _run, _get_optimized_plan_from_model)
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
    - `plan`: BrachyPlan := the brachytherapy plan to be optimized.
    - `solver`: str := the solver used for optimization. default is "gurobi
    - `model`: Model := the Gurobi optimization model.
    - `catheter_vars`: List[CatheterVar_Gurobi] := the catheter
        variables used in the optimization.
    - `dwellTimeVariables`: List[DwellTime_Gurobi] := the dwell time
        variables used in the optimization.
    - `roi_bounds`: List[List[float]] := the bounds of the regions of interest
        used in the optimization.
    - `roi_margin_mm`: float := margin in mm to add around the ROIs when resampling to the optimization grid.
    - `solution_found`: bool := whether a solution was found.
    - `solve_time`: float := the time taken to solve the optimization problem.
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
        - `multi_processing`: bool := whether to use multi-processing for cropping, masking and resampling 
        dose rate maps.
        """
        # # Initialize the attributes to their default values
        self.plan: BrachyPlan = plan
        self.solver = "gurobi"
        self.model = None
        self.catheter_vars: List[CatheterVar_Gurobi] = []
        self.dwellTimeVariables: List[DwellTime_Gurobi] = []
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
        self.dwellTimeVariables = list(chain.from_iterable(self.catheter_vars))
        self.roi_bounds = get_optimization_roi_bounds(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
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
    
    def _set_hyperparameters_per_structure(
        self,
        optimization_config: Optimization_Config,
        structure_name: str,
        model: Model,):
        r"""
        ### Purpose:
        - sets the hyper-parameters for each structure in the optimization model. This allows us to 
        avoid keeping track of the index of each variable.
        The hyper-parameters are stored in optimization config of each structure.
        They include:
            - target dose
            - penalty weights for linear, quadratic, uniformity and hotspot penalties
            - hotspot threshold
        ### Inputs:
        - `optimization_config`: Optimization_Config := the optimization configuration for the structures.
        - `model`: Model := the Gurobi model to which the variables will be added.
        """
        td = model.addVar(name=f"voxel_goal_{structure_name}")
        model.addConstr(
            td == optimization_config.dose_voxel_goal,
            name=f"voxel_goal_value_{structure_name}"
        )
        linear_weight = model.addVar(name=f"linear_weight_{structure_name}")
        model.addConstr(
            linear_weight == optimization_config.penalty_weight_linear,
            name=f"linear_weight_value_{structure_name}"
        )
        quadratic_weight = model.addVar(name=f"quadratic_weight_{structure_name}")
        model.addConstr(
            quadratic_weight == optimization_config.penalty_weight_quadratic,
            name=f"quadratic_weight_value_{structure_name}"
        )
        uniformity_weight = model.addVar(name=f"uniformity_weight_{structure_name}")
        model.addConstr(
            uniformity_weight == optimization_config.penalty_weight_uniformity,
            name=f"uniformity_weight_value_{structure_name}"
        )
        hotspot_threshold = model.addVar(name=f"hotspot_threshold_{structure_name}")
        model.addConstr(
            hotspot_threshold == optimization_config.hotspot_threshold,
            name=f"hotspot_threshold_value_{structure_name}"
        )
        hotspot_weight = model.addVar(name=f"hotspot_weight_{structure_name}")
        model.addConstr(
            hotspot_weight == optimization_config.penalty_weight_hotspot,
            name=f"hotspot_weight_value_{structure_name}"
        )
        penalty_weight_variance_time = model.addVar(name=f"variance_time_weight_{structure_name}")
        model.addConstr(
            penalty_weight_variance_time == optimization_config.penalty_weight_variance_time,
            name=f"variance_time_weight_value_{structure_name}"
        )
        model.update()
        return model

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

            model = self._set_hyperparameters_per_structure(
                optimization_config=structure.optimization_config,
                structure_name=structure.name,
                model=model,
            )
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
            # Build dose rate matrix and dwell time vector for this structure
            dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
                self.dwellTimeVariables,
                plan,
                structure.name,
                structure_mask,
                optim_spacing,
                self.roi_bounds, # XXX ensure cropping is optional for high efficiency
                max_workers=16,
                shift_origin=True,
                multi_processing=multi_processing
            )
            if not dose_rate_matrices:
                continue

            # now sort the dose rate matrices and dwell vars per catheter
            t_MVar = MVar(dwell_vars)
            c_MVar = MVar([c._model_variable for c in catheter_vars for _ in c])
            A_sparse = np.column_stack(dose_rate_matrices)
            num_dose_points = A_sparse.shape[0]
            if num_dose_points == 0:
                continue

            voxel_goal_vec = MVar([
                model.getVarByName(f"voxel_goal_{structure.name}") 
                for _ in range(num_dose_points)])

            if structure.target_volume:
                if linear_weight > 0 or quadratic_weight > 0:
                    x_slack = model.addMVar(
                        shape=num_dose_points,
                        lb=0.0,
                        ub=target_dose - min_dose,
                        name=f"dose_slack_{structure.name}"
                        )
                    model.addConstr(
                        A_sparse @ (c_MVar * t_MVar) + x_slack >= voxel_goal_vec,
                        name=f"dose_target_{structure.name}"
                        )
                if linear_weight > 0:
                    linear_weight_vec = MVar([
                        model.getVarByName(f"linear_weight_{structure.name}")
                        for _ in range(num_dose_points)])
                    penalty_terms["linear"] += (linear_weight_vec/num_dose_points) @ x_slack

                if quadratic_weight > 0:
                    quadratic_weight_vec = MVar([
                        model.getVarByName(f"quadratic_weight_{structure.name}")
                        for _ in range(num_dose_points)])
                    penalty_terms["quadratic"] += (quadratic_weight_vec/num_dose_points) @ (x_slack * x_slack)

                if uniformity_weight > 0:
                    y_uniform = model.addMVar(
                        shape=num_dose_points,
                        lb=-GRB.INFINITY,
                        ub=target_dose - min_dose,
                        name=f"uniform_slack_{structure.name}"
                    )
                    # Uniformity constraints: A @ dwell_times + y_uniform == target_dose
                    model.addConstr(
                        A_sparse @ (c_MVar * t_MVar) + y_uniform == voxel_goal_vec,
                        name=f"dose_uniform_{structure.name}"
                    )
                    uniformity_weight_vec = MVar([
                        model.getVarByName(f"uniformity_weight_{structure.name}")
                        for _ in range(num_dose_points)])
                    penalty_terms["uniformity"] += (uniformity_weight_vec/num_dose_points) @ (y_uniform * y_uniform)

                if hotspot_weight > 0 and hotspot_threshold is not None:
                    self._set_hotspot_penalty_and_constraints(
                        plan=plan,
                        model=model,
                        optim_spacing=optim_spacing,
                        roi_bounds=self.roi_bounds,
                        structure_name=structure.name,
                        catheter_vars=catheter_vars)

                if penalty_weight_variance_time > 0:
                    # XXX conver to gurobi variable and set it using a constraint
                    pass
            # OAR constraints and penalties
            else:
                if linear_weight > 0 or quadratic_weight > 0:
                    x_slack_oar = model.addMVar(
                        shape=num_dose_points,
                        lb=0.0,
                        ub=max_dose - min_dose,
                        name=f"dose_slack_oar_{structure.name}"
                    )
                    model.addConstr(
                        A_sparse @ (c_MVar * t_MVar) - x_slack_oar <= voxel_goal_vec,
                        name=f"dose_oar_{structure.name}"
                    )

                if linear_weight > 0:
                    linear_weight_vec_oar = MVar([
                        model.getVarByName(f"linear_weight_{structure.name}")
                        for _ in range(num_dose_points)])
                    penalty_terms["linear"] += (linear_weight_vec_oar/num_dose_points) @ x_slack_oar

                if quadratic_weight > 0:
                    quadratic_weight_vec_oar = MVar([
                        model.getVarByName(f"quadratic_weight_{structure.name}")
                        for _ in range(num_dose_points)])
                    penalty_terms["quadratic"] += (quadratic_weight_vec_oar/num_dose_points) @ (x_slack_oar * x_slack_oar)

        # Set the objective function
        model.setObjective(
            penalty_terms["linear"]
            + penalty_terms["quadratic"]
            + penalty_terms["uniformity"]
            + penalty_terms["hotspot"],
            GRB.MINIMIZE
        )
        model.update()

    def _set_hotspot_penalty_and_constraints(
        self,
        plan: BrachyPlan,
        model: Model,
        optim_spacing: float,
        roi_bounds: List[List[float]],
        structure_name: str,
        catheter_vars: List[CatheterVar_Gurobi],
        ) -> LinExpr:
        r"""
        ### Purpose:
        - sets the hotspot penalty and constraints for the optimization model.
        ### Inputs:
        - None XXX to be implemented later.
        """
        hotspot_masks = [
            structure.mask for structure in plan.structure_list
            if "hotspot_estimator" in structure.name]
        if len(hotspot_masks) == 1:
            processed_mask = resample_crop_the_mask_or_contour_to_optimGrid(
                template_dose_obj=plan.combined_dose,
                structure_mask=hotspot_masks[0],
                optim_spacing=optim_spacing,
                roi_bounds=roi_bounds
            )
            dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
                dwellTimeVariables=self.dwellTimeVariables,
                plan=plan,
                structure_name=processed_mask.name,
                structure_mask=processed_mask,
                optim_spacing=optim_spacing,
                roi_bounds=self.roi_bounds, # XXX ensure cropping is optional for high efficiency
                shift_origin=True
            )

            t_MVar = MVar.fromlist(dwell_vars)
            c_MVar = MVar([c._model_variable for c in catheter_vars for _ in c])
            A = np.column_stack(dose_rate_matrices)
            num_dose_points = A.shape[0]
            x_slack = model.addMVar(
                shape=num_dose_points,
                name=f"hotspot_slack_{processed_mask.name.replace(':', '_')}")

            voxel_goal_vec = MVar([
                model.getVarByName(f"voxel_goal_{structure_name}") 
                for _ in range(num_dose_points)])

            model.addConstr(
                A @ (c_MVar * t_MVar) - x_slack <= (voxel_goal_vec * model.getVarByName(f"hotspot_threshold_{structure_name}")),
                name=f"hotspot_constraint_{processed_mask.name.replace(':', '_')}",
            )
            hotspot_weight = model.getVarByName(f"hotspot_weight_{structure_name}")
            hotspot_penalty = sum((x_slack))*hotspot_weight/num_dose_points
            return hotspot_penalty
        else:
            raise NotImplementedError("Multiple hotspot estimators not supported, please use \
optim_gurobi.BrachyOptim_Gurobi instead if only using dwell time optimization. else implement it \
youself :p.")

    def run(self):
        r"""
        ### Purpose:
        - A function to run the optimizer. See `BrachyDwellTimeOptim.run` for details. 
        """
        self.model, self.solution_found, self.solve_time = _run(self.model)

    def get_optimized_plan_from_model(
        self,
        inplace:bool=True,
        ) -> BrachyPlan | None:
        r"""
        See `BrachyDwellTime.get_optimized_plan_from_model` for details.
        """
        # XXX adapt this for catheter table optimization, maintain the signature!
        self.model, outplan, self.solution_found, self.solve_time = _get_optimized_plan_from_model(
            plan=self.plan,
            model=self.model,
            inplace=inplace
            )
        return outplan

    def bound_variables(
        self,
        new_bounds: Dict[str, Dict[str, float]],
        ):
        r"""
        ### Purpose:
        - bound specific catheter or dwell time variables in the optimization model.
        ### Inputs:
        - `new_bounds`: Dict[str, Dict[str, float]] := a dictionary where keys are catheter or dwell time variable names
            and values are dictionaries with 'equality', 'lower' and 'upper' keys for the new bounds.
        """
        pass
