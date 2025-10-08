# from abc import ABC, abstractmethod
from typing import List
import tqdm
import time
from copy import deepcopy
import warnings
import time
from multiprocessing import Pool
import os 
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gurobipy import Model, Var, GRB, MVar, Env, QuadExpr, LinExpr
import SimpleITK as sitk
import numpy as np
import pandas as pd

from opentps.core.data.images import ROIMask
from opentps.core.processing.imageProcessing.sitkImageProcessing import image3DToSITK
from brachyutils.types import BrachyPlan
from brachyutils.planning.optimization.optim_utils import (
    BrachyDwellTimeOptim, BrachyDwellTime, 
    crop_resample_dose_rate_map_and_mask
)


def process_variable(variable, structure_name, structure_mask, plan, optim_spacing, roi_bounds, shift_origin:bool=False):

    if "hotspot_estimator:" in structure_name.lower():
        relevant_dwells = structure_name.lower().split("hotspot_estimator:")[1].split("/")
        if variable.name not in relevant_dwells:
            return None
    dwell_var = variable._model_variable

    if (
        isinstance(structure_mask, ROIMask)
        and np.allclose(plan.combined_dose.dose_image.origin, structure_mask.origin)
        and np.allclose(plan.combined_dose.dose_image.spacing, structure_mask.spacing)
        and np.allclose(plan.combined_dose.dose_image.spacing, optim_spacing)
        and np.all(np.swapaxes(variable.dose_rate_map, 0, 2).shape == structure_mask.imageArray.shape)
    ):
        masked_dose_array = np.swapaxes(variable.dose_rate_map, 0, 2)
    else:
        dose_rate_obj, structure_for_masking = crop_resample_dose_rate_map_and_mask(
            dose_rate_map=variable.dose_rate_map,
            template_dose_obj=plan.combined_dose,
            roi_bounds=roi_bounds,
            structure_mask=structure_mask,
            optim_spacing=optim_spacing,
            sitk_interpolator_dose=sitk.sitkLinear,
            # Using Linear instead of NearestNeighbor since NN does a bad job when downsampling
            sitk_interpolator_contour=sitk.sitkLinear, #sitkNearestNeighbor # sitkLinear
            shift_origin=shift_origin
        )
        masked_dose_array = dose_rate_obj.dose_image.imageArray.astype(float)
        structure_mask = structure_for_masking

    structure_for_masking = structure_mask.imageArray.astype(bool)
    valid_dose_points = masked_dose_array[structure_for_masking == 1].flatten()

    return dwell_var, valid_dose_points

def compute_dose_rate_matrices(
        dwellTimeVariables, structure, structure_mask, plan, 
        optim_spacing, roi_bounds, max_workers:int=8, shift_origin:bool=False):
    dose_rate_matrices = []
    dwell_vars = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_variable,
                variable,
                structure.name,
                structure_mask,
                plan,
                optim_spacing,
                roi_bounds, 
                shift_origin
            ): variable
            for variable in dwellTimeVariables
        }

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Processing dwell positions dose rates for {structure.name}"):
            result = future.result()
            if result is not None:
                dwell_var, valid_dose_points = result
                dwell_vars.append(dwell_var)
                dose_rate_matrices.append(valid_dose_points)

    return dwell_vars, dose_rate_matrices

class DwellTime_Gurobi(BrachyDwellTime):
    r"""
    ### Purpose:
    - A class to represent a DwellTimeVariable in the dwell time optimization problem using Gurobi.
    See `BrachyDwellTime` for more details on the attributes and methods.
    """
    def build_backend_variable(self, model):
        if not isinstance(model, Model):
            raise ValueError("Model is not a Gurobi model. Please provide a Gurobi model.")
        self._model_variable = model.addVar(
            lb=self.lower_bound,
            ub=self.upper_bound,
            name=self.name,
            vtype=GRB.CONTINUOUS
        )

    def set_bounds(self, *, lower_bound: float | None = None, upper_bound: float | None = None) -> None:
        r"""
        See `BrachyDwellTime.set_bounds` for details.
        """
        if lower_bound is not None:
            self.lower_bound = lower_bound
            self._model_variable.lb = lower_bound
        if upper_bound is not None:
            self.upper_bound = upper_bound
            self._model_variable.ub = upper_bound
    
    def __init__(self, model: Model, **data):
        r"""
        ### Purpose:
        - A function to initialize the DwellTimeVariable.
        ### Inputs:
        - model: Model := The Gurobi model object.
        - data: dict := The data to initialize the DwellTimeVariable.
        """
        super().__init__(**data)
        self.build_backend_variable(model) 



def change_model_dose_to_target(new_target_dose:float, model:Model, coords_target_constraint:list):
    r'''This model changes the target dose of the dose calculation points that reside inside the tumor target volume on a gurobi model. 
    inputs: 
        - new_target_dose := the new target dose value for tumor in Gy. 
        -  model := is a gurobi model to be changed. 
        - coords_target_constraint := a list of coordinates that says which constraints belong to the target volume. 
            this list is obtained in the initialization step 
    outputs:
        - VOID := this function changes the state of the input model and returns nothing. 
    '''
    constr_list = list(model.getConstrs())

    model.setAttr('RHS', constr_list[coords_target_constraint[0]:coords_target_constraint[-1]], new_target_dose)
    model.update()


def _run(model: Model):
    r"""
    ### Purpose:
    - A function to run the optimizer. See `BrachyDwellTimeOptim.run` for details. 
    """
    time_start = time.time()
    model.optimize()
    time_end = time.time()
    solve_time = time_end - time_start
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        solution_found = True
    else:
        print("No optimal solution found.")
        solution_found = False
    return model, solution_found, solve_time

def _get_optimized_plan_from_model(
    plan: BrachyPlan,
    model: Model,
    inplace=True,
    ) -> BrachyPlan | None:
    r"""
    See `BrachyDwellTime.get_optimized_plan_from_model` for details.
    """
    if plan is None:
        raise ValueError("Plan is not set. Please set the plan first.")
    if model is None:
        raise ValueError("Model is not set. Please set the model first.")
    
    model, solution_found, solve_time = _run(model)

    dwelltime_and_name = []
    for x in model.getVars():
        if ("catheter" in x.VarName) and ("dwell" in x.VarName):
            dwelltime_and_name.append((x.X, x.VarName))
    
    if not solution_found:
        warnings.warn(
            "No optimal solution found. Return None.",
            stacklevel=2)
        return None

    # set the dwell time to the plan
    if inplace:
        outplan:BrachyPlan = plan
    else:
        outplan:BrachyPlan = deepcopy(plan)     

    for dwell_time, name in dwelltime_and_name:
        # set the dwell time to the optimized value
        if dwell_time < 0.1:
            dwell_time = 0
        for catheter in outplan.catheter_table:
            for dwell_position in catheter.dwells:
                if (
                    f"catheter_{catheter.index}_dwell_{dwell_position.index}"
                    == name
                ):
                    dwell_position.time = dwell_time
    # update the plan with the new dwell times
    outplan.update_plan_from_catheter_table()
    return model, outplan, solution_found, solve_time


class BrachyOptim_Gurobi(BrachyDwellTimeOptim):
    r"""
    ### Purpose:
    - A class using Gurobi to do dwell time optimization.
    See `BrachyDwellTimeOptim` for more details on the attributes and methods.
    """
    def __init__(
        self,
        plan:BrachyPlan,
        roi_margin_mm: List[float] | float = 5.0):
        r"""
        ### Purpose:
        - A function to initialize the optimizer.
        ### Parameters:
        - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
        - roi_margin_mm: The distance from the furthest dwell position along each axis:
            - [x_margin_mm, y_margin_mm, z_margin_mm] or a single float value for all axes.
        """
        super().__init__()
        self.plan = plan
        self.roi_margin_mm = roi_margin_mm if isinstance(roi_margin_mm, list) else [roi_margin_mm] * 3
        self.solver = "gurobi"
        self.target_constraints_coords = []
        self.structure_variables_d = {}
        self.model = self.initialize_model(self.solver)
        self.dwellTimeVariables = self.set_dwellTimeVariables(plan=self.plan)
        self.roi_bounds: List[List[float]] = self.get_optimization_roi_bounds(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            roi_margin_mm=self.roi_margin_mm,
            )
        self.set_penalty_function_and_constraints(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            model=self.model)

    def initialize_model(
        self,
        solver: str,
        pth_logfile:str = None) -> Model:
        r"""
        ### Purpose:
        - See `BrachyDwellTimeOptim.initialize_model` for details.
        ### Inputs:
        - solver:str := The name of the solver to be used. Default is None.
        - pth_logfile:str := The path to the log file for the solver. Default is None.
        ### Outputs:
        - model: Gurobi.Model := The gurobi model object.
        """
        if pth_logfile is None:
            pth_logfile = Path("temp_data/gurobi_model.log").resolve()
        pth_logfile.parent.mkdir(parents=True, exist_ok=True)
        if solver == "gurobi":
            model = Model("dwellTimeOptimizer")
            model.setParam("LogToConsole", 1)
            model.setParam("LogFile", str(pth_logfile))
            return model

    def set_dwellTimeVariables(
        self,
        plan: BrachyPlan,
        initial_dwell_time: float = 0.0,
        lower_bound: float = 0.0,
        upper_bound: float = 100,
    ) -> List[DwellTime_Gurobi]:
        r"""
        See `BrachyDwellTime.set_dwellTimeVariables` for details.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please initialize the model first.")
        dwellTimeVariable_list = []
        dwell_counter = 0
        for catheter in plan.catheter_table:
            for dwell_position in catheter.dwells:
                ag = np.where(plan.dose_rate_tensor[dwell_counter] == plan.dose_rate_tensor[dwell_counter].max())
                i = image3DToSITK(plan.combined_dose.dose_image)
                rev_argmax = list(int(u[0]) for u in ag)[::-1]
                tmp_pos = i.TransformIndexToPhysicalPoint(rev_argmax)
                tmp_idx = i.TransformPhysicalPointToIndex(list(dwell_position.position))
                # Arbitrary tolerance of 3 voxels and 3*spacing mm
                assert np.all(np.array(rev_argmax) - np.array(tmp_idx) < 3), (
                    "Dwell position does not match max dose rate position. Check dose rate tensor and dwell positions."
                )
                assert np.all(dwell_position.position - np.array(tmp_pos) < max(i.GetSpacing()) * 3), (
                    "Dwell position does not match max dose rate position. Check dose rate tensor and dwell positions."
                )

                dt_var_name = f"catheter_{catheter.index}_dwell_{dwell_position.index}"
                dwellTimeVariable_list.append(
                    DwellTime_Gurobi(
                        model=self.model,
                        name=dt_var_name,
                        dwell_time=initial_dwell_time,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        coordinates=dwell_position.position,
                        dose_rate_map=plan.dose_rate_tensor[dwell_counter]
                    )
                )
                dwell_counter += 1
        self.model.update()
        return dwellTimeVariable_list

    def get_optimization_roi_bounds(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[DwellTime_Gurobi],
        roi_margin_mm: List[float] = [5.0, 5.0, 5.0],
    ) -> List[List[float]]:
        r"""
        See `BrachyDwellTime_ABC.get_optimization_roi_bounds` for details.
        """
        return super().get_optimization_roi_bounds(
            plan=plan,
            dwellTimeVariables=dwellTimeVariables,
            roi_margin_mm=roi_margin_mm
        )

    def set_penalty_function_and_constraints(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[DwellTime_Gurobi],
        model: Model, 
        multi_processing: bool = True
        ) -> None:
        r"""
        ### Purpose:
        - A function to set up the optimization model's objective function and constraints based 
        on the plan. For target structures, slack variables are added to ensure doses meet target
        goals with linear and quadratic penalties for underdosing, plus uniformity penalties. 
        There is also the hot spot term for the hotspot estimator structures. For OARs, slack 
        variables with linear and quadratic penalties penalize overdosing above the target dose.

        The objective function takes the form:

        minimize sum(weights * penalties) where penalties include:
        - Linear penalties for over/under dosing relative to target dose
        - Quadratic penalties for over/under dosing 
        - Uniformity penalties for target volumes
        - Hotspot penalties for hotspot estimator structures encapsulating two closest dwell positions.

        ### Inputs:
        - plan: BrachyPlan := The plan containing structures and optimization configs
        - dwellTimeVariables: List[DwellTimeVariable] := The dwell time variables to optimize
        - model: Model := The Gurobi optimization model

        ### Outputs:
        None - sets up the model objective function and constraints directly
        """
        if not plan.structure_list:
            raise ValueError("Plan does not contain any structures.")

        from scipy import sparse as sp
        penalty_terms = {
        "linear": 0,
        "quadratic": 0,
        "hotspot": 0,
        "uniformity": 0
        }
    
        for structure in plan.structure_list:
            if structure.optimization_config is None:
                continue
            structure_mask = structure.mask
            optim_spacing = structure.optimization_config.spacing_mm
            target_dose = structure.optimization_config.dose_voxel_goal
            linear_weight = structure.optimization_config.penalty_weight_linear
            quadratic_weight = structure.optimization_config.penalty_weight_quadratic
            uniformity_weight = structure.optimization_config.penalty_weight_uniformity
            min_dose = structure.optimization_config.min_dose
            structure_max_dose = structure.optimization_config.max_dose
            hotspot_threshold = structure.optimization_config.hotspot_threshold
            hotspot_weight = structure.optimization_config.penalty_weight_hotspot
            # Build dose rate matrix and dwell time vector for this structure
            if not multi_processing:
                dose_rate_matrices = []
                dwell_vars = []
                for var in dwellTimeVariables:
                    dwell_var, valid_dose_points = process_variable(
                        var,
                        structure.name,
                        structure_mask,
                        plan,
                        optim_spacing,
                        self.roi_bounds, 
                        shift_origin=True
                        )
                    dwell_vars.append(dwell_var)
                    dose_rate_matrices.append(valid_dose_points)
            else:
                dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
                    dwellTimeVariables,
                    structure,
                    structure_mask,
                    plan,
                    optim_spacing,
                    self.roi_bounds,
                    max_workers=4, 
                    shift_origin=True
                )

            if not dose_rate_matrices:
                continue

            # convert the list of variables to a Gurobi variable Vector (MVar)
            t_MVar = MVar.fromlist(dwell_vars)
            # Stack dose rate matrices to create A matrix
            A = np.column_stack(dose_rate_matrices)  # Shape: (num_dose_points, num_variables)
            num_dose_points = A.shape[0]
            if num_dose_points == 0:
                continue
            # Convert A to sparse matrix
            A_sparse = sp.csr_matrix(A)
            # Create target dose vector
            target_dose_vec = np.full(num_dose_points, target_dose)

            if structure.target_volume:
                # Target volume constraints and penalties
                # Create slack variables for underdosing
                x_slack = model.addMVar(
                    shape=num_dose_points,
                    lb=0.0,
                    ub=target_dose - min_dose,
                    name=f"dose_slack_{structure.name}"
                )

                # Create slack variables for uniformity
                y_uniform = model.addMVar(
                    shape=num_dose_points,
                    lb=-GRB.INFINITY,
                    ub=target_dose - min_dose,
                    name=f"uniform_slack_{structure.name}"
                )
                # Dose constraints: A @ dwell_times + x_slack >= target_dose
                model.addConstr(
                    A_sparse @ t_MVar + x_slack >= target_dose_vec,
                    name=f"dose_target_{structure.name}"
                )

                # Uniformity constraints: A @ dwell_times + y_uniform == target_dose
                model.addConstr(
                    A_sparse @ t_MVar + y_uniform == target_dose_vec,
                    name=f"dose_uniform_{structure.name}"
                )
                ## Saving variables info for later potential resetting of the model
                self.structure_variables_d[structure.name] = {
                    "is_target_volume": structure.target_volume,
                    "dose_slack": x_slack,
                    "num_dose_points": num_dose_points,
                    "uniform_slack": y_uniform, 
                    "hotspot_estimator": "hotspot_estimator" in structure.name.lower()
                }
                # Add penalty terms using matrix operations
                linear_weight_vec = np.full(num_dose_points, linear_weight / num_dose_points)
                quadratic_weight_vec = np.full(num_dose_points, quadratic_weight / num_dose_points)
                uniformity_weight_vec = np.full(num_dose_points, uniformity_weight / (num_dose_points * 1000))

                # Linear penalty: sum(linear_weight_vec @ x_slack)
                penalty_terms["linear"] += linear_weight_vec @ x_slack
                # Quadratic penalty: sum(quadratic_weight_vec * x_slack * x_slack)
                penalty_terms["quadratic"] += quadratic_weight_vec @ (x_slack * x_slack)
                # Uniformity penalty: sum(uniformity_weight_vec * y_uniform * y_uniform)
                penalty_terms["uniformity"] += uniformity_weight_vec @ (y_uniform * y_uniform)

            elif "hotspot_estimator:" in structure.name.lower():
                # slack variable for hotspot estimator
                x_slack = model.addMVar(
                    shape=num_dose_points,
                    lb=0.0,
                    ub=hotspot_threshold * target_dose - min_dose,
                    name=f"hotspot_slack_{structure.name}"
                )
                # Hotspot estimator constraints
                model.addConstr(
                    A_sparse @ t_MVar - x_slack <= hotspot_threshold * target_dose_vec,
                )
                ## Saving variables info for later potential resetting of the model
                self.structure_variables_d[structure.name] = {
                    "is_target_volume": structure.target_volume,
                    "dose_slack": x_slack,
                    "num_dose_points": num_dose_points,
                    "uniform_slack": None, 
                    "hotspot_estimator": "hotspot_estimator" in structure.name.lower()
                }
                hotspot_weight_vec = np.full(num_dose_points, hotspot_weight / num_dose_points)
                penalty_terms["hotspot"] += (hotspot_weight_vec @ x_slack)

            else:
                # OAR (Organ at Risk) constraints and penalties
                # Create slack variables for overdosing
                x_slack = model.addMVar(
                    shape=num_dose_points,
                    lb=0.0,
                    ub=structure_max_dose - target_dose,
                    name=f"oar_slack_{structure.name}"
                )
                # Dose constraints: A @ dwell_times - x_slack <= target_dose
                model.addConstr(
                    A_sparse @ t_MVar - x_slack <= target_dose_vec,
                    name=f"dose_oar_{structure.name}"
                )
                ## Saving variables info for later potential resetting of the model
                self.structure_variables_d[structure.name] = {
                    "is_target_volume": structure.target_volume,
                    "dose_slack": x_slack,
                    "num_dose_points": num_dose_points,
                    "uniform_slack": None, 
                    "hotspot_estimator": "hotspot_estimator" in structure.name.lower()
                }

                # Add penalty terms
                linear_weight_vec = np.full(num_dose_points, linear_weight / num_dose_points)
                quadratic_weight_vec = np.full(num_dose_points, quadratic_weight / num_dose_points)

                penalty_terms["linear"] += linear_weight_vec @ x_slack
                penalty_terms["quadratic"] += quadratic_weight_vec @ (x_slack * x_slack)

        # Set objective function
        model.setObjective(
            penalty_terms["linear"]
            + penalty_terms["quadratic"]
            + penalty_terms["uniformity"]
            + penalty_terms["hotspot"],
            GRB.MINIMIZE
        )
        model.update()

        # obtain the first and last constraints that belong to target volume
        # generate the constraint list from the gurobi model file
        self.target_constraints_coords = []
        constr_list = list(model.getConstrs())
        for i in range(len(constr_list)):
            # get the coordinates of the constraints whose right hand side is equal to tumor target dose
            if constr_list[i].rhs == self.plan.dvh_metric_goals["target_dose"]:
                self.target_constraints_coords.append(i)


    def reset_model_from_config(self, model: Model, config: dict) -> None:
        penalty_terms = {
            "linear": 0,
            "quadratic": 0,
            "uniformity": 0,
            "hotspot": 0
        }

        # Storing new_target dose into a variable and then removing it from the config dict
        new_target_dose = None
        found = False
        for k, v in config.items():
            if "td_" in k.lower() :
                new_target_dose = deepcopy(v)
                to_remove = k
                found = True
                break
        if found:
            config.pop(to_remove, None)

        ## Reorganize dict by structure name because input config used in MOBO is flat
        reorganized_config = {}
        for k, v in config.items():
            struct_name = k.split("_")[2]
            if struct_name not in reorganized_config:
                reorganized_config[struct_name] = {}
            param_name = k.split("_")[0] 
            reorganized_config[struct_name][param_name] = v

        for struct_name, struct_d in reorganized_config.items():

            if "linear" not in struct_d.keys():
                linear_weight = 1
            else:
                linear_weight = struct_d["linear"]
            if "quadratic" not in struct_d.keys():
                quadratic_weight = 1
            else:
                quadratic_weight = struct_d["quadratic"]

            if "uniformity" in struct_d.keys():
                uniformity_weight = struct_d["uniformity"]
            else:
                uniformity_weight = 0 
            if "hotspot" in struct_d.keys():
                hotspot_weight = struct_d["hotspot"] 
            else:
                hotspot_weight = 0

            var_d = self.structure_variables_d[struct_name]
            x_slack = var_d["dose_slack"]
            y_uniform = var_d["uniform_slack"]
            num_dose_points = var_d["num_dose_points"]
            if var_d["is_target_volume"]:
                linear_weight_vec = np.full(num_dose_points, linear_weight / num_dose_points)
                quadratic_weight_vec = np.full(num_dose_points, quadratic_weight / num_dose_points)
                uniformity_weight_vec = np.full(num_dose_points, uniformity_weight / (num_dose_points * 1000))

                # Linear penalty: sum(linear_weight_vec @ x_slack)
                penalty_terms["linear"] += linear_weight_vec @ x_slack
                # Quadratic penalty: sum(quadratic_weight_vec * x_slack * x_slack)
                penalty_terms["quadratic"] += quadratic_weight_vec @ (x_slack * x_slack)
                # Uniformity penalty: sum(uniformity_weight_vec * y_uniform * y_uniform)
                penalty_terms["uniformity"] += uniformity_weight_vec @ (y_uniform * y_uniform)

            elif var_d["hotspot_estimator"]:
                hotspot_weight_vec = np.full(num_dose_points, hotspot_weight / num_dose_points)
                penalty_terms["hotspot"] += (hotspot_weight_vec @ x_slack)

            else:
                linear_weight_vec = np.full(num_dose_points, linear_weight / num_dose_points)
                quadratic_weight_vec = np.full(num_dose_points, quadratic_weight / num_dose_points)

                penalty_terms["linear"] += linear_weight_vec @ x_slack
                penalty_terms["quadratic"] += quadratic_weight_vec @ (x_slack * x_slack)
        # Set objective function
        model.setObjective(
            penalty_terms["linear"]
            + penalty_terms["quadratic"]
            + penalty_terms["uniformity"]
            + penalty_terms["hotspot"],
            GRB.MINIMIZE
        )
        model.update()

        if found:
            change_model_dose_to_target(
                new_target_dose=new_target_dose, 
                model=model, 
                coords_target_constraint=self.target_constraints_coords
            )
        return model
        

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

        self.model, outplan, self.solution_found, self.solve_time = _get_optimized_plan_from_model(
            plan=self.plan,
            model=self.model,
            inplace=inplace
            )
        return outplan

    def bound_dwell_time(
        self,
        name: str,
        lower_bound: float = None,
        upper_bound: float = None
        ) -> None:
        r"""
        See `BrachyDwellTime.bound_dwell_time` for details.
        """
        for variable in self.dwellTimeVariables:
            if variable.name == name:
                variable.set_bounds(lower_bound=lower_bound, upper_bound=upper_bound)
                break
        self.model.update()

    def evaluate_penaltyWeight(self, config: dict, return_plan:bool=False, inplace:bool=False) -> dict:
        r"""
        ### Purpose:
        - A function to evaluate penalty weights by resetting the model with new penalty weights
        and re-optimizing. The results are collected in a dictionary.
        ### Inputs:
        - config: dict := A dictionary containing the penalty weights for each structure.
            Example:
            {'linear_w_Skin': 96.60924792289734, 
            'linear_w_Chestwall': 90.46064639091492, 
            'linear_w_PTV': 1000.0, 
            'quadratic_w_PTV': 1.0}
        """
        if not len(config.keys()) > 0:
            raise ValueError("Config is empty. Please provide a valid config.")
        config_wo_td = deepcopy(config)
        # The method directly modifies the self.model object used later
        # in get_optimized_plan_from_model function
        _ = self.reset_model_from_config(self.model, config_wo_td)

        optimized_plan = self.get_optimized_plan_from_model(inplace=inplace)
        dvh_metrics = optimized_plan.get_dvh_metrics()
        output = {}
        for dvh_metric_name, dvh_value in dvh_metrics.items():
            output[dvh_metric_name] = float(dvh_value)

        output.update(config_wo_td)
        if return_plan:
            return output, optimized_plan
        else:
            return output

    def evaluate_penaltyWeight_space(self, list_of_configs: List[dict], return_plan:bool=False) -> dict:
        r"""

        # WARNING: For now, this multiprocessed function is not faster than looping through
        # the configs and calling evaluate_penaltyWeight function when returning the plans
        # because of the overhead of deepcopying the BrachyPlan object with every process

        ### Purpose:
        - A function to evaluate the penalty weight space by resetting the model with new penalty weights
        and re-optimizing in parallel. The results are collected in a dataframe.
        ### Inputs:
        - list_of_configs: List[dict] := A list of dictionaries containing the penalty weights
        for each structure.
            Example:
            [{'linear_w_Skin': 96.60924792289734, 
            'linear_w_Chestwall': 90.46064639091492, 
            'linear_w_PTV': 1000.0, 
            'quadratic_w_PTV': 1.0},
            {'linear_w_Skin': 50.0, 
            'linear_w_Chestwall': 50.0, 
            'linear_w_PTV': 500.0, 
            'quadratic_w_PTV': 0.5}]
        dose. Default is False.
        ### Outputs:
        - results: pd.DataFrame := A dataframe containing the DVH metrics and penalty weights
        for each configuration.
        """
        model_inputs_data = []
        for config in list_of_configs:
            if not len(config.keys()) > 0:
                raise ValueError("Config is empty. Please provide a valid config.")
            # Making a copy of the config because we mess it up in the reset_model_from_config function
            config_wo_td = deepcopy(config)
            ### Resetting the model with the new config cannot be done in parallel
            # since gurobi variables are not pickeable. So we do it in a loop
            # until we found a better solution
            model = self.reset_model_from_config(self.model, config_wo_td)
            ### Here we could write the model to an mps file and then pass the 
            # path as input to the worker function instead of the model itself
            # to avoid pickling the model. However, reading from file is slower
            # than passing the model data and reconstructing the model in the worker.
            model_data = get_model_data(model)
            model_inputs_data.append(model_data)

        ## Pickling the BrachyPlan is doable but it makes the Pool operation sequential. 
        # so we use a global variable _plan instead that we initialize with self.plan
        with Pool(min(10, os.cpu_count(), len(list_of_configs)), initializer=_init_worker, initargs=(self.plan,)) as pl:

            res = pl.starmap(_run_and_organize_results, zip(
                range(len(list_of_configs)),  # dummy arg instead of self.plan
                model_inputs_data,
                # If you wnt to return plans you need to pass the inplace arg as False
                # otherwise all your plans are the same object which will be the last
                # optimized plan
                [not return_plan]*len(list_of_configs),
                list_of_configs,
                [return_plan]*len(list_of_configs),
            )
            )
        if return_plan:
            weights_and_dvh_space = []
            optimized_plans = {}
            for i, r in enumerate(res):
                weights_and_dvh_space.append(r[0])
                optimized_plans[f"trial_{i}"] = r[1]
            return pd.DataFrame(weights_and_dvh_space), optimized_plans
        else:
            return pd.DataFrame(res)
        


def _run_and_organize_results(
    _, 
    model_inputs_data:dict,
    inplace:bool=False,
    config_wo_td:dict = {},
    return_plan:bool=False
):
    # print(f"PID {os.getpid()} starting work")
    # time.sleep(2)
    # print(f"PID {os.getpid()} done")
    # return None
    
    # use global _plan because pickling BrachyPlan prevents multiprocessing
    # and makes the pool execution sequential
    plan = deepcopy(_plan)

    with Env() as env, Model(env=env) as model:
        model = update_model_from_data(model_inputs_data, model)
        _, optimized_plan, _, _ = _get_optimized_plan_from_model(plan, model, inplace=inplace)

    dvh_metrics = optimized_plan.get_dvh_metrics()
    output = {}
    for dvh_metric_name, dvh_value in dvh_metrics.items():
        output[dvh_metric_name] = float(dvh_value)
    output.update(config_wo_td)

    if return_plan:
        return output, optimized_plan
    else:
        return output

def save_quadexpr(quadexpr):
    # Save QuadExpr parameters: constant, linear terms, quadratic terms
    saved = {}
    
    linear_expr = quadexpr.getLinExpr()
    c = linear_expr.getConstant()
    saved['constant'] = c
    saved_linear = []
    for i in range(linear_expr.size()):
        var = linear_expr.getVar(i)
        coeff = linear_expr.getCoeff(i)
        saved_linear.append((var.varName, var.VType, var.LB, var.UB, coeff))
    saved['linear_list'] = saved_linear
    
    saved_quad = []
    for i in range(quadexpr.size()):
        v1, v2 = quadexpr.getVar1(i), quadexpr.getVar2(i)
        coeff = quadexpr.getCoeff(i)
        saved_quad.append((v1.varName, v2.varName, coeff))
    saved['quadratic_list'] = saved_quad
    
    return saved

def load_quadexpr(saved, model):
    # Reconstruct QuadExpr from saved parameters given variable name mapping
    quad_expr = QuadExpr()
    quad_expr.addConstant(saved['constant'])
    
    # Add linear terms
    for vname, vtype, lb, ub, coeff in saved['linear_list']:
        var = model.getVarByName(vname)
        quad_expr.add(coeff * var)

    # Add quadratic terms
    for vname1, vname2, coeff in saved['quadratic_list']:
        quad_expr.add(coeff * model.getVarByName(vname1) * model.getVarByName(vname2))
    
    return quad_expr


def compare_gurobi_models(model1, model2):
    # Compare variables
    
    vars1 = {v.VarName: (v.VType, v.LB, v.UB, v.VarName) for v in model1.getVars()}
    vars2 = {v.VarName: (v.VType, v.LB, v.UB, v.VarName) for v in model2.getVars()}
    if vars1 != vars2:
        return False, "Variables differ"

    # Compare constraints
    constrs1 = {}
    for c in model1.getConstrs():
        expr = model1.getRow(c)
        coeffs = {expr.getVar(i).VarName: expr.getCoeff(i) for i in range(expr.size())}
        constrs1[c.ConstrName] = (c.Sense, c.RHS, coeffs)

    constrs2 = {}
    for c in model2.getConstrs():
        expr = model2.getRow(c)
        coeffs = {expr.getVar(i).VarName: expr.getCoeff(i) for i in range(expr.size())}
        constrs2[c.ConstrName] = (c.Sense, c.RHS, coeffs)

    if constrs1 != constrs2:
        return False, "Constraints differ"

    # Compare objectives
    obj1 = model1.getObjective()
    obj2 = model2.getObjective()
    if model1.ModelSense != model2.ModelSense:
        return False, "Objectives have different senses"
    if obj1.size() != obj2.size():
        return False, "Objectives have different number of terms"
    else:
        if type(obj1) != type(obj2):
            return False, f"Objectives are of different types, you have {type(obj1)} and {type(obj2)}"
        if isinstance(obj1, QuadExpr) and isinstance(obj2, QuadExpr):
            for i in range(obj1.size()):
                var1 = obj1.getVar1(i)
                var2 = obj2.getVar1(i)
                if var1.VarName != var2.VarName or obj1.getCoeff(i) != obj2.getCoeff(i):
                    return False, "Objective terms differ from first variable "
                var1 = obj1.getVar2(i)
                var2 = obj2.getVar2(i)
                if var1.VarName != var2.VarName or obj1.getCoeff(i) != obj2.getCoeff(i):
                    return False, "Objective terms differ from second variable "
        else:
            assert isinstance(obj1, LinExpr) and isinstance(obj2, LinExpr), "Objective is neither linear nor quadratic"
            for i in range(obj1.size()):
                var1 = obj1.getVar(i)
                var2 = obj2.getVar(i)
                if var1.VarName != var2.VarName or obj1.getCoeff(i) != obj2.getCoeff(i):
                    return False, "Objective terms differ from variable "
                
    for var1, var2 in tqdm.tqdm(zip(model1.getVars(), model2.getVars()), total=len(model1.getVars()), desc="Comparing objective coefficients"):
        for const1, const2 in zip(model1.getConstrs(), model2.getConstrs()):
            if model1.getCoeff(const1, var1) != model2.getCoeff(const2, var2):
                return False, "Objective coefficients differ"

    return True, "Models match"




def update_model_from_data(input_data: dict, model:Model):

    model.ModelSense = input_data['model_sense']
    
    model.addMVar((len(input_data['varnames']),), lb=input_data['lb'], ub=input_data['ub'],
                    obj=input_data['var_obj'], vtype=input_data['vtype'],
                    name=input_data['varnames'])
    # model.update()  # <-- This is critical
    model.addMConstr(A=input_data['A'], x=None, sense=input_data['con_sense'], b=input_data['rhs'],
                        name=input_data['connames'])
    model.update()  # <-- This is critical otherwise the load_quadexpr cannot access variables.

    objective = load_quadexpr(input_data["objective"], model) # {v.VarName: v for v in model.getVars()})
    model.setObjective(objective)
    model.update()
    return model

def get_model_data(model: Model):
    model_data = dict()
    model_data['name'] = model.ModelName
    model_data['A'] = model.getA()
    model_data['model_sense'] = model.ModelSense
    model_data['con_sense'] = np.array(model.getAttr("Sense"))
    model_data['rhs'] = np.array(model.getAttr("rhs"))
    model_data['lb'] = np.array(model.getAttr("LB"))
    model_data['ub'] = np.array(model.getAttr("UB"))
    model_data['vtype'] = np.array(model.getAttr("Vtype"))
    model_data['var_obj'] = np.array(model.getAttr("Obj"))
    model_data['varnames'] = model.getAttr("VarName")
    model_data['connames'] = model.getAttr("ConstrName")

    objective = model.getObjective()
    assert isinstance(objective, QuadExpr), "Objective is not a quadratic expression"
    model_data["objective"] = save_quadexpr(objective)

    return model_data

_plan = None  # global in worker processes

def _init_worker(plan):
    global _plan
    _plan = plan

if __name__ == "__main__":
    import gurobipy as gb
    env = Env(empty=True)
    env.start()
    model = gb.read("/app/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results_test_mobo_clean/Dataset007Dataset050/val_benchmark_fold_0/259984/manual_clinical_struct\
ures__clinical_dwellpos__auto_optimization/model_0.mps", env)
    model_data = get_model_data(model)
    print('model_data["lb"]', model_data["lb"])
    with Env() as env, Model(env=env) as model:
        model_remade = update_model_from_data(model_data, model)
        print(compare_gurobi_models(model, model_remade))
        exit()
   # https://support.gurobi.com/hc/en-us/community/posts/12678106466321-Fast-creation-of-Gurobi-models

