# from abc import ABC, abstractmethod
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
    process_variable, compute_dose_rate_matrices, Optimization_Config
)
import multiprocessing as mp
from functools import partial

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

def change_model_dose_to_target(new_target_dose:float, model:Model, coords_target_constraint:List[int], 
                                 coords_hotspot_constraint:List[int], hotspot_threshold:float):
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
    assert len(coords_target_constraint) > 0, "No target constraints found in the model. Cannot change target dose."
    assert coords_target_constraint[-1] == coords_target_constraint[0] + len(coords_target_constraint) - 1, (
        "Target constraints are not consecutive. Cannot change target dose constraints."
    )
    model.setAttr('RHS', constr_list[coords_target_constraint[0]:coords_target_constraint[-1]+1], new_target_dose)
    if len(coords_hotspot_constraint) >0:
        assert coords_hotspot_constraint[-1] == coords_hotspot_constraint[0] + len(coords_hotspot_constraint) - 1, (
            "Hotspot constraints are not consecutive. Cannot change hotspot dose constraints."
        )
        model.setAttr('RHS', constr_list[coords_hotspot_constraint[0]:coords_hotspot_constraint[-1]+1], hotspot_threshold*new_target_dose)
    
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
                    f"catheter_{catheter.index+1}_dwell_{dwell_position.index+1}"
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
        roi_margin_mm: List[float] | float = 5.0,
        multi_processing: bool = True
        ):
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
        self.hotspot_constraints_coords = []
        self.hotspot_threshold = None
        self.structure_weights_d = {}
        self.multi_processing = multi_processing
        # self._cached_A_matrix = np.ndarray([], dtype=object)
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
            model=self.model,
            multi_processing=self.multi_processing)

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

                dt_var_name = f"catheter_{catheter.index+1}_dwell_{dwell_position.index+1}"
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
        See `BrachyDwellTime.get_optimization_roi_bounds` for details.
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

        penalty_terms = {
        "linear": 0,
        "quadratic": 0,
        "hotspot": 0,
        "uniformity": 0
        }
        constraint_counter = 0
        
        for structure in plan.structure_list:
            if structure.optimization_config is None:
                continue
            if "hotspot_estimator:" in structure.name.lower():
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

            structure_mask = resample_crop_the_mask_or_contour_to_optimGrid(
                structure_mask=structure_mask,
                template_dose_obj=plan.combined_dose,
                optim_spacing=optim_spacing,
                )

            # Build dose rate matrix and dwell time vector for this structure
            dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
                dwellTimeVariables,
                plan,
                structure.name,
                structure_mask,
                optim_spacing,
                self.roi_bounds,
                max_workers=4, 
                shift_origin=True,
                multi_processing=multi_processing
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
            # Convert A to sparse matrix -> No need
            # A_sparse = sp.csr_matrix(A)
            A_sparse = A
            # Create target dose vector
            target_dose_vec = np.full(num_dose_points, target_dose)

            # Target volume constraints and penalties
            if structure.target_volume:
                if linear_weight > 0 or quadratic_weight > 0:
                    x_slack = model.addMVar(
                        shape=num_dose_points,
                        lb=0.0,
                        ub=target_dose - min_dose,
                        name=f"dose_slack_{structure.name}"
                        )
                    model.addConstr(
                        A_sparse @ t_MVar + x_slack >= target_dose_vec,
                        name=f"dose_target_{structure.name}"
                        )
                    self.target_constraints_coords.extend(list(range(constraint_counter, constraint_counter + num_dose_points)))
                    constraint_counter += num_dose_points

                # handle the linear penalty
                if linear_weight > 0:
                    linear_weight_vec = np.full(num_dose_points, linear_weight / num_dose_points)
                    penalty_terms["linear"] += linear_weight_vec @ x_slack

                if quadratic_weight > 0:
                # Create slack variables for uniformity
                    quadratic_weight_vec = np.full(num_dose_points, quadratic_weight / num_dose_points)
                    penalty_terms["quadratic"] += quadratic_weight_vec @ (x_slack * x_slack)

                if uniformity_weight > 0:
                    y_uniform = model.addMVar(
                        shape=num_dose_points,
                        lb=-GRB.INFINITY,
                        ub=target_dose - min_dose,
                        name=f"uniform_slack_{structure.name}"
                    )
                    # Uniformity constraints: A @ dwell_times + y_uniform == target_dose
                    model.addConstr(
                        A_sparse @ t_MVar + y_uniform == target_dose_vec,
                        name=f"dose_uniform_{structure.name}"
                    )
                    self.target_constraints_coords.extend(list(range(constraint_counter, constraint_counter + num_dose_points)))
                    constraint_counter += num_dose_points
                    uniformity_weight_vec = np.full(num_dose_points, uniformity_weight / (num_dose_points * 1000))
                    penalty_terms["uniformity"] += uniformity_weight_vec @ (y_uniform * y_uniform)

                ## Saving weights info for later potential resetting of the model
                self.structure_weights_d[structure.name] = {
                    "linear_weight": linear_weight,
                    "quadratic_weight": quadratic_weight,
                    "uniformity_weight": uniformity_weight,
                    "num_dose_points": num_dose_points,
                    "linear_coeff":linear_weight / num_dose_points,
                    "quadratic_coeff":quadratic_weight / num_dose_points,
                    "uniformity_coeff":uniformity_weight / (num_dose_points * 1000) # is quadratic weight
                }
                
                if hotspot_weight > 0 and hotspot_threshold > 0:
                    penalty_terms["hotspot"] = self.set_penalty_constraint_hotspot_estimators(
                        target_dose=target_dose,
                        hotspot_threshold = hotspot_threshold,
                        hotspot_weight = hotspot_weight,
                        plan = plan,
                        optim_spacing=optim_spacing,
                        roi_bounds = self.roi_bounds,
                        model = model,
                        dwellTimeVariables = dwellTimeVariables,
                        constraint_counter=constraint_counter
                    )

            # OAR (Organ at Risk) constraints and penalties
            else:
                if linear_weight > 0 or quadratic_weight > 0:
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
                    self.target_constraints_coords.extend(list(range(constraint_counter, constraint_counter + num_dose_points)))
                if linear_weight > 0:
                    constraint_counter += num_dose_points
                    linear_weight_vec = np.full(num_dose_points, linear_weight / num_dose_points)
                    penalty_terms["linear"] += linear_weight_vec @ x_slack

                if quadratic_weight > 0:
                    quadratic_weight_vec = np.full(num_dose_points, quadratic_weight / num_dose_points)
                    penalty_terms["quadratic"] += quadratic_weight_vec @ (x_slack * x_slack)
                    constraint_counter += num_dose_points

                ## Saving weights info for later potential resetting of the model
                self.structure_weights_d[structure.name] = {
                    "linear_weight": linear_weight,
                    "quadratic_weight": quadratic_weight,
                    "num_dose_points": num_dose_points,
                    "linear_coeff": linear_weight / num_dose_points,
                    "quadratic_coeff": quadratic_weight / num_dose_points,
                }

        # Set objective function
        model.setObjective(
            penalty_terms["linear"]
            + penalty_terms["quadratic"]
            + penalty_terms["uniformity"]
            + penalty_terms["hotspot"],
            GRB.MINIMIZE
        )
        model.update()

    def set_penalty_constraint_hotspot_estimators(
        self,
        target_dose: float,
        hotspot_threshold: float,
        hotspot_weight: float,
        plan: BrachyPlan,
        optim_spacing: List[float],
        roi_bounds: List[List[float]],
        model: Model,
        dwellTimeVariables: List[DwellTime_Gurobi],
        constraint_counter: int
        ) -> LinExpr:
        r"""
        ### Purpose:
        - Make penalty and constraints for the hotspot estimator structures.
        ### Inputs:
        - target_dose: float := The target dose for the structure.
        - hotspot_threshold: float := The hotspot threshold for the structure.
        - hotspot_weight: float := The hotspot weight for the structure.
        - plan: BrachyPlan := The brachytherapy plan.
        - optim_spacing: List[float] := The optimization spacing.
        - roi_bounds: List[List[float]] := The ROI bounds.
        - model: Model := The Gurobi model.
        - dwellTimeVariables: List[DwellTime_Gurobi] := The dwell time variables.
        - constraint_counter: int := The current constraint counter (used for resetting weights later).
        ### Outputs:
        - hotspot_penalty: LinExpr := The hotspot penalty term to be added to the objective function.
        """
        if self.hotspot_threshold is None:
            self.hotspot_threshold = hotspot_threshold
        else:
            assert np.isclose(self.hotspot_threshold, hotspot_threshold), (
                "All structures with hotspot estimator should have the same hotspot threshold."
                "Since only one structure (PTV or CTV) should have the hotspot estimator."
                f" Found {self.hotspot_threshold} and {hotspot_threshold}"
            )

        # resample hotspot masks and crop to roi bounds
        hotspot_masks = [
            structure.mask for structure in plan.structure_list
            if "hotspot_estimator" in structure.name
            ]
        with mp.Pool(processes=8) as pool:
            partial_func = partial(
                resample_crop_the_mask_or_contour_to_optimGrid,
                template_dose_obj=plan.combined_dose,
                optim_spacing=optim_spacing,
                roi_bounds=roi_bounds
            )
            processed_masks = tqdm.tqdm(list(
                    pool.imap(partial_func, hotspot_masks),
                    total=len(hotspot_masks),
                    desc="Resampling and cropping hotspot estimator masks"
                    )
                )
        # setup a general A matrix once for all the hotspot estimators at once.
        dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
            dwellTimeVariables=dwellTimeVariables,
            plan=plan,
            structure_name=None,
            structure_mask=None,
            optim_spacing=optim_spacing,
            roi_bounds=self.roi_bounds,
            max_workers=8, 
            shift_origin=True
        )
        t_MVar = MVar.fromlist(dwell_vars)
        A = np.column_stack(dose_rate_matrices)
        print("let's pause here for debugging hotspot estimator")            
        unmasked_dose = A @ t_MVar
        hotspot_penalty = 0
        for mask in processed_masks:
            mask_array = mask.imageArray.swapaxes(0, 2).flatten().astype(bool)
            num_dose_points = np.sum(mask_array)
            masked_dose = unmasked_dose[mask_array]
            # slack variable for hotspot estimator
            x_slack = model.addVar(
                    lb=0.0,
                    ub=hotspot_threshold * target_dose,
                    name=f"hotspot_slack_{mask.name.split(":")[-1]}"
                )
            # Hotspot estimator constraints
            model.addConstr(
                sum(masked_dose)/num_dose_points - x_slack <= (target_dose*hotspot_threshold),
            )
            hotspot_penalty += (hotspot_weight * x_slack)/num_dose_points
            
            self.hotspot_constraints_coords.extend(list(range(constraint_counter, constraint_counter + num_dose_points)))
            constraint_counter += num_dose_points

            ## Saving weights info for later potential resetting of the model
            self.structure_weights_d[mask.name] = {
                "hotspot_weight": hotspot_weight,
                "num_dose_points": num_dose_points,
                "hotspot_coeff": hotspot_weight / num_dose_points # is a linear coeff
            }

        return hotspot_penalty

    def reset_model_from_config(
        self,
        config_list:List[Optimization_Config] = None,
        config: Optional[dict] = None) -> None:
        r"""
        ### Purpose:
        - A function to reset the model with new penalty weights.
        ### Inputs:
        - model: Model := The Gurobi model to be reset.
        - 
        - config: dict := A dictionary containing the penalty weights for each structure.
            Example:
            {'td_PTV': 3.5358764542564374,
            'linear_w_Skin': 96.60924792289734, 
            'linear_w_Chestwall': 90.46064639091492, 
            'linear_w_PTV': 1000.0, 
            'quadratic_w_PTV': 1.0}

        ### Outputs:
        - model: Model := The reset Gurobi model.
        """
        if config_list is None and config is None:
            # create config dict from config_list
            # to match seb's previous implementation
            # TODO(sebers): factor out custom configs and use Optimization_Config everywhere
            config = defaultdict(float)
            for opt in config_list:
                config[f"linear_w_{opt.name}"] = opt.penalty_weight_linear
                config[f"quadratic_w_{opt.name}"] = opt.penalty_weight_quadratic
                if opt.name == "PTV" or opt.name == "CTV":
                    config[f"td_{opt.name}"] = opt.dose_voxel_goal

        _, updated_structure_weights_d = modify_model_objective_with_new_penalty_weights_and_td(
            model=self.model, 
            # self.structure_weights_d will be modified with the next set of penalty weights
            # It will store the current "state" of the model.
            og_setup_for_objective=self.structure_weights_d, 
            new_penalty_weights=config, 
            target_constraints_coords=self.target_constraints_coords, 
            hotspot_constraints_coords=self.hotspot_constraints_coords, 
            hotspot_threshold=self.hotspot_threshold)

        ### Updating state of the penalty weights of the model dict
        self.structure_weights_d = updated_structure_weights_d

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

    def evaluate_penaltyWeight(self, config: dict, return_cat_table:bool=False, inplace:bool=False) -> dict:
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
        - return_cat_table: bool := Whether to return the catheter table along with the DVH metrics. Default is False.
        - inplace: bool := Whether to modify the original plan or return a new optimized plan. Default is False.
        ### Outputs:
        - output: dict := A dictionary containing the DVH metrics and penalty weights.
        If return_cat_table is True, returns a tuple (output, catheter_table).
        WARNING: If inplace is True, the original plan will be modified.
        By returning the catheter table, one can reconstruct the optimized plan afterwards
        by manually setting the catheter_table attribute of the BrachyPlan object and then
        calling the update_plan_from_catheter_table() method.
        """
        if not len(config.keys()) > 0:
            raise ValueError("Config is empty. Please provide a valid config.")
        config_wo_td = deepcopy(config)
        # The method directly modifies the self.model object used later
        # in get_optimized_plan_from_model function
        self.reset_model_from_config(config_wo_td)

        optimized_plan = self.get_optimized_plan_from_model(inplace=inplace)
        dvh_metrics = optimized_plan.get_dvh_metrics()
        output = {}
        for dvh_metric_name, dvh_value in dvh_metrics.items():
            output[dvh_metric_name] = float(dvh_value)

        output.update(config)
        if return_cat_table:
            return output, deepcopy(optimized_plan.catheter_table)
        else:
            return output

    def evaluate_penaltyWeight_space(self, list_of_configs: List[dict], return_cat_table:bool=False) -> dict:
        r"""
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
       
        model_data = get_model_data(self.model)

        ## Pickling the BrachyPlan is doable but it makes the Pool operation sequential. 
        # so we use a global variable _plan instead that we initialize with self.plan
        with Pool(min(10, os.cpu_count(), len(list_of_configs)), initializer=_init_worker, initargs=(self.plan,)) as pl:

            res = pl.starmap(_run_and_organize_results, zip(
                range(len(list_of_configs)),  # dummy arg instead of self.plan
                [model_data] * len(list_of_configs),
                # If you want to return plans you need to pass the inplace arg as False
                # otherwise all your plans are the same object which will be the last
                # optimized plan. However, deepcopying the plan makes the Pool operation
                # sequential and slow. Instead one can return only the catheter table
                [True]*len(list_of_configs),
                list_of_configs,
                [return_cat_table]*len(list_of_configs),
                [deepcopy(self.structure_weights_d)]*len(list_of_configs), 
                [self.target_constraints_coords]*len(list_of_configs),
                [self.hotspot_constraints_coords]*len(list_of_configs),
                [self.hotspot_threshold]*len(list_of_configs)
            )
            )
        if return_cat_table:
            weights_and_dvh_space = []
            optimized_cat_table = {}
            for i, r in enumerate(res):
                weights_and_dvh_space.append(r[0])
                optimized_cat_table[f"trial_{i}"] = r[1]
            return pd.DataFrame(weights_and_dvh_space), optimized_cat_table
        else:
            return pd.DataFrame(res)
        
def _format_path(p, k):
    return f"{p}.{k}" if p else str(k)

def deep_diff(d1, d2, path=""):
    r"""
    Recursively compare two objects (dicts, lists, tuples, numpy arrays, sparse matrices, scalars).
    Returns a list of differences as strings. Useful for debugging model serialization.
    Was mostly useful for debugging the get_model_data and update_model_from_data functions.
    ### Inputs:
    - d1: dict := The first dictionary to compare, obtained from get_model_data function
    - d2: dict := The second dictionary to compare, obtained from get_model_data function
    ### Outputs:
    None - prints differences to console.
    """
    diffs = []

    # Compare dicts
    if isinstance(d1, dict) and isinstance(d2, dict):
        keys1 = set(d1.keys())
        keys2 = set(d2.keys())
        for k in keys1 - keys2:
            diffs.append(f"Key '{_format_path(path, k)}' missing in second dict")
        for k in keys2 - keys1:
            diffs.append(f"Key '{_format_path(path, k)}' missing in first dict")
        for k in keys1 & keys2:
            diffs.extend(deep_diff(d1[k], d2[k], _format_path(path, k)))
        return diffs

    # Compare lists/tuples
    if isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):
        if len(d1) != len(d2):
            diffs.append(f"Length mismatch at '{path}': {len(d1)} vs {len(d2)}")
        else:
            for i, (v1, v2) in enumerate(zip(d1, d2)):
                diffs.extend(deep_diff(v1, v2, f"{path}[{i}]"))
        return diffs

    # Compare numpy arrays
    if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
        if d1.shape != d2.shape:
            diffs.append(f"Shape mismatch at '{path}': {d1.shape} vs {d2.shape}")
        elif np.issubdtype(d1.dtype, np.number) and np.issubdtype(d2.dtype, np.number):
            if not np.allclose(d1, d2, equal_nan=True):
                idx = np.where(~np.isclose(d1, d2, equal_nan=True))
                for i in zip(*idx):
                    diffs.append(f"Value mismatch at '{path}{i}': {d1[i]} vs {d2[i]}")
        else:
            # fallback for non-numeric arrays (e.g., string arrays)
            if not np.array_equal(d1, d2):
                idx = np.where(d1 != d2)
                for i in zip(*idx):
                    diffs.append(f"Value mismatch at '{path}{i}': {d1[i]} vs {d2[i]}")
        return diffs

    # Compare sparse matrices
    if isinstance(d1, csr_matrix) and isinstance(d2, csr_matrix):
        if d1.shape != d2.shape:
            diffs.append(f"Sparse shape mismatch at '{path}': {d1.shape} vs {d2.shape}")
        elif (d1 != d2).nnz != 0:
            diffs.append(f"Sparse value mismatch at '{path}'")
        return diffs

    # Compare floats
    if isinstance(d1, float) and isinstance(d2, float):
        if not np.isclose(d1, d2, equal_nan=True):
            diffs.append(f"Float mismatch at '{path}': {d1} vs {d2}")
        return diffs

    # Compare everything else
    if d1 != d2:
        diffs.append(f"Value mismatch at '{path}': {d1} vs {d2}")
    return diffs


def _run_and_organize_results(
    _, 
    model_inputs_data:dict,
    inplace:bool=True,
    config_wo_td:dict = {},
    return_cat_table:bool=False,
    og_setup_for_objective:dict = {},
    target_constraints_coords: List[int] = [],
    hotspot_constraints_coords: List[int] = [],
    hotspot_threshold: float = 1.2

):
    # print(f"PID {os.getpid()} starting work")
    # time.sleep(2)
    # print(f"PID {os.getpid()} done")
    # return None
    
    # use global _plan because pickling BrachyPlan prevents multiprocessing
    # and makes the pool execution sequential
    plan = deepcopy(_plan)
    new_weight_config = deepcopy(config_wo_td)
    with Env() as env, Model(env=env) as model:
        model = update_model_from_data(model_inputs_data, model)       
        _ = modify_model_objective_with_new_penalty_weights_and_td(
            model=model, 
            og_setup_for_objective=og_setup_for_objective, 
            new_penalty_weights=new_weight_config, 
            target_constraints_coords=target_constraints_coords, 
            hotspot_constraints_coords=hotspot_constraints_coords, 
            hotspot_threshold=hotspot_threshold)
        _, optimized_plan, _, _ = _get_optimized_plan_from_model(plan, model, inplace=inplace)

    dvh_metrics = optimized_plan.get_dvh_metrics()
    output = {}
    for dvh_metric_name, dvh_value in dvh_metrics.items():
        output[dvh_metric_name] = float(dvh_value)
    output.update(config_wo_td)

    if return_cat_table:
        return output, deepcopy(optimized_plan.catheter_table)
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
        saved_quad.append(deepcopy((v1.varName, v2.varName, coeff)))
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


def modify_model_objective_with_new_penalty_weights_and_td(
        model: Model, 
        og_setup_for_objective: dict, 
        new_penalty_weights: dict,
        target_constraints_coords: List[int] = [],
        hotspot_constraints_coords: List[int] = [],
        hotspot_threshold: float = 1.2
          # , inplace: bool = True
    ) -> Model:
    """

    A function to modify the objective function of a Gurobi model by updating the penalty weights
    for different structures based on a provided configuration. The function identifies the variables
    associated with each structure and adjusts their coefficients in the objective function accordingly.
    Modifies the model inplace. 
    ### Inputs:
    - model: Model := The Gurobi optimization model whose objective function is to be modified.
    - og_setup_for_objective: dict := A dictionary containing the original setup for each structure,
    - including weights and coefficients.
    - new_penalty_weights: dict := A dictionary containing the new penalty weights for each structure.
    og_setup is something like:
    {
    "PTV": {
        "linear_weight": 1000.0,
        "quadratic_weight": 1.0,
        "uniformity_weight": 1.0,
        "num_dose_points": 1788,
        "coeff": 0.5592841163310962
    },
    "Skin": {
        "linear_weight": 100.0,
        "quadratic_weight": 1.0,
        "num_dose_points": 5620,
        "coeff": 0.017793594306049824
    },
    "Chestwall": {
        "linear_weight": 100.0,
        "quadratic_weight": 1.0,
        "num_dose_points": 6710,
        "coeff": 0.014903129657228018
    },
    "hotspot_estimator:catheter_0_dwell_2/catheter_0_dwell_3": {
        "hotspot_weight": 1.0,
        "num_dose_points": 12,
        "coeff": 0.08333333333333333
    },
    ....
    }

    new_penalty_weights is something like:

    {'td_PTV': 3.5358764542564374, 
     'linear_w_PTV': 1000.0, 
     'quadratic_w_PTV': 1.0, 
     'linear_w_Skin': 351.9170277168017, 
     'linear_w_Chestwall': 7.679852357735809}
     which we reorganize into 
    {"PTV": {"linear":500, "quadratic":0.5}, "Skin": {"linear":50, "quadratic":0.5}, "Chestwall": {"linear":50, "quadratic":0.5}}, 
    """


    new_setup_for_model_penalty_weights = {}

    # Extract target dose from new_penalty_weights if present and remove it from the dict
    new_target_dose = None
    found = False

    penalty_weights_to_set = deepcopy(new_penalty_weights)
    for k, v in penalty_weights_to_set.items():
        if "td_" in k.lower() :
            new_target_dose = deepcopy(v)
            to_remove = k
            found = True
            break
    if found:
        penalty_weights_to_set.pop(to_remove, None)

    ## Reorganize dict by structure name because input config used in MOBO is flat
    reorganized_new_penalty_weights = {}
    for k, v in penalty_weights_to_set.items():
        struct_name = k.split("_")[2]
        if struct_name not in reorganized_new_penalty_weights:
            reorganized_new_penalty_weights[struct_name] = {}
        param_name = k.split("_")[0] 
        reorganized_new_penalty_weights[struct_name][param_name] = v

    old_objective = model.getObjective()
    assert isinstance(old_objective, QuadExpr), "Objective is not a quadratic expression as expected"
    new_objective = QuadExpr()
    # Constant does not change so we do not need to call model.getConstant()

    old_linear_expr = old_objective.getLinExpr()
   
    for i in range(old_linear_expr.size()):
        var = old_linear_expr.getVar(i)
        coeff = old_linear_expr.getCoeff(i)

        # Determine which structure this variable belongs to
        struct_name = None
        for sname in og_setup_for_objective.keys():
            if coeff == og_setup_for_objective[sname].get("linear_coeff", None) or \
                coeff == og_setup_for_objective[sname].get("hotspot_coeff", None):
                struct_name = sname

        if struct_name is None:
            raise ValueError(f"Variable {var.VarName} does not belong to any known structure, with og_setup {og_setup_for_objective}, and coeff {coeff}, and new config {new_penalty_weights}")
        # Get the original coefficient and weight
        if not "hotspot" in struct_name:
            og_weight = og_setup_for_objective[struct_name]["linear_weight"]
            if struct_name in reorganized_new_penalty_weights:
                if "linear" in reorganized_new_penalty_weights[struct_name]:
                    new_weight = reorganized_new_penalty_weights[struct_name]["linear"]
                else:
                    new_weight = og_weight
            else:
                new_weight = og_weight
        else:
            og_weight = og_setup_for_objective[struct_name]["hotspot_weight"]
            if struct_name in reorganized_new_penalty_weights:
                if "hotspot" in reorganized_new_penalty_weights[struct_name]:
                    new_weight = reorganized_new_penalty_weights[struct_name]["hotspot"]
                else:
                    new_weight = og_weight
            else:
                new_weight = og_weight
     
        # Calculate new coefficient
        new_coeff = (new_weight / og_weight) * coeff
        # Store in new setup for model penalty weights
        if not struct_name in new_setup_for_model_penalty_weights:
            new_setup_for_model_penalty_weights[struct_name] = {}
            if not "hotspot" in struct_name:
                new_setup_for_model_penalty_weights[struct_name]["linear_weight"] = new_weight
                new_setup_for_model_penalty_weights[struct_name]["linear_coeff"] = new_coeff
            else:
                new_setup_for_model_penalty_weights[struct_name]["hotspot_weight"] = new_weight
                new_setup_for_model_penalty_weights[struct_name]["hotspot_coeff"] = new_coeff
            new_setup_for_model_penalty_weights[struct_name]["num_dose_points"] = og_setup_for_objective[struct_name]["num_dose_points"]

        # Add to new objective if
        if new_coeff != 0:
            new_objective.addTerms(new_coeff, var)

    coeff_type = None
    for i in range(old_objective.size()):
        v1, v2 = old_objective.getVar1(i), old_objective.getVar2(i)
        coeff = old_objective.getCoeff(i)
        # Determine which structure this variable belongs to
        struct_name = None
        for sname in og_setup_for_objective.keys():
            if coeff == og_setup_for_objective[sname].get("quadratic_coeff", None) or \
                coeff == og_setup_for_objective[sname].get("uniformity_coeff", None):
                struct_name = sname
                if coeff == og_setup_for_objective[sname].get("quadratic_coeff", None):
                    coeff_type = "quadratic"
                else:
                    coeff_type = "uniformity"

        if struct_name is None:
            raise ValueError(f"Variable pair {v1.VarName}, {v2.VarName} does not belong to any known structure, with og_setup {og_setup_for_objective}, and coeff {coeff}, and new config {new_penalty_weights}")
        
        # Get the original coefficient and weight
        og_weight = og_setup_for_objective[struct_name][f"{coeff_type}_weight"]
        if struct_name in reorganized_new_penalty_weights:
            if coeff_type in reorganized_new_penalty_weights[struct_name]:
                new_weight = reorganized_new_penalty_weights[struct_name][coeff_type]
            else:
                new_weight = og_weight
        else:
            new_weight = og_weight
      
                
        # Calculate new coefficient
        new_coeff = (new_weight / og_weight) * coeff
        # Store in new setup for model penalty weights
        if coeff_type == "quadratic":
            if not "quadratic_weight" in new_setup_for_model_penalty_weights[struct_name]:
                new_setup_for_model_penalty_weights[struct_name]["quadratic_weight"] = new_weight
                new_setup_for_model_penalty_weights[struct_name]["quadratic_coeff"] = new_coeff
        else:
            if not "uniformity_weight" in new_setup_for_model_penalty_weights[struct_name]:
                new_setup_for_model_penalty_weights[struct_name]["uniformity_weight"] = new_weight
                new_setup_for_model_penalty_weights[struct_name]["uniformity_coeff"] = new_coeff
            
        if new_coeff != 0:
            new_objective.addTerms(new_coeff, v1, v2)

   
    model.setObjective(new_objective)
    model.update()
    if found:
        change_model_dose_to_target(
            new_target_dose=new_target_dose, 
            model=model, 
            coords_target_constraint=target_constraints_coords, 
            coords_hotspot_constraint=hotspot_constraints_coords, 
            hotspot_threshold=hotspot_threshold
        )
    
    return model, new_setup_for_model_penalty_weights

def update_model_from_data(input_data: dict, model:Model):

    model.ModelSense = input_data['model_sense']
    
    model.addMVar((len(input_data['varnames']),), lb=input_data['lb'], ub=input_data['ub'],
                    obj=input_data['var_obj'], vtype=input_data['vtype'],
                    name=input_data['varnames'])
    model.addMConstr(A=input_data['A'], x=None, sense=input_data['con_sense'], b=input_data['rhs'],
                        name=input_data['connames'])
    model.update()  # <-- This is critical otherwise the load_quadexpr cannot access variables.

    objective = load_quadexpr(input_data["objective"], model) # {v.VarName: v for v in model.getVars()})
    model.setObjective(objective)
    model.update()
    return model

def get_model_data(model: Model):
    model_data = dict()
    model_data['name'] = deepcopy(model.ModelName)
    model_data['A'] = deepcopy(model.getA())
    model_data['model_sense'] = deepcopy(model.ModelSense)
    model_data['con_sense'] = deepcopy(np.array(model.getAttr("Sense")))
    model_data['rhs'] = deepcopy(np.array(model.getAttr("rhs")))
    model_data['lb'] = deepcopy(np.array(model.getAttr("LB")))
    model_data['ub'] = deepcopy(np.array(model.getAttr("UB")))
    model_data['vtype'] = deepcopy(np.array(model.getAttr("Vtype")))
    model_data['var_obj'] = deepcopy(np.array(model.getAttr("Obj")))
    model_data['varnames'] = deepcopy(model.getAttr("VarName"))
    model_data['connames'] = deepcopy(model.getAttr("ConstrName"))

    objective = model.getObjective()
    assert isinstance(objective, QuadExpr), "Objective is not a quadratic expression"
    model_data["objective"] = save_quadexpr(objective)

    return model_data

_plan = None  # global in worker processes

def _init_worker(plan):
    global _plan
    _plan = plan
