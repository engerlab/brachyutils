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
    BrachyDwellTimeOptim, BrachyDwellTime, 
    process_variable, compute_dose_rate_matrices, Optimization_Config
)

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
        self.multi_processing = multi_processing
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
            structure_mask = structure.mask
            optim_spacing = structure.optimization_config.spacing_mm
            target_dose = structure.optimization_config.dose_voxel_goal
            linear_weight = structure.optimization_config.penalty_weight_linear
            quadratic_weight = structure.optimization_config.penalty_weight_quadratic
            uniformity_weight = structure.optimization_config.penalty_weight_uniformity
            min_dose = structure.optimization_config.min_dose
            structure_max_dose = structure.optimization_config.max_dose
            hotspot_threshold = structure.optimization_config.hotspot_threshold

            assert structure.name == structure.optimization_config.structure_name, (
                "Structure name does not match optimization config structure name."
                f"you have {structure.name} and {structure.optimization_config.structure_name}"
            )
            
            if hotspot_threshold != 0.:
                if self.hotspot_threshold is None:
                    self.hotspot_threshold = hotspot_threshold
                else:
                    assert np.isclose(self.hotspot_threshold, hotspot_threshold), (
                        "All structures with hotspot estimator should have the same hotspot threshold."
                        "Since only one structure (PTV or CTV) should have the hotspot estimator."
                        f" Found {self.hotspot_threshold} and {hotspot_threshold}"
                    )

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

            structure.optimization_config.num_dose_points = num_dose_points

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
                    linear_weight_vec = np.full(num_dose_points, structure.optimization_config.linear_coeff)
                    penalty_terms["linear"] += linear_weight_vec @ x_slack

                if quadratic_weight > 0:
                # Create slack variables for uniformity
                    quadratic_weight_vec = np.full(num_dose_points, structure.optimization_config.quadratic_coeff)
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
                    uniformity_weight_vec = np.full(num_dose_points, structure.optimization_config.uniformity_coeff)
                    penalty_terms["uniformity"] += uniformity_weight_vec @ (y_uniform * y_uniform)

            # OAR (Organ at Risk) constraints and penalties
            # or hot spot volume
            else:
                if ("hotspot_estimator:" in structure.name.lower()
                    and hotspot_weight > 0):
                    x_slack = model.addVar(
                    # slack variable for hotspot estimator
                        # shape=num_dose_points,
                        lb=0.0,
                        ub=hotspot_threshold * target_dose - min_dose,
                        name=f"hotspot_slack_{structure.name.split(":")[-1]}"
                    )
                    # Hotspot estimator constraints
                    model.addConstr(
                        sum(A_sparse @ t_MVar)/num_dose_points - x_slack <= (target_dose*hotspot_threshold),
                    )
        
                    self.hotspot_constraints_coords.extend(list(range(constraint_counter, constraint_counter + 1)))
                    constraint_counter += 1

                    penalty_terms["hotspot"] += x_slack * (structure.optimization_config.hotspot_coeff)
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
                        constraint_counter += num_dose_points
                    if linear_weight > 0:
                        linear_weight_vec = np.full(num_dose_points, structure.optimization_config.linear_coeff)
                        penalty_terms["linear"] += linear_weight_vec @ x_slack

                    if quadratic_weight > 0:
                        quadratic_weight_vec = np.full(num_dose_points, structure.optimization_config.quadratic_coeff)
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


    def reset_model_from_config(
        self,
        new_config_list:List[Optimization_Config] = None,
        new_target_dose:float = None) -> None:
        r"""
        ### Purpose:
        - A function to reset the model with new penalty weights.
        ### Inputs:
        - new_config_list: List[Optimization_Config] := A list of new optimization configurations for each structure.
        - new_target_dose: float := The new target dose for the target structures.
        ### Outputs:
        - model: Model := The reset Gurobi model.
        """
        _ = modify_model_objective_with_new_penalty_weights_and_td(
            model=self.model, 
            # self.plan.structure_list optimization configs will be modified with the next set of penalty weights
            # It will store the current "state" of the model.
            og_optim_config_list=[x.optimization_config for x in self.plan.structure_list if x.optimization_config is not None],
            new_optim_config_list=new_config_list, 
            new_target_dose=new_target_dose,
            target_constraints_coords=self.target_constraints_coords, 
            hotspot_constraints_coords=self.hotspot_constraints_coords, 
            hotspot_threshold=self.hotspot_threshold)


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

    def evaluate_penaltyWeight(
            self, 
            optim_config_list: List[Optimization_Config], 
            target_dose: float,
            return_cat_table:bool=False, 
            inplace:bool=False) -> dict:
        r"""
        ### Purpose:
        - A function to evaluate penalty weights by resetting the model with new penalty weights
        and re-optimizing. The results are collected in a dictionary.
        ### Inputs:
        - optim_config_list: List[Optimization_Config] := A list of optimization configurations
        for each structure.
        - target_dose: float := The new target dose.
        - return_cat_table: bool := Whether to return the catheter table along with the DVH metrics. Default is False.
        - inplace: bool := Whether to modify the original plan or return a new optimized plan. Default is False.
        The Optimization_Config objects of the structure list will still be modified!!!
        ### Outputs:
        - output: dict := A dictionary containing the DVH metrics and penalty weights.
        If return_cat_table is True, returns a tuple (output, catheter_table).
        WARNING: If inplace is True, the original plan will be modified.
        By returning the catheter table, one can reconstruct the optimized plan afterwards
        by manually setting the catheter_table attribute of the BrachyPlan object and then
        calling the update_plan_from_catheter_table() method.
        """
        struct_name_existing = [x.name for x in self.plan.structure_list if x.optimization_config is not None]
        new_struct_names = [x.structure_name for x in optim_config_list]
        assert len(new_struct_names) == len(struct_name_existing), (
            "Length of optim_config_list does not match number of structures with optimization config in the plan." \
            f" Got configs for {optim_config_list} VS existing: {struct_name_existing}."
        )
        
        # The method directly modifies the self.model object used later
        # in get_optimized_plan_from_model function
        self.reset_model_from_config(new_config_list=optim_config_list, new_target_dose=target_dose)

        optimized_plan = self.get_optimized_plan_from_model(inplace=inplace)
        dvh_metrics = optimized_plan.get_dvh_metrics()
        output = {}
        for dvh_metric_name, dvh_value in dvh_metrics.items():
            output[dvh_metric_name] = float(dvh_value)

        for opt_conf in optim_config_list:
            s_name = opt_conf.structure_name
            for attr_name, attr_value in opt_conf.to_dict().items():
                if attr_name != "structure_name":
                    output[f"{attr_name}_{s_name}"] = attr_value

        if return_cat_table:
            return output, deepcopy(optimized_plan.catheter_table)
        else:
            return output

    def evaluate_penaltyWeight_space(
            self, 
            list_of_opt_config_lists: List[List[Optimization_Config]], 
            list_of_target_doses: List[float] = None,
            return_cat_table: bool = False) -> dict:
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

        list_original_opt_config_lists = []
        for _ in list_of_opt_config_lists:
            list_original_opt_config_lists.append(
                deepcopy(
                    [x.optimization_config for x in self.plan.structure_list if x.optimization_config is not None]
                )
            )

        ## Pickling the BrachyPlan is doable but it makes the Pool operation sequential. 
        # so we use a global variable _plan instead that we initialize with self.plan
        with Pool(min(10, os.cpu_count(), len(list_of_opt_config_lists)), initializer=_init_worker, initargs=(self.plan,)) as pl:
        
            res = pl.starmap(_run_and_organize_results, zip(
                range(len(list_of_opt_config_lists)),  # dummy arg instead of self.plan
                [model_data] * len(list_of_opt_config_lists),
                # If you want to return plans you need to pass the inplace arg as False
                # otherwise all your plans are the same object which will be the last
                # optimized plan. However, deepcopying the plan makes the Pool operation
                # sequential and slow. Instead one can return only the catheter table
                [True]*len(list_of_opt_config_lists),
                [return_cat_table]*len(list_of_opt_config_lists),
                # original opt config list
                list_original_opt_config_lists,
                # new opt config list
                list_of_opt_config_lists,
                # new target doses
                list_of_target_doses,
                [self.target_constraints_coords]*len(list_of_opt_config_lists),
                [self.hotspot_constraints_coords]*len(list_of_opt_config_lists),
                [self.hotspot_threshold]*len(list_of_opt_config_lists)
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
    return_cat_table:bool=False,
    og_optim_config_list:List[Optimization_Config] = [],
    new_config_list:List[Optimization_Config] = [],
    new_target_dose:float = None,
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
    with Env() as env, Model(env=env) as model:
        model = update_model_from_data(model_inputs_data, model)       
        _ = modify_model_objective_with_new_penalty_weights_and_td(
            model=model, 
            og_optim_config_list=og_optim_config_list, 
            new_optim_config_list=new_config_list, 
            new_target_dose=new_target_dose,
            target_constraints_coords=target_constraints_coords, 
            hotspot_constraints_coords=hotspot_constraints_coords, 
            hotspot_threshold=hotspot_threshold)
        _, optimized_plan, _, _ = _get_optimized_plan_from_model(plan, model, inplace=inplace)

    dvh_metrics = optimized_plan.get_dvh_metrics()
    output = {}
    for dvh_metric_name, dvh_value in dvh_metrics.items():
        output[dvh_metric_name] = float(dvh_value)
    for opt_conf in new_config_list:
        s_name = opt_conf.structure_name
        for attr_name, attr_value in opt_conf.to_dict().items():
            if attr_name != "structure_name":
                output[f"{attr_name}_{s_name}"] = attr_value

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
        og_optim_config_list: List[Optimization_Config],
        new_optim_config_list: dict,
        new_target_dose: float = None,
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
    - og_optim_config_list: List[Optimization_Config] := The original list of optimization configurations for each structure.
    - new_optim_config_list: dict := A list of new optimization configurations for each structure.
    - new_target_dose: float := The new target dose for the target structures. If None, the original target dose is retained.
    - target_constraints_coords: List[int] := The list of constraint indices corresponding to target dose constraints.
    - hotspot_constraints_coords: List[int] := The list of constraint indices corresponding to hotspot constraints.
    - hotspot_threshold: float := The hotspot threshold used in the optimization.
    ### Outputs:
    - model: Model := The modified Gurobi optimization model with updated objective function.
    """



    # Extract target dose from new_penalty_weights if present and remove it from the dict

    optim_config_list_to_set = deepcopy(new_optim_config_list)
    if not(new_target_dose is None):
        constr_list = list(model.getConstrs())
        target_constr = model.getAttr(
            'RHS', constr_list[target_constraints_coords[0]:target_constraints_coords[-1]+1])
        assert len(set(target_constr)) == 1, "Target dose constraints have varying RHS, cannot determine unique target dose"
        og_target_dose = target_constr[0]
        if og_target_dose == new_target_dose:
            warnings.warn(f" When modifying your model, the new target dose set: {new_target_dose}Gy"
                          f" is the same as the original target dose.")

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
        opt_conf_of_interest = None
        for opt_conf in og_optim_config_list:
            if np.isclose(coeff, opt_conf.linear_coeff, atol=1e-6) or \
                np.isclose(coeff, opt_conf.hotspot_coeff, atol=1e-6):
                struct_name = opt_conf.structure_name
                opt_conf_of_interest = opt_conf
        for new_opt_conf in optim_config_list_to_set:
            if new_opt_conf.structure_name == struct_name:
                new_opt_conf_of_interest = new_opt_conf
                assert new_opt_conf_of_interest.num_dose_points == opt_conf_of_interest.num_dose_points, (
                    f"Number of dose points for structure {struct_name} has changed from "
                    f"{opt_conf_of_interest.num_dose_points} to {new_opt_conf_of_interest.num_dose_points}. "
                    "This is not allowed when modifying penalty weights."
                )
                break

        if struct_name is None:
            raise ValueError(f"Variable {var.VarName} does not belong to any known structure, with og_conf_list {og_optim_config_list}, and coeff {coeff}, and new opt config list {new_optim_config_list}")
        # Get the original coefficient and weight
        if not "hotspot" in struct_name:
            new_coeff = new_opt_conf_of_interest.linear_coeff
        else:
            new_coeff = new_opt_conf_of_interest.hotspot_coeff

        # Add to new objective if
        if new_coeff != 0:
            new_objective.addTerms(new_coeff, var)

    coeff_type = None
    for i in range(old_objective.size()):
        v1, v2 = old_objective.getVar1(i), old_objective.getVar2(i)
        coeff = old_objective.getCoeff(i)
        # Determine which structure this variable belongs to
        struct_name = None
        opt_conf_of_interest = None
        for opt_conf in og_optim_config_list:
            if np.isclose(coeff, opt_conf.quadratic_coeff, atol=1e-6):
                struct_name = opt_conf.structure_name
                opt_conf_of_interest = opt_conf
                coeff_type = "quadratic"
            elif np.isclose(coeff, opt_conf.uniformity_coeff, atol=1e-6):
                struct_name = opt_conf.structure_name
                opt_conf_of_interest = opt_conf
                coeff_type = "uniformity"
        for new_opt_conf in optim_config_list_to_set:
            if new_opt_conf.structure_name == struct_name:
                new_opt_conf_of_interest = new_opt_conf
                assert new_opt_conf_of_interest.num_dose_points == opt_conf_of_interest.num_dose_points, (
                    f"Number of dose points for structure {struct_name} has changed from "
                    f"{opt_conf_of_interest.num_dose_points} to {new_opt_conf_of_interest.num_dose_points}. "
                    "This is not allowed when modifying penalty weights."
                )
                if coeff_type == "quadratic":
                    new_coeff = new_opt_conf_of_interest.quadratic_coeff
                else:
                    new_coeff = new_opt_conf_of_interest.uniformity_coeff
                break

        if struct_name is None:
            raise ValueError(f"Variable pair {v1.VarName}, {v2.VarName} does not belong to any known structure, with og_conf_list {og_optim_config_list}, and coeff {coeff}, and new opt config list {new_optim_config_list}")
      
        if new_coeff != 0:
            new_objective.addTerms(new_coeff, v1, v2)


    # Updating conf list at the end so that we have the correct weights when updating linear and hotspot terms
    # This time the reference is the new_objective: we use the new opt conf list to update the old one
    # We cannot directly touch the original opt conf list in the loop above because we need to update each
    # dose voxel constraint coefficient first. If modiying the weight after first voxel is changed then 
    # mapping with the coeff will be lost for the next voxel.
    new_linear_expr = new_objective.getLinExpr()
    for i in range(new_linear_expr.size()):
        coeff = new_linear_expr.getCoeff(i)
        # Determine which structure this variable belongs to
        struct_name = None
        opt_conf_of_interest = None
        # This time the reference is the updated opt conf list 
        for new_opt_conf in optim_config_list_to_set:
            if np.isclose(coeff, new_opt_conf.linear_coeff, atol=1e-6) or \
               np.isclose(coeff, new_opt_conf.hotspot_coeff, atol=1e-6):
                struct_name = new_opt_conf.structure_name
                new_opt_conf_of_interest = new_opt_conf
        for og_opt_conf in og_optim_config_list:
            if og_opt_conf.structure_name == struct_name:
                og_opt_conf_of_interest = og_opt_conf
                break
        # Get the original coefficient and weight
        if not "hotspot" in struct_name:
            new_weight = new_opt_conf_of_interest.penalty_weight_linear
            og_opt_conf_of_interest.penalty_weight_linear = new_weight
        else:
            new_weight = new_opt_conf_of_interest.penalty_weight_hotspot
            og_opt_conf_of_interest.penalty_weight_hotspot = new_weight

    # Updating conf list at the end so that we have the correct weights when updating quadratic terms
    for i in range(new_objective.size()):
        coeff = new_objective.getCoeff(i)
        # Determine which structure this variable belongs to
        struct_name = None
        opt_conf_of_interest = None
        for new_opt_conf in optim_config_list_to_set:
            if np.isclose(coeff, new_opt_conf.quadratic_coeff, atol=1e-6):
                struct_name = new_opt_conf.structure_name
                new_opt_conf_of_interest = new_opt_conf
                coeff_type = "quadratic"
            elif np.isclose(coeff, new_opt_conf.uniformity_coeff, atol=1e-6):
                struct_name = new_opt_conf.structure_name
                new_opt_conf_of_interest = new_opt_conf
                coeff_type = "uniformity"
        for og_opt_conf in og_optim_config_list:
            if og_opt_conf.structure_name == struct_name:
                og_opt_conf_of_interest = og_opt_conf
                if coeff_type == "quadratic":
                    new_weight = new_opt_conf_of_interest.penalty_weight_quadratic
                    og_opt_conf_of_interest.penalty_weight_quadratic = new_weight
                else:
                    new_weight = new_opt_conf_of_interest.penalty_weight_uniformity
                    og_opt_conf_of_interest.penalty_weight_uniformity = new_weight
                break

   
    model.setObjective(new_objective)
    model.update()
    if not (new_target_dose is None):
        change_model_dose_to_target(
            new_target_dose=new_target_dose, 
            model=model, 
            coords_target_constraint=target_constraints_coords, 
            coords_hotspot_constraint=hotspot_constraints_coords, 
            hotspot_threshold=hotspot_threshold
        )
    
    return model

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

# if __name__ == "__main__":
#     import gurobipy as gb
#     import json
#     env = Env(empty=True)
#     env.start()

#     model = gb.read("/app/EngerLab/tests/brachyutils/optim/model.mps")
#     # print(len(model.getAttr("VarName")))
#     # print(len(model.getAttr("Obj")))
#     # print(np.array(model.getAttr("rhs")))
#     print(len(np.array(model.getAttr("Obj"))))
#     print(np.unique(np.array(model.getAttr("Obj")), return_counts=True))
#     print(np.array(model.getAttr("Obj")))
#     exit()
#     # model = gb.read("/app/EngerLab/AI_Assisted_Brachytherapy/ai_pipeline_results_test_mobo_clean/Dataset007Dataset050/val_benchmark_fold_0/259984/manual_clinical_structures__clinical_dwellpos__auto_optimization/model_0.mps", env)
#     model_data = get_model_data(model)
#     with open("/app/EngerLab/tests/brachyutils/optim/coeffs.json", "r") as file:
#         coeffs = json.load(file)
#     # print(coeffs)
#     with Env() as env, Model(env=env) as new_model:
#         model_remade = update_model_from_data(model_data, new_model)
#         model_remade = modify_model_objective_with_new_penalty_weights(
#             model_remade, coeffs, 
#             # {"PTV": {"linear":500, "quadratic":0.5}, "Skin": {"linear":50, "quadratic":0.5}, "Chestwall": {"linear":50, "quadratic":0.5}}, 
#             {"PTV": {"linear":1000, "quadratic":1}, "Skin": {"linear":100, "quadratic":1}, "Chestwall": {"linear":100, "quadratic":1}}, 
#             inplace=True)
#         print(model_remade.getVarByName("C275(1)").Obj)
#         print(model.getVarByName("C275(1)").Obj)
#         print(compare_gurobi_models(model, model_remade))
#         exit(0)

#     print(model_data["objective"].keys())
#     print(model_data["objective"]["linear_list"][:2])
#     print(model.getVarByName("C276(1)").Obj)
#     exit()
#     print('model_data["lb"]', model_data["lb"])
#     with Env() as env, Model(env=env) as model:
#         model_remade = update_model_from_data(model_data, model)
#         print(compare_gurobi_models(model, model_remade))
#         exit()

#    # https://support.gurobi.com/hc/en-us/community/posts/12678106466321-Fast-creation-of-Gurobi-models

