# from abc import ABC, abstractmethod
from typing import List
import tqdm
import time
from copy import deepcopy
import warnings
import time
import numpy as np
from pathlib import Path
from gurobipy import Model, Var, GRB, MVar
from concurrent.futures import ThreadPoolExecutor, as_completed

from opentps.core.data.images import ROIMask

from brachyutils.types import BrachyPlan
from brachyutils.planning.optimization.optim_utils import (
    BrachyDwellTimeOptim, BrachyDwellTime, crop_mask_resample_dose_rate_map
)


def process_variable(variable, structure_name, structure_mask, plan, optim_spacing, roi_bounds):

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
        masked_dose_rate_obj, structure_for_masking = crop_mask_resample_dose_rate_map(
            dose_rate_map=variable.dose_rate_map,
            template_dose_obj=plan.combined_dose,
            roi_bounds=roi_bounds,
            structure_mask=structure_mask,
            optim_spacing=optim_spacing,
        )
        masked_dose_array = masked_dose_rate_obj.dose_image.imageArray.astype(float)
        structure_mask = structure_for_masking

    structure_for_masking = structure_mask.imageArray.astype(bool)
    valid_dose_points = masked_dose_array[structure_for_masking == 1].flatten()

    return dwell_var, valid_dose_points

def compute_dose_rate_matrices(dwellTimeVariables, structure, structure_mask, plan, optim_spacing, roi_bounds, max_workers=8):
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
                roi_bounds
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
                dwellTimeVariable_list.append(
                    DwellTime_Gurobi(
                        model=self.model,
                        name=f"catheter_{catheter.index}_dwell_{dwell_position.index}",
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
                        self.roi_bounds
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
                    max_workers=4
                )

            if not dose_rate_matrices:
                continue

            # conver the list of varaibles to a Gurobi variable Vector (MVar)
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

    def run(self):
        r"""
        ### Purpose:
        - A function to run the optimizer. See `BrachyDwellTimeOptim.run` for details. 
        """
        time_start = time.time()
        self.model.optimize()
        time_end = time.time()
        self.solve_time = time_end - time_start
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
            self.solution_found = True
        else:
            print("No optimal solution found.")

    def get_optimized_plan_from_model(
        self,
        inplace=True,
        ) -> BrachyPlan | None:
        r"""
        See `BrachyDwellTime.get_optimized_plan_from_model` for details.
        """
        if self.plan is None:
            raise ValueError("Plan is not set. Please set the plan first.")
        if self.model is None:
            raise ValueError("Model is not set. Please set the model first.")
        if self.dwellTimeVariables is None:
            raise ValueError("DwellTimeVariables are not set. Please set the DwellTimeVariables first.")

        # run the optimization
        self.run()
        if self.solution_found == False:
            warnings.warn(
                "No optimal solution found. Return None.",
                stacklevel=2)
            return None

        for variable in self.dwellTimeVariables:
            # set the dwell time to the optimized value
            variable.dwell_time = variable._model_variable.X
            if variable.dwell_time < 0.1:
                variable.dwell_time = 0
            # set the dwell time to the plan
            if inplace:
                outplan:BrachyPlan = self.plan
            else:
                outplan:BrachyPlan = deepcopy(self.plan)
            for catheter in outplan.catheter_table:
                for dwell_position in catheter.dwells:
                    if (
                        f"catheter_{catheter.index}_dwell_{dwell_position.index}"
                        == variable.name
                    ):
                        dwell_position.time = variable.dwell_time        
        # update the plan with the new dwell times
        outplan.update_plan_from_catheter_table()
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
