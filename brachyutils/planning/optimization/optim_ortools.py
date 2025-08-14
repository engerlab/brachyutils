from typing import List, Literal
from copy import deepcopy
import warnings
import time
import numpy as np
from pathlib import Path
from brachyutils.types import BrachyPlan
from brachyutils.planning.optimization.optim_utils import (
    BrachyDwellTimeOptim, BrachyDwellTime_ABC, crop_mask_resample_dose_rate_map
)
from ortools.math_opt.python.mathopt import (
    Model,
    solve,
    SolverType,
    TerminationReason
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
        if not isinstance(model, Model):
            raise TypeError("The model must be an instance of ortools.math_opt.python.mathopt.Model.")

        self._model_variable = model.add_variable(
            lb=self.lower_bound,
            ub=self.upper_bound,
            name=self.name,
            is_integer=False)
    
    def set_bounds(self, *, lower_bound: float | None = None, upper_bound: float | None = None) -> None:
        r"""
        See `BrachyDwellTime_ABC.set_bounds` for details.
        """
        if lower_bound is not None:
            self.lower_bound = lower_bound
            self._model_variable.lower_bound = lower_bound
        if upper_bound is not None:
            self.upper_bound = upper_bound
            self._model_variable.upper_bound = upper_bound


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


class BrachyOptim_ORTools(BrachyDwellTimeOptim):
    r"""
    ### Purpose:
    - Optimize dwell times using OR-Tools.
    see `BrachyDwellTimeOptim` for more details on the attributes and methods.
    """
    def __init__(
        self,
        plan: BrachyPlan,
        solver: Literal["GLOP", "PDLP", "GSCIP", "SCIP", "GLPK"]="GLPK",
        roi_margin_mm: float = 0.0,
        pth_logfile: Path | str | None = None):
        r"""
        ### Purpose:
        - Initialize the BrachyOptim_ORTools object.
        ### Inputs:
        - plan: A BrachyPlan object containing the plan to optimize.
        - solver: The solver to use for optimization. Default is "GLOP" (Google's Linear Optimization Program).
        - roi_margin_mm: Margin in mm to apply to the ROIs for dose rate calculations.
        - pth_logfile: Optional path to a log file for the model.
        """
        super().__init__()
        self.plan: BrachyPlan = plan
        self.roi_margin_mm = roi_margin_mm if isinstance(roi_margin_mm, list) else [roi_margin_mm] * 3
        self.solver = solver
        self.model = self.initialize_model(pth_logfile=pth_logfile)
        self.dwellTimeVariables:DwellTime_ORTools = self.set_dwellTimeVariables(plan=self.plan)
        self.roi_bounds: List[List[float]] = self.get_optimization_roi_bounds(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            roi_margin_mm=self.roi_margin_mm
        )
        self.set_penalty_function_and_constraints(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            model=self.model
        )

    def initialize_model(self, pth_logfile = None):
        r"""
        ### Purpose:
        - Initialize the OR-Tools model.
        ### Inputs:
        - solver: The solver to use for optimization.
        - pth_logfile: Optional path to a log file for the model.
        """
        if pth_logfile is not None:
            warnings.warn("pth_logfile is not implemented in OR-Tools optimization.")
        model = Model(name="dwellTimeOptimizer")
        return model

    def set_dwellTimeVariables(
        self,
        plan: BrachyPlan,
        initial_dwell_time: float = 0,
        lower_bound: float = 0,
        upper_bound: float = 100):
        r"""
        See `BrachyDwellTime_ABC.set_dwellTimeVariables` for details.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please initialize the model first.")
        dwellTimeVariable_list = []
        dwell_counter = 0
        for catheter in plan.catheter_table:
            for dwell_position in catheter.dwells:
                dwellTimeVariable_list.append(
                    DwellTime_ORTools(
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

        return dwellTimeVariable_list

    def get_optimization_roi_bounds(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[DwellTime_ORTools],
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
        dwellTimeVariables: List[DwellTime_ORTools],
        model: Model,
        ):
        r"""
        ### Purpose:
        - A function to set up the optimization model's objective function and constraints based on the plan.
        For target structures, slack variables are added to ensure doses meet target goals with linear and quadratic
        penalties for underdosing, plus uniformity penalties. For OARs, slack variables with linear and quadratic
        penalties penalize overdosing above the target dose.
        
        The objective function takes the form:
        minimize sum(weights * penalties) where penalties include:
        - Linear penalties for over/under dosing relative to target dose
        - Quadratic penalties for over/under dosing 
        - Uniformity penalties for target volumes
        - Hotspot penalties for hotspot estimator structures encapsulating two closest dwell positions.
        
        ### Inputs:
        - plan: BrachyPlan := The plan containing structures and optimization configs
        - dwellTimeVariables: List[DwellTimeVariable] := The dwell time variables to optimize
        - model: Model := The OR-Tools optimization model

        ### Outputs:
        None - sets up the model objective function and constraints directly
        """
        # from scipy import sparse as sp

        print("Building OR-Tools optimization model...")
        total_dwells = len(dwellTimeVariables)
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
            dose_rate_matrices = []
            dwell_vars = []
            for variable in dwellTimeVariables:
                if "hotspot_estimator:" in structure.name.lower():
                    relevant_dwells = structure.name.lower().split("hotspot_estimator:")[1].split("/")
                    if variable.name not in relevant_dwells:
                        continue
                dwell_vars.append(variable._model_variable)
                cropped_resampled_dose_rate_map = crop_mask_resample_dose_rate_map(
                    dose_rate_map=variable.dose_rate_map,
                    template_dose_obj=plan.combined_dose,
                    roi_bounds=self.roi_bounds,
                    structure_mask=structure_mask,
                    optim_spacing=optim_spacing
                )
                # Extract valid dose points and flatten
                valid_dose_points = cropped_resampled_dose_rate_map[
                    cropped_resampled_dose_rate_map > 0
                ].flatten()
                dose_rate_matrices.append(valid_dose_points)

            if not dose_rate_matrices:
                continue

            A = np.column_stack(dose_rate_matrices)
            num_dose_points = A.shape[0]
            if num_dose_points == 0:
                continue

            if structure.target_volume:
                for i in range(num_dose_points):
                    # Linear penalty for underdosing
                    x_slack = model.add_variable(
                        lb=0.0,
                        ub=target_dose-min_dose,
                        name=f"{structure.name}_slack_{i}",
                        is_integer=False)
                    # quadratic penalty for uniformity
                    y_slack = model.add_variable(
                        lb=0.0,
                        ub=target_dose - min_dose,
                        name=f"{structure.name}_uniformity_slack_{i}",
                        is_integer=False
                    )

                    # add linear constraint for underdosing
                    model.add_linear_constraint(
                        sum(
                            A[i, j] * dwell_vars[j] for j in range(len(dwell_vars))
                        ) + x_slack >= target_dose
                    )
                    
                    # linear constarint for uniformity
                    model.add_linear_constraint(
                        sum(
                            A[i, j] * dwell_vars[j] for j in range(len(dwell_vars))
                        ) + y_slack == target_dose
                    )

                    # Add penalties to the objective function
                    penalty_terms["linear"] += linear_weight/num_dose_points * x_slack
                    penalty_terms["uniformity"] += uniformity_weight/num_dose_points * y_slack * y_slack
                    penalty_terms["quadratic"] += quadratic_weight/num_dose_points * x_slack * x_slack

            elif "hotspot_estimator:" in structure.name.lower():
                for i in range(num_dose_points):
                    x_slack = model.add_variable(
                        lb=0.0,
                        ub=hotspot_threshold*target_dose-min_dose,
                        name=f"{structure.name}_slack_{i}",
                        is_integer=False
                    )
                    model.add_linear_constraint(
                        sum(
                            A[i, j] * dwell_vars[j] for j in range(len(dwell_vars))
                        ) - x_slack <= hotspot_threshold * target_dose
                    )
                    penalty_terms["hotspot"] += hotspot_weight/num_dose_points * x_slack
            else:
                for i in range(num_dose_points):
                    # Linear penalty for overdosing
                    x_slack = model.add_variable(
                        lb=0.0,
                        ub=structure_max_dose - target_dose,
                        name=f"{structure.name}_slack_{i}",
                        is_integer=False)
                    # add the linear constraint
                    model.add_linear_constraint(
                        sum(
                            A[i, j] * dwell_vars[j] for j in range(len(dwell_vars))
                        ) - x_slack <= target_dose
                    )
                    # Add penalties to the objective function
                    penalty_terms["linear"] += linear_weight/num_dose_points * x_slack
                    penalty_terms["quadratic"] += quadratic_weight/num_dose_points * x_slack * x_slack
        # Set the objective function
        model.minimize(
            penalty_terms["linear"] +
            penalty_terms["quadratic"] +
            penalty_terms["hotspot"] +
            penalty_terms["uniformity"]
        )

    def run(self, solver: Literal["GLOP", "PDLP", "GSCIP", "SCIP", "GLPK"]):
        r"""
        ### Purpose:
        - A function to run the optimizer. See `BrachyDwellTimeOptim.run` for details. 
        """
        if solver == "GLOP":
            solver_type = SolverType.GLOP
        elif solver == "PDLP":
            solver_type = SolverType.PDLP
        elif solver == "SCIP":
            solver_type = SolverType.SCIP
        elif solver == "GLPK":
            solver_type = SolverType.GLPK
        elif solver == "GSCIP":
            solver_type = SolverType.GSCIP
        else:
            raise ValueError(f"Unsupported solver: {solver}. Supported solvers are: GLOP, PDLP, SCIP, GLPK.")
        
        time_start = time.time()
        results = solve(self.model, solver_type=solver_type)
        self.solve_time = time.time() - time_start
        
        if results.termination.reason == TerminationReason.OPTIMAL:
            self.solution_found = True
            print("Optimal solution found.")
            return results

    def get_optimized_plan_from_model(
        self, 
        solver:Literal["GLOP", "PDLP", "GSCIP", "SCIP", "GLPK"]=None, 
        inplace=True):
        r"""
        See `BrachyDwellTime_ABC.get_optimized_plan_from_model` for details.
        """
        if self.plan is None:
            raise ValueError("Plan is not set. Please set the plan first.")
        if self.model is None:
            raise ValueError("Model is not set. Please set the model first.")
        if self.dwellTimeVariables is None:
            raise ValueError("DwellTimeVariables are not set. Please set the DwellTimeVariables first.")
        if solver is None:
            solver = self.solver
        # run the optimization
        result = self.run(solver=solver)
        result_vars = result.variable_values()
        if self.solution_found == False:
            warnings.warn(
                "No optimal solution found. Return None.",
                stacklevel=2)
            return None

        for variable in self.dwellTimeVariables:
            # set the dwell time to the optimized value
            variable.dwell_time = result_vars[variable._model_variable]
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
        See `BrachyDwellTime_ABC.bound_dwell_time` for details.
        """
        for variable in self.dwellTimeVariables:
            if variable.name == name:
                variable.set_bounds(
                    model=self.model,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                )
                break
