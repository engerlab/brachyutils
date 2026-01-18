from typing import List, Any
from copy import deepcopy
import warnings
import time
import numpy as np
from pathlib import Path
from brachyutils.types import BrachyPlan
from brachyutils.planning.optimization.optim_utils import (
    BrachyDwellTimeOptim, BrachyDwellTime, get_optimization_roi_bounds, resample_crop_the_mask_or_contour_to_optimGrid,
    compute_dose_rate_matrices, Optimization_Config
)
from amplpy import AMPL

class DwellTime_AMPL(BrachyDwellTime):
    r"""
    ### Purpose:
    - A class to represent a DwellTimeVariable in the dwell time optimization problem using AMPL.
    See `BrachyDwellTime` for more details on the attributes and methods.
    """
    def _ampl_variable_exists(self, model: AMPL, name: str) -> bool:
        """
        Check if a variable with the given name exists in the AMPL model.
        """
        try:
            model.getVariable(name)
            return True
        except RuntimeError:
            return False

    def build_backend_variable(self, model):
        if not isinstance(model, AMPL):
            raise ValueError("Model is not an AMPL model. Please provide an AMPL model.")
        
        # check if variable already exists, if not, create it
        # if not self._ampl_variable_exists(model, self.name):
        model.eval(f"param {self.name}_lb; let {self.name}_lb := {self.lower_bound};")
        model.eval(f"param {self.name}_ub; let {self.name}_ub := {self.upper_bound};")
        model.eval(
            f"var {self.name} "
            f">= {self.name}_lb <= {self.name}_ub;"
        )
        self._model_variable = model.getVariable(self.name)

    def set_bounds(
        self, *,
        model: AMPL,
        lower_bound: float | None = None,
        upper_bound: float | None = None) -> None:
        r"""
        See `BrachyDwellTime.set_bounds` for details.
        """
        if lower_bound is not None:
            self.lower_bound = lower_bound
            model.eval(f"let {self.name}_lb := {self.lower_bound};")
        if upper_bound is not None:
            self.upper_bound = upper_bound
            model.eval(f"let {self.name}_ub := {self.upper_bound};")

    def __init__(self, model: AMPL, **data):
        r"""
        ### Purpose:
        - A function to initialize the DwellTimeVariable.
        ### Inputs:
        - model: AMPL := The AMPL model object.
        - data: dict := The data to initialize the DwellTimeVariable.
        """
        super().__init__(**data)
        self.build_backend_variable(model)

class BrachyOptim_AMPL(BrachyDwellTimeOptim):
    """
    ### Purpose:
    A class to solve dwell time optimization problems using AMPL. AMPL, allows for using a variety
    of solvers, for now we use it for HiGHS, but it can be used with other solvers as well.
    See `BrachyDwellTimeOptim` for more details on the attributes and methods.
    """
    def __init__(
        self,
        plan: BrachyPlan,
        roi_margin_mm: List[float] | float = 5.0,
        solver: str = "highs",
        verbose: bool = True):
        r"""
        ### Purpose:
        - A function to initialize the optimizer.
        ### Parameters:
        - roi_margin_mm: The distance from the furthest dwell position along each axis
        to consider voxels the dose rate maps. for each axis:
            roi_bounds = [first dwell - margin : last dwell + margin]
        - verbose: Whether to show AMPL solver output and progress
        """
        super().__init__()
        self.plan: BrachyPlan = plan
        self.roi_margin_mm = roi_margin_mm if isinstance(roi_margin_mm, list) else [roi_margin_mm] * 3
        self.solver = solver
        self.verbose = verbose
        self.model = self.initialize_model(self.solver)
        self.dwellTimeVariables:DwellTime_AMPL = self.set_dwellTimeVariables(plan=self.plan)
        self.roi_bounds: List[List[float]] = get_optimization_roi_bounds(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            roi_margin_mm=self.roi_margin_mm
        )
        self.set_penalty_function_and_constraints(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            model=self.model
        )

    def initialize_model(self, solver, pth_logfile = None):
        r"""
        ### Purpose:
        - See `BrachyDwellTimeOptim.initialize_model` for details.
        ### Inputs:
        - solver: str := The name of the solver to be used. Default is "highs".
        - pth_logfile: str := The path to the log file for the solver. Default is None.
        ### Outputs:
        - model: AMPL := The AMPL model object.
        """
        if pth_logfile is None:
            pth_logfile = Path(f"temp_data/{self.solver}.log").resolve()
        pth_logfile.parent.mkdir(parents=True, exist_ok=True)
        list_of_solvers = [
            "highs", "gurobi", "xpress",
            "cplex", "scip", "gcg", 
            "couenne", "bonmin", "copt",
            "mosek", "ipopt", "cuopt"
            ]
        if solver not in list_of_solvers:
            raise ValueError(f"Unsupported solver: {solver}. Supported solvers are {list_of_solvers}.")

        model = AMPL()
        model.option["solver"] = solver
            
        # Configure verbose output
        if self.verbose:
            # model.option["display_1col"] = 20  # Display up to 20 columns
            # model.option["display_eps"] = 1e-6  # Display precision
            # model.option["display_round"] = 6   # Rounding precision

            # Set log file
            model.option["log_file"] = str(pth_logfile)
            print(f"AMPL log file: {pth_logfile}")
            print(f"Using solver: {solver}")

        # set 10 minute time limit
        model.option["timelim"] = 600

        return model

    def set_dwellTimeVariables(
        self,
        plan: BrachyPlan,
        initial_dwell_time: float = 0.0,
        lower_bound: float = 0.0,
        upper_bound: float = 100,
    ) -> List[DwellTime_AMPL]:
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
                    DwellTime_AMPL(
                        model=self.model,
                        name=f"catheter_{catheter.index+1}_dwell_{dwell_position.index+1}",
                        dwell_time=initial_dwell_time,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        coordinates=dwell_position.position,
                        dose_rate_map=plan.dose_rate_dict[dwell_counter]
                    )
                )
                dwell_counter += 1

        return dwellTimeVariable_list


    # def get_optimization_roi_bounds(
    #     self,
    #     plan: BrachyPlan,
    #     dwellTimeVariables: List[DwellTime_AMPL],
    #     roi_margin_mm: List[float] = [5.0, 5.0, 5.0],
    # ) -> List[List[float]]:
    #     r"""
    #     See `BrachyDwellTime.get_optimization_roi_bounds` for details.
    #     """
    #     return get_optimization_roi_bounds(
    #         plan=plan,
    #         dwellTimeVariables=dwellTimeVariables,
    #         roi_margin_mm=roi_margin_mm
    #     )

    def set_penalty_function_and_constraints(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[DwellTime_AMPL],
        model: AMPL,
        multi_processing: bool = True,
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
        - model: Model := The AMPL optimization model

        ### Outputs:
        None - sets up the model objective function and constraints directly
        """
        # from scipy import sparse as sp
        
        print("Building AMPL optimization model...")
        
        # Initialize global model parameters once
        total_dwells = len(dwellTimeVariables)
        if self.verbose:
            print(f"Setting up {total_dwells} dwell time variables...")
        
        # Initialize objective function components
        objective_terms = []
        structure_counter = 0
        model.eval(f"param total_dwells := {total_dwells};")
        model.eval("set ALL_DWELLS := 1 .. total_dwells;")
        model.eval("var t_vec {ALL_DWELLS};")
        # Link individual dwell variables to the global vector
        for i, d_var in enumerate(dwellTimeVariables):
            model.eval(f"subject to t_def_{i+1}: t_vec[{i+1}] = {d_var._model_variable.name()};")

        # structures_with_config = [s for s in plan.structure_list if s.optimization_config is not None]
        # if self.verbose:
        #     print(f"Processing {len(structures_with_config)} structures with optimization configs...")
        
        # get structure optimization configs for hotspot estimators
        for structure in plan.structure_list:
            if structure.is_target and structure.optimization_config.penalty_weight_hotspot > 0:
                hotspot_config = structure.optimization_config
                # hotspot_config.structure_name = "hotspot"
        for structure in plan.structure_list:
            if "hotspot_estimator_" in structure.name.lower():
                structure_counter += 1
                structure_mask = structure.mask
                hotspot_config.structure_name = structure.name
                optim_spacing = hotspot_config.spacing_mm
                target_dose = hotspot_config.dose_voxel_goal
                linear_weight = hotspot_config.penalty_weight_linear
                quadratic_weight = hotspot_config.penalty_weight_quadratic
                uniformity_weight = hotspot_config.penalty_weight_uniformity
                min_dose = hotspot_config.min_dose
                structure_max_dose = hotspot_config.max_dose
                hotspot_weight = hotspot_config.penalty_weight_hotspot
                hotspot_threshold = hotspot_config.hotspot_threshold

            elif structure.optimization_config is None:
                continue
            else:
                structure_counter += 1
                structure_mask = structure.mask
                optim_spacing = structure.optimization_config.spacing_mm
                target_dose = structure.optimization_config.dose_voxel_goal
                linear_weight = structure.optimization_config.penalty_weight_linear
                quadratic_weight = structure.optimization_config.penalty_weight_quadratic
                uniformity_weight = structure.optimization_config.penalty_weight_uniformity
                min_dose = structure.optimization_config.min_dose
                structure_max_dose = structure.optimization_config.max_dose

            structure_mask = resample_crop_the_mask_or_contour_to_optimGrid(
                structure_mask=structure_mask,
                template_dose_obj=plan.combined_dose,
                optim_spacing=optim_spacing,
                roi_bounds=self.roi_bounds,
                )

            # Build dose rate matrix and dwell time vector for this structure
            dwell_vars_chaos, dose_rate_matrices_chaos = compute_dose_rate_matrices(
                dwellTimeVariables,
                plan,
                structure.name,
                structure_mask,
                optim_spacing,
                self.roi_bounds,
                # max_workers=8, 
                shift_origin=True,
                multi_processing=multi_processing
            )

            if not dose_rate_matrices_chaos:
                continue
            dwell_vars = []
            dose_rate_matrices = []
            # sort the dwell_vars and dose_rate_matrices according to the original dwellTimeVariables order
            for var in dwellTimeVariables:
                for var_mat in zip(dwell_vars_chaos, dose_rate_matrices_chaos):
                    if var_mat[0].name() == var.name:
                        dwell_vars.append(var_mat[0])
                        dose_rate_matrices.append(var_mat[1])
            # create the dose rate matrix A (n x m) for this structure
            A = np.column_stack(dose_rate_matrices)
            num_dose_points = A.shape[0]
            num_dwells = len(dwell_vars)
            if num_dose_points == 0:
                continue
            # Define structure-specific sets and parameters
            struct_id = f"s{structure_counter}"
            model.eval(f"param num_dose_points_{struct_id} := {num_dose_points};")
            model.eval(f"param num_dwells_{struct_id} := {num_dwells};")
            model.eval(f"set D_{struct_id} := 1 .. num_dose_points_{struct_id};")
            model.eval(f"set T_{struct_id} := 1 .. num_dwells_{struct_id};")
            # Define structure-specific dose rate matrix
            model.eval(f"param A_{struct_id}{{{{D_{struct_id},T_{struct_id}}}}};")
            model.param[f"A_{struct_id}"] = A
            
            # Define structure-specific parameters
            model.eval(f"param target_dose_{struct_id} := {target_dose};")
            model.eval(f"param min_dose_{struct_id} := {min_dose};")
            # Set up constraints and objective terms based on structure type
            if structure.is_target:
                if linear_weight > 0 or quadratic_weight > 0:
                    # Create structure-specific slack variables for underdosing
                    model.eval(f"var x_slack_{struct_id} {{D_{struct_id}}} >= 0 <= target_dose_{struct_id} - min_dose_{struct_id};")
                # Create structure-specific constraints
                    model.eval(
                        f"""
                        subject to dose_constraint_{struct_id} {{i in D_{struct_id}}}:
                            sum{{j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j] + x_slack_{struct_id}[i] >= target_dose_{struct_id};
                        """)
                if linear_weight > 0:
                    # Add penalty terms to objective
                    linear_term = f"({linear_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]"
                    objective_terms.extend([linear_term])
                if quadratic_weight > 0:
                    quadratic_term = f"({quadratic_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]^2"
                    objective_terms.extend([quadratic_term])
                if uniformity_weight > 0:
                    # slack variable for uniformity
                    model.eval(f"var y_slack_{struct_id} {{D_{struct_id}}} >= -Infinity <= target_dose_{struct_id} - min_dose_{struct_id};")

                    model.eval(
                        f"""
                        subject to uniformity_constraint_{struct_id} {{i in D_{struct_id}}}:
                            sum{{j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j] + y_slack_{struct_id}[i] = target_dose_{struct_id};
                        """)
                    uniformity_term = f"({uniformity_weight / (num_dose_points * 1000)}) * sum{{i in D_{struct_id}}} y_slack_{struct_id}[i]^2"
                    objective_terms.extend([uniformity_term])
                
            elif "hotspot_estimator_" in structure.name.lower():
                # Scalar slack variable for hotspot estimator
                model.eval(f"var x_slack_{struct_id} >= 0 <= {hotspot_threshold} * target_dose_{struct_id} - min_dose_{struct_id};")
                
                # Single constraint: average dose across all points minus slack
                model.eval(
                    f"""
                    subject to hotspot_constraint_{struct_id}:
                        (sum{{i in D_{struct_id}, j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j]) / card(D_{struct_id}) - x_slack_{struct_id} <= {hotspot_threshold} * target_dose_{struct_id};
                    """)
                
                # Add hotspot penalty term (no summation needed - scalar variable)
                hotspot_term = f"{hotspot_weight} * x_slack_{struct_id}"
                objective_terms.extend([hotspot_term])

            else:
                # OAR (Organ at Risk) constraints and penalties
                if linear_weight >0 or quadratic_weight >0:
                    
                    model.eval(f"param structure_max_dose_{struct_id} := {structure_max_dose};")
                    model.eval(f"var x_slack_{struct_id} {{{{D_{struct_id}}}}} >= 0 <= structure_max_dose_{struct_id} - target_dose_{struct_id};")
                    model.eval(
                        f"""
                        subject to oar_constraint_{struct_id} {{i in D_{struct_id}}}:
                            sum{{j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j] - x_slack_{struct_id}[i] <= target_dose_{struct_id};
                        """)
                if linear_weight > 0:
                    linear_term = f"({linear_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]"
                    objective_terms.extend([linear_term])
                if quadratic_weight > 0:
                    quadratic_term = f"({quadratic_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]^2"
                    objective_terms.extend([quadratic_term])

        # Create the combined objective function once all structures are processed
        if objective_terms:
            combined_objective = " + ".join(objective_terms)
            if self.verbose:
                print(f"Setting objective with {len(objective_terms)} penalty terms...")
            model.eval(f"minimize objective_function: {combined_objective};")
        else:
            # Fallback objective if no structures have optimization configs
            print("Warning: No optimization configs found, using dummy objective")
            model.eval("minimize objective_function: 0;")
        
        if self.verbose:
            print("Model building complete!")
            print("\n=== Pre-solve Model Summary ===")
            # Add some model validation
            try:
                # Check if model can be solved
                print("Validating model consistency...")
                model.eval("check;")  # AMPL's built-in consistency check
                print("Model validation passed!")
            except Exception as e:
                print(f"Model validation warning: {e}")

    def run(self):
        r"""
        ### Purpose:
        - A function to run the optimizer. See `BrachyDwellTimeOptim.run` for details. 
        """
        print("Starting AMPL optimization...")
        print(f"Number of variables: {len(self.dwellTimeVariables)}")
        print(f"Number of structures: {len([s for s in self.plan.structure_list if s.optimization_config])}")
        
        # Display model statistics before solving
        if self.verbose:
            print("\n=== Model Statistics ===")
            try:
                print(f"Variables: {self.model.get_value('_nvars')}")
                print(f"Constraints: {self.model.get_value('_ncons')}")
            except:
                print("Could not retrieve model statistics")
        
        start_time = time.time()

        solve_options = [
            "outlev=1" if self.verbose else "outlev=0",
            "logfile=" + str(self.model.option["log_file"]),
            # "timelimit=20",
            # "log_to_console=true",
        ]
        self.model.solve(solver=self.solver, options=solve_options)
        self.solve_time = time.time() - start_time

        # Get solve results
        solve_result = self.model.solve_result
        # solve_message = self.model.get_value("solve_message")
        
        print(f"\n=== Solve Results ===")
        print(f"Solve time: {self.solve_time:.2f} seconds")
        print(f"Solve result: {solve_result}")
        # print(f"Solve message: {solve_message}")
        
        if solve_result == "solved":
            print("✓ Optimal solution found.")
            self.solution_found = True
            if self.verbose:
                try:
                    obj_val = self.model.get_value("objective_function")
                    print(f"Objective value: {obj_val:.6e}")
                except:
                    print("Could not retrieve objective value")
        else:
            print("✗ No optimal solution found.")
            if self.verbose:
                print("Consider:")
                print("- Relaxing constraints")
                print("- Checking data consistency") 
                print("- Increasing solver time limit")

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

        self.run()
        if not self.solution_found:
            raise ValueError(
                f"The solver: {self.solver} did not find an optimal solution."
            )
        for variable in self.dwellTimeVariables:
            # set the dwell time to the optimized value
            variable.dwell_time = self.model.get_value(variable.name)
            if variable.dwell_time < 0.1:
                variable.dwell_time = 0

        # set the dwell time to the plan
        if inplace:
            outplan: BrachyPlan = self.plan
        else:
            outplan: BrachyPlan = deepcopy(self.plan)

        for variable in self.dwellTimeVariables:
            for catheter in outplan.catheter_table:
                for dwell_position in catheter.dwells:
                    if (
                        f"catheter_{catheter.index+1}_dwell_{dwell_position.index+1}"
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
                variable.set_bounds(
                    model=self.model,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                )
                break