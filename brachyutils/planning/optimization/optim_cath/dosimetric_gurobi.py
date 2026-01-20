from typing import Dict, List, Optional
from tqdm import tqdm
from pathlib import Path

from gurobipy import Model, Var, GRB, MVar
import numpy as np

from brachyutils.types import BrachyPlan, BrachyDose
from brachyutils.planning.optimization.optim_utils import (
    get_optimization_roi_bounds, resample_crop_the_mask_or_contour_to_optimGrid,
    compute_dose_rate_matrices, Optimization_Config, Constraint_Config
)
from brachyutils.planning.optimization.optim_gurobi import (
    DwellTime_Gurobi, _run, _get_optimized_plan_from_model)

# likley to be factored out later
from brachyutils.geometry.catheter_utils.catheter_table import Catheter
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
        dose_rates: Optional[List[np.ndarray] | Dict[str, BrachyDose]] = None,
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
            dwell_var_name=f"{self.name}_dwell_{dwell.index+1}"
            self.dwelltime_variables.append(
                DwellTime_Gurobi(
                    model = model,
                    name = dwell_var_name,
                    dwell_time = dwell.time,
                    lower_bound = lower_dwelltime,
                    upper_bound = upper_dwelltime,
                    coordinates = dwell.position,
                    dose_rate_map = self.dose_rates.get(dwell_var_name) if self.dose_rates is not None else None,
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

        # attributes for later developement. may not be needed XXX
        # these may not be needed
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

        if self.roi_margin_mm[0] is not None:
            self.roi_bounds = get_optimization_roi_bounds(
                plan=self.plan,
                dwellTimeVariables=self.dwellTimeVariables,
                roi_margin_mm=self.roi_margin_mm,
            )
        set_dwell_coef_dict_per_structure(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            optim_roi_bounds=self.roi_bounds,
            multi_processing=multi_processing
        )
        self.set_penalty_function_and_constraints(
            optimization_configs=[
                struc.optimization_config
                for struc in self.plan.structure_list],
            dwellTimeVariables=self.dwellTimeVariables,
            catheter_vars=self.catheter_vars,
            model=self.model,
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
        catheter_vars_to_keep:List[CatheterVar_Gurobi]=None,
        ) -> List[CatheterVar_Gurobi]:
        r"""
        ### Purpose:
        - To extract catheter variables from the plan (catheter table and dose rate dict).
        If a catheter is already in catheter_vars_to_keep, it will not be re-written
        ### Inputs:
        - plan:= Brachy plan with a new catheter table and dose rate dict
        - model:= the optimization model that will have the new catheter and dwell time variables
        - catheter_vars_to_keep:= The list of catheter variables that we want to keep in the model
        otherwise, they will be re-written.
        ### Output:
        - List[CatheterVar_Gurobi] := the new list of catheter variables that have been added
        to the model.
        """
        return set_catheter_variables(
            plan=plan,
            model=model,
            catheter_vars_to_keep=catheter_vars_to_keep
        )

    def set_penalty_function_and_constraints(
    self,
    optimization_configs:List[Optimization_Config],
    dwellTimeVariables:List[DwellTime_Gurobi],
    catheter_vars: List[CatheterVar_Gurobi],
    model: Model,
    ):
        r"""
        ### Purpose:
        - sets the penalty function and constraints for the optimization model.
        ### Inputs:
        - `optimization_configs`: List[Optimization_Config] := List of optimization configs containing the
        penalty weights, target dose, mask, dwell_coef_dict and other attibutes.
        - `catheter_vars`: List[CatheterVar_Gurobi] := the catheter variables to be used in the optimization.
        - `model`: Model := the Gurobi model to which the variables will be added.
        - `multi_processing`: bool := whether to use multi-processing for dose rate matrix computations.
        """
        set_penalty_function_and_constraints(
        optimization_configs = optimization_configs,
        dwellTimeVariables = dwellTimeVariables,
        catheter_vars = catheter_vars,
        model = model,
        )

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

    def bound_variables(
        self,
        constraint_configs:List[Constraint_Config],
        ):
        """
        ### Purpose:
        - To bound the model variables according the list of constraint config. The bound could be on the 
        lower bound, upper bound or equality value of the variable.
        - The name of the constraints on the number of catheters (sum of binary variable) or the total
        dwell times should being with "sum_catheters" and "sum_dwelltimes".

        ### Inputs:
        - constraint_configs (List[Constraint_Config]): Each item in this list contains the name of the
        variable as well as minimum, maximum and equality constraints on that variable.
        - model (Model): The model containing the variables. The name of the variables in the constraint list 
        should match the name of the variable. Otherwies, Error will be thrown.
        ### Outputs:
        - None: model is updated with the new constraints
        """
        bound_variables(
            constraint_configs=constraint_configs,
            model=self.model
        )

def set_catheter_variables(
    plan: BrachyPlan,
    model: Model,
    catheter_vars_to_keep:List[CatheterVar_Gurobi]=None,
    ) -> List[CatheterVar_Gurobi]:
    r"""
    ### Purpose:
    - To extract catheter variables from the plan (catheter table and dose rate dict).
    If a catheter is already in catheter_vars_to_keep, it will not be re-written
    ### Inputs:
    - plan:= Brachy plan with a new catheter table and dose rate dict
    - model:= the optimization model that will have the new catheter and dwell time variables
    - catheter_vars_to_keep:= The list of catheter variables that we want to keep in the model
    otherwise, they will be re-written.
    ### Output:
    - List[CatheterVar_Gurobi] := the new list of catheter variables that have been added
    to the model.
    """
    catheter_vars_to_add = []
    if catheter_vars_to_keep:
        name_cath_to_keep = [cath.name for cath in catheter_vars_to_keep]
    else:
        catheter_vars_to_keep=[]
        name_cath_to_keep=[]
    for catheter in tqdm(
        plan.catheter_table,
        total=len(plan.catheter_table.catheter_list),
        desc="Creating optimization variables from new catheters"):
        if f"catheter_{catheter.index+1}" in name_cath_to_keep:
            continue
        dose_rates = plan.get_dose_rate_matrices_for_catheter(catheter.index)

        catheter_vars_to_add.append(
            CatheterVar_Gurobi(
            catheter=catheter,
            model=model,
            dose_rates=dose_rates,
            )
        )
    model.update()
    return catheter_vars_to_keep+catheter_vars_to_add

def set_penalty_function_and_constraints(
    optimization_configs:List[Optimization_Config],
    dwellTimeVariables:List[DwellTime_Gurobi],
    catheter_vars: List[CatheterVar_Gurobi],
    model: Model,
    ):
    r"""
    ### Purpose:
    - sets the penalty function and constraints for the optimization model.
    XXX more details to be added for the docs
    ### Inputs:
    - `optimization_configs`: List[Optimization_Config] := List of optimization configs containing the
    penalty weights, target dose, mask, dwell_coef_dict and other attibutes.
    - `dwellTimeVariables`: The list of the dwell times variables in the catheter table. it should be
    synched up with catheter_vars.
    - `catheter_vars`: List[CatheterVar_Gurobi] := the catheter variables to be used in the optimization.
    - `model`: Model := the Gurobi model to which the variables will be added.
    - `multi_processing`: bool := whether to use multi-processing for dose rate matrix computations.
    ### Output:
    - None: The model is updated with the constraints and penalty function.
    """
    # if not plan.structure_list:
    #     raise ValueError("Plan does not contain any structures.")

    penalty_terms = {
    "linear": 0,
    "quadratic": 0,
    "hotspot": 0,
    "uniformity": 0,
    "dwelltimes":0
    }
    # # create the 
    t_MVar = MVar([dt._model_variable for dt in dwellTimeVariables])
    c_MVar = MVar([c._model_variable for c in catheter_vars for _ in c])

    for optimization_config in optimization_configs:
        if not optimization_config.dwell_coef_dict:
            raise ValueError("The coefficint dictionary is empty. \
please run set_dwell_coef_dict_per_structure")

        min_dose = optimization_config.min_dose
        max_dose = optimization_config.max_dose

        voxel_goal = optimization_config.dose_voxel_goal
        linear_weight = optimization_config.penalty_weight_linear
        quadratic_weight = optimization_config.penalty_weight_quadratic
        uniformity_weight = optimization_config.penalty_weight_uniformity

        penalty_weight_variance_time = optimization_config.penalty_weight_variance_time

        # now sort the dose rate matrices and dwell vars per catheter
        A_sparse = np.column_stack(list(optimization_config.dwell_coef_dict.values()))
        num_dose_points = A_sparse.shape[0]
        if num_dose_points == 0:
            continue

        voxel_goal_vec = np.ones(num_dose_points)*voxel_goal

        if optimization_config.is_target:
            if linear_weight > 0 or quadratic_weight > 0:
                x_slack = model.addMVar(
                    shape=num_dose_points,
                    lb=0.0,
                    ub=voxel_goal - min_dose,
                    name=f"p_L_{optimization_config.structure_name}"
                    )
                model.addConstr(
                    A_sparse @ (c_MVar * t_MVar) + x_slack >= voxel_goal_vec,
                    name=f"c_L_{optimization_config.structure_name}"
                    )
            if linear_weight > 0:
                linear_weight_vec = np.ones_like(voxel_goal_vec)*linear_weight/num_dose_points
                penalty_terms["linear"] += sum((linear_weight_vec) * x_slack)

            if quadratic_weight > 0:
                quadratic_weight_vec = np.ones_like(voxel_goal_vec)*quadratic_weight/num_dose_points
                penalty_terms["quadratic"] += sum((quadratic_weight_vec) * (x_slack * x_slack))

            if uniformity_weight > 0:
                y_uniform = model.addMVar(
                    shape=num_dose_points,
                    lb=-GRB.INFINITY,
                    ub=voxel_goal - min_dose,
                    name=f"p_U_{optimization_config.structure_name}"
                )
                # Uniformity constraints: A @ dwell_times + y_uniform == voxel_goal
                model.addConstr(
                    A_sparse @ (c_MVar * t_MVar) + y_uniform == voxel_goal_vec,
                    name=f"c_U_{optimization_config.structure_name}"
                )
                uniformity_weight_vec = np.ones_like(voxel_goal_vec)*uniformity_weight/num_dose_points
                penalty_terms["uniformity"] += sum((uniformity_weight_vec) * (y_uniform * y_uniform))

            if penalty_weight_variance_time > 0:
                mean_dwell_time = sum(t_MVar) / t_MVar.size
                penalty_terms["dwelltimes"] += (
                    penalty_weight_variance_time * 1e-3 
                    * sum((t_MVar - mean_dwell_time) * (t_MVar - mean_dwell_time))/ t_MVar.size
                )

        elif "hotspot_estimator_" in optimization_config.structure_name:
            x_slack_hotspot = model.addMVar(
                shape=num_dose_points,
                name=f"p_H_{optimization_config.structure_name}"
            )
            model.addConstr(
            (A_sparse @ (c_MVar * t_MVar)) - x_slack_hotspot <= (voxel_goal_vec),
            name=f"c_H_{optimization_config.structure_name}",
            )
            hotspot_weight_vec = np.ones_like(voxel_goal_vec)*linear_weight/num_dose_points
            penalty_terms["hotspot"] += sum(hotspot_weight_vec * x_slack_hotspot)

        # OAR constraints and penalties
        else:
            if linear_weight > 0 or quadratic_weight > 0:
                x_slack_oar = model.addMVar(
                    shape=num_dose_points,
                    lb=0.0,
                    ub=max_dose - min_dose,
                    name=f"p_L_{optimization_config.structure_name}"
                )
                model.addConstr(
                    A_sparse @ (c_MVar * t_MVar) - x_slack_oar <= voxel_goal_vec,
                    name=f"c_L_{optimization_config.structure_name}"
                )

            if linear_weight > 0:
                linear_weight_vec_oar = np.ones_like(voxel_goal_vec)*linear_weight/num_dose_points
                penalty_terms["linear"] += sum(linear_weight_vec_oar * x_slack_oar)

            if quadratic_weight > 0:
                quadratic_weight_vec_oar = np.ones_like(voxel_goal_vec)*quadratic_weight/num_dose_points
                penalty_terms["quadratic"] += sum(quadratic_weight_vec_oar * (x_slack_oar * x_slack_oar))

    # Set the objective function
    model.setObjective(
        penalty_terms["linear"]
        + penalty_terms["quadratic"]
        + penalty_terms["uniformity"]
        + penalty_terms["hotspot"]
        + penalty_terms["dwelltimes"],
        GRB.MINIMIZE
    )
    model.update()


def set_dwell_coef_dict_per_structure(
    plan: BrachyPlan,
    dwellTimeVariables:List[DwellTime_Gurobi],
    optim_roi_bounds:List[List[float]]=None,
    multi_processing:bool=False,
    ):
    r"""
    ### Purpose:
    - To build the coefficients tensor (A matrix) per each structure to be used later for
    the constraint and penalty weight creation.
    If structure.optimization_config.mask is not None, this function will not recreate it.
    If new catheters are inserted, only provide the dwellTimeVariables from the new catheters
    ### Inputs:
    - plan: BrachyPlan:= a treatment plan containing the masks of the structures and catheter table
    - dwellTimeVariables:= a list of dwell time variables. they could belong to the etire
    catheter table or just a new set of catheters.
    - optim_roi_bounds:= The optimization region of interest (roi) from the plan.
    - multi_processing:= whether to use multi-processing for cropping, masking and resampling 
    dose rate maps.
    """
    for structure in plan.structure_list:
        if structure.optimization_config is None:
            continue
        if structure.optimization_config.mask is None:
            structure_mask = resample_crop_the_mask_or_contour_to_optimGrid(
                structure_mask=structure.mask,
                template_dose_obj=plan.combined_dose,
                optim_spacing=structure.optimization_config.spacing_mm,
                roi_bounds=optim_roi_bounds,
                )
            structure.optimization_config.mask = structure_mask
        # Build dose rate matrix and dwell time vector for this structure
        dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
            dwellTimeVariables,
            # plan,
            structure_name=structure.name,
            structure_mask=structure_mask,
            optim_spacing=structure.optimization_config.spacing_mm,
            roi_bounds=optim_roi_bounds,
            max_workers=16,
            shift_origin=True,
            multi_processing=multi_processing,
            )
        # build the coeff matricies
        for var, coeff in zip(dwell_vars, dose_rate_matrices):
            structure.optimization_config.dwell_coef_dict[var.VarName] = coeff

def bound_variables(
    constraint_configs:List[Constraint_Config],
    model:Model,
    ):
    """
    ### Purpose:
    - To bound the model variables according the list of constraint config. The bound could be on the 
    lower bound, upper bound or equality value of the variable.
    - The name of the constraints on the number of catheters (sum of binary variable) or the total
    dwell times should being with "sum_catheters" and "sum_dwelltimes".

    ### Inputs:
    - constraint_configs (List[Constraint_Config]): Each item in this list contains the name of the
    variable as well as minimum, maximum and equality constraints on that variable.
    - model (Model): The model containing the variables. The name of the variables in the constraint list 
    should match the name of the variable. Otherwies, Error will be thrown.
    ### Outputs:
    - None: model is updated with the new constraints
    """
    for constraint in constraint_configs:
        # check if the constraint already exists, if yes remove it
        old_constraint = model.getConstrByName(f"c_{constraint.name}")
        if old_constraint:
            model.remove(old_constraint)
            model.update()

        # if the constraint is on the sum of catheters or dwell times
        if constraint.name.startswith("sum_"):
            all_vars = model.getVars()
            var_target = constraint.name.split("_")[-1]
            vars_needed = []
            # gatheter all catheter or dwell variables
            for this_var in all_vars:
                if var_target == "catheters":
                    # we are looking for catheter variables only
                    if (this_var.name.startswith("catheter") and 
                        not "dwell" in this_var.name):
                        vars_needed.append(this_var)
                elif var_target == "dwelltimes":
                    if (this_var.name.startswith("catheter") and 
                        "dwell" in this_var.name):
                        vars_needed.append(this_var)
            # apply the constraint
            vars_needed = MVar(vars_needed)
            if constraint.minimum:
                model.addConstr(
                    sum(vars_needed) >= constraint.minimum,
                    name=f"c_{constraint.name}"
                )
            if constraint.maximum:
                model.addConstr(
                    sum(vars_needed) <= constraint.maximum,
                    name=f"c_{constraint.name}"
                )
            if constraint.equal:
                model.addConstr(
                    sum(vars_needed) == constraint.equal,
                    name=f"c_{constraint.name}"
                )
        # other
        else:
            variable = model.getVarByName(constraint.name)
            if not variable:
                raise ValueError(f"No variable with name {constraint.name} was found. \
Ensure the constraint name is correct.")
            if constraint.minimum:
                variable.LB = constraint.minimum
            if constraint.maximum:
                variable.UB = constraint.maximum
            if constraint.equal:
                model.addConstr(
                    variable == constraint.equal, 
                    name=f"c_{constraint.name}"
                )
    model.update()
