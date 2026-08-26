from time import time
from typing import Dict, List, Optional, Literal
from tqdm import tqdm
from pathlib import Path
import re

from gurobipy import Model, Var, GRB, MVar, GurobiError
import numpy as np

from brachyutils.brachy_types import BrachyPlan, BrachyDose
from brachyutils.planning.optimization.optim_utils import (
    get_optimization_roi_bounds, resample_crop_the_mask_or_contour_to_optimGrid,
    compute_dose_rate_matrices)

from brachyutils.planning.optimization.optim_configs import Optimization_Config, Constraint_Config
from brachyutils.planning.optimization.optim_gurobi import (
    DwellTime_Gurobi, _run, _get_optimized_plan_from_model)

# likley to be factored out later
from brachyutils.geometry.catheter_utils.catheter_table import Catheter
from itertools import chain
from collections import defaultdict


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
        upper_dwelltime: Optional[float] | Dict[str, float] = 5000.0,
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
        self.name: str = f"catheter_{catheter.name_id}"
        self.dwelltime_variables: List[DwellTime_Gurobi] = []
        self.dose_rates = dose_rates
        self.build_backend_variable(model=model)
        for dwell in catheter.dwells:
            dwell_var_name = f"dwell_{dwell.name_id}"
            self.dwelltime_variables.append(
                DwellTime_Gurobi(
                    model=model,
                    name=dwell_var_name,
                    dwell_time=dwell.time,
                    lower_bound=lower_dwelltime,
                    upper_bound=upper_dwelltime,
                    coordinates=dwell.position,
                    dose_rate_map=self.dose_rates.get(dwell_var_name) if self.dose_rates is not None else None,
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
            ub=1,
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
    - `solver`: str := the solver used for optimization. default is "gurobi"
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
    - `build_time`: float := the time taken to build the optimization model from plan.

    ### No handles, no blueprint - names are the single source of truth:
    Penalty slack variables and constraints follow a fixed, predictable naming convention
    (`p_L_{structure_name}[i]`, `c_L_{structure_name}[i]`, `p_U_{structure_name}[i]`,
    `c_U_{structure_name}[i]`, `p_H_{hotspot_suffix}[i]`, `c_H_{hotspot_suffix}[i]`). Nothing about
    them is cached on this class or anywhere else: `update_penalty_weights_and_voxel_goals` regenerates
    these names on the fly from each `Optimization_Config` (`structure_name`, `num_dose_points`,
    `hotspot_masks`, `hotspot_num_dose_points`) and resolves the live Gurobi objects via
    `model.getVarByName`/`model.getQConstrByName`. This means there is nothing to keep in sync, and
    nothing that becomes stale or invalid if the model is copied, saved/reloaded, or passed around.

    ### Important note on constraint attributes:
    Every dose constraint created here (`c_L_*`, `c_U_*`, `c_H_*`) involves the bilinear term
    `c_MVar * t_MVar` (catheter indicator times dwell time). Gurobi therefore stores these as
    QUADRATIC constraints (`QConstr`/`MQConstr`), even though they look like plain linear dose
    constraints. That means their right-hand side is read/written through `QCRHS`, NOT `RHS`, and
    they must be looked up with `getQConstrByName`, NOT `getConstrByName`.
    """

    def __init__(
        self,
        plan: BrachyPlan,
        roi_margin_mm: float = None,
        multi_processing: bool = False,
        pth_logfile: Path = "temp_data/gurobi_model.log"
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
        # Initialize the attributes to their default values
        self.plan: BrachyPlan = plan
        self.solver = "gurobi"
        self.model = None
        self.catheter_vars: List[CatheterVar_Gurobi] = []
        self.dwellTimeVariables: List[DwellTime_Gurobi] = []
        self.roi_margin_mm: List[float] = None
        self.roi_bounds: List[List[float]] = None

        if roi_margin_mm is not None:
            self.roi_margin_mm: float = (
                roi_margin_mm if isinstance(roi_margin_mm, list)
                else [roi_margin_mm] * 3
            )

        self.solution_found: bool = False
        self.solve_time: float = 0.0
        self.build_time: float = 0.0
        self.multi_processing = multi_processing

        # start building this optimization object
        t0_build = time()
        self.model = self.initialize_model(self.solver, pth_logfile=pth_logfile)
        self.catheter_vars = self.set_catheter_variables(
            plan=self.plan,
            model=self.model,
        )

        self.dwellTimeVariables = list(chain.from_iterable(self.catheter_vars))

        if self.roi_margin_mm is not None:
            self.roi_bounds = get_optimization_roi_bounds(
                plan=self.plan,
                dwellTimeVariables=self.dwellTimeVariables,
                roi_margin_mm=self.roi_margin_mm,
            )

        set_dwell_coef_dict_per_structure(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            optim_roi_bounds=self.roi_bounds,
            multi_processing=multi_processing,
        )

        self._bound_dwell_times_to_catheters(
            dwellTimeVariables=self.dwellTimeVariables,
            catheter_vars=self.catheter_vars,
            model=self.model,
        )

        self.set_penalty_function_and_constraints(
            optimization_configs=[
                struc.optimization_config
                for struc in self.plan.structure_list
                if struc.optimization_config is not None],
            dwellTimeVariables=self.dwellTimeVariables,
            catheter_vars=self.catheter_vars,
            model=self.model,
        )

        if plan.optimization_constraint_dict is not None:
            self.set_constraints(
                constraint_config_dict=plan.optimization_constraint_dict)
        t1_build = time()
        self.build_time = t1_build - t0_build

    def initialize_model(self, solver: str, pth_logfile: str = None) -> Model:
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
        pth_logfile = Path(pth_logfile)
        pth_logfile.parent.mkdir(parents=True, exist_ok=True)
        model = Model("CatheterTable_Optimization")
        model.Params.TimeLimit = 180  # set a time limit.
        # model.setParam("MIPFocus", 1)  # was not helpful.
        model.setParam("PreSOS1BigM", -1)
        model.setParam("LogFile", str(pth_logfile))
        return model

    def set_catheter_variables(
        self,
        plan: BrachyPlan,
        model: Model,
        catheter_vars_to_keep: List[CatheterVar_Gurobi] = None,
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
            catheter_vars_to_keep=catheter_vars_to_keep,
        )

    def set_penalty_function_and_constraints(
        self,
        optimization_configs: List[Optimization_Config],
        dwellTimeVariables: List[DwellTime_Gurobi],
        catheter_vars: List[CatheterVar_Gurobi],
        model: Model,
        cleanup: bool = True,
    ):
        r"""
        ### Purpose:
        - sets the penalty function and constraints for the optimization model.
        ### Inputs:
        - `optimization_configs`: List[Optimization_Config] := List of optimization configs containing the
        penalty weights, target dose, mask, dwell_coef_dict and other attibutes.
        - `catheter_vars`: List[CatheterVar_Gurobi] := the catheter variables to be used in the optimization.
        - `model`: Model := the Gurobi model to which the variables will be added.
        - `cleanup`: bool := if True (default), any slack variables/constraints previously created by a
        prior call of this function for the *same* structures are removed first. This makes it safe to
        call this function repeatedly (e.g. after changing the structure list or masks) without
        accumulating duplicate variables/constraints, since Gurobi does NOT deduplicate by name on its own.
        ### Output:
        - None: the model's variables/constraints/objective are created and set to match the current
        `optimization_configs` values. Nothing is returned or cached on `self` - every subsequent
        weight/target-dose update resolves objects by name via `update_penalty_weights_and_voxel_goals`.
        """
        set_penalty_function_and_constraints(
            optimization_configs=optimization_configs,
            dwellTimeVariables=dwellTimeVariables,
            catheter_vars=catheter_vars,
            model=model,
            cleanup=cleanup,
        )

    def update_penalty_weights_and_voxel_goals(
        self,
        optimization_configs: List[Optimization_Config],
    ):
        r"""
        ### Purpose:
        - Dynamically updates penalty weights and/or the per-voxel target dose for structures that
        already have slack variables/constraints in `self.model` (i.e.
        `set_penalty_function_and_constraints` has already been called at least once). Does NOT call
        `addVar`/`addConstr`: it regenerates the expected variable/constraint names from each
        `Optimization_Config` and resolves them via `getVarByName`/`getQConstrByName`, then mutates
        `Obj`, `QCRHS` and `UB` attributes in place.
        ### Inputs:
        - `optimization_configs`: List[Optimization_Config] := the (updated) optimization configs.
        ### Output:
        - None: `self.model`'s objective, quadratic-constraint RHS, and variable bounds are updated in place.
        """
        update_penalty_weights_and_voxel_goals(
            model=self.model,
            optimization_configs=optimization_configs,
        )

    def run(self):
        r"""
        ### Purpose:
        - A function to run the optimizer. See `BrachyDwellTimeOptim.run` for details.
        """
        self.model, self.solution_found, self.solve_time = _run(self.model)

    def get_optimized_plan_from_model(
        self,
        inplace: bool = True,
    ) -> BrachyPlan | None:
        r"""
        See `BrachyDwellTime.get_optimized_plan_from_model` for details.
        """
        self.model, outplan, self.solution_found, self.solve_time = _get_optimized_plan_from_model(
            plan=self.plan,
            model=self.model,
            inplace=inplace,
        )
        return outplan

    def set_constraints(
        self,
        constraint_config_dict: Dict[str, Constraint_Config],
    ):
        """
        ### Purpose:
        - To update the model with the new constraints. The type of the constraints can be
        "bound", "sum", "uniqueness", "continuity", "num_catheters", "collision", each is explained in
        Constraint_Config class. The target variables for the constraints can be "catheter" or "dwell".
        - Each constraint should have a unique name generated automatically by the Constraint_Config class.

        ### Inputs:
        - constraint_configs Dict[str, Constraint_Config]: See Constraint_Config for details.
        The key of this dictionary is the name of the constraint, which should be unique.
        - model (Model): The model containing the variables. The name of the variables in the constraint list
        should match the name of the variable. Otherwies, Error will be thrown.

        ### Outputs:
        - None: model is updated with the new constraints
        """
        set_constraints(
            constraint_config_dict=constraint_config_dict,
            model=self.model,
        )

    def _bound_dwell_times_to_catheters(
        self,
        dwellTimeVariables: List[DwellTime_Gurobi],
        catheter_vars: List[CatheterVar_Gurobi],
        model: Model,
        cohesion_type: Literal["BigM", "Indicator"] = "Indicator"):
        r"""
        We bound each dwell time variable to be equal to that variable multiplied by its corresponding
        catheter variable. t = ct. This ensures that if a catheter variable is zeor, the dwell time is
        also zero.
        """
        t_MVar = MVar([dt._model_variable for dt in dwellTimeVariables])
        c_MVar = MVar([c._model_variable for c in catheter_vars for _ in c])

        if cohesion_type == "BigM":
            t_MVar_Max = np.array([
                dt._model_variable.UB
                for dt in dwellTimeVariables])
            model.addConstr(
                t_MVar <= c_MVar * t_MVar_Max,
                name="cohesion")
        elif cohesion_type == "Indicator":
            for c_var, t_var in zip(c_MVar.tolist(), t_MVar.tolist()):
                model.addGenConstrIndicator(
                    c_var, False, t_var, GRB.EQUAL, 0.0,
                    name="cohesion")
        model.update()

    def remove_constraints(
        self,
        constraint_config_dict: Dict[str, Constraint_Config],
    ):
        r"""
        ### Purpose:
        - To remove a set of constraints from the model by their name
        """
        remove_constraints(
            constraint_config_dict=constraint_config_dict,
            model=self.model,
        )


def set_catheter_variables(
    plan: BrachyPlan,
    model: Model,
    catheter_vars_to_keep: List[CatheterVar_Gurobi] = None,
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
        catheter_vars_to_keep = []
        name_cath_to_keep = []
    for catheter in tqdm(
        plan.catheter_table,
        total=len(plan.catheter_table.catheters_list),
        desc="Creating optimization variables from new catheters"):
        if f"catheter_{catheter.name_id}" in name_cath_to_keep:
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
    return catheter_vars_to_keep + catheter_vars_to_add


def _remove_variables_by_prefix(model: Model, prefix: str) -> bool:
    r"""
    ### Purpose:
    - Removes every decision variable whose name matches `prefix` exactly (scalar Var) or is an
    entry of an MVar created with that base name (i.e. `prefix[0]`, `prefix[1]`, ...). Gurobi does
    NOT discard/overwrite existing variables when you `addVar`/`addMVar` again with the same name -
    it silently creates duplicates - so this must be called explicitly before rebuilding.
    ### Inputs:
    - `model`: Model := the Gurobi model to clean up.
    - `prefix`: str := the base variable name (without the `[index]` suffix) to remove.
    ### Output:
    - bool := True if any variable was removed (caller should then call `model.update()`).
    """
    to_remove = [
        v for v in model.getVars()
        if v.VarName == prefix or v.VarName.startswith(prefix + "[")
    ]
    if to_remove:
        model.remove(to_remove)
        return True
    return False


def _remove_constraints_by_prefix(model: Model, prefix: str) -> bool:
    r"""
    ### Purpose:
    - Removes every constraint whose name matches `prefix` exactly or is an entry of a vectorized
    constraint created with that base name (i.e. `prefix[0]`, `prefix[1]`, ...). Checks the quadratic
    constraint list (`getQConstrs`), since the `c_L_*`/`c_U_*`/`c_H_*` constraints built from
    `c_MVar * t_MVar` are quadratic (`MQConstr`/`QConstr`) even though they look linear at a glance.
    ### Inputs:
    - `model`: Model := the Gurobi model to clean up.
    - `prefix`: str := the base constraint name (without the `[index]` suffix) to remove.
    ### Output:
    - bool := True if any constraint was removed (caller should then call `model.update()`).
    """
    to_remove_quad = [
        c for c in model.getQConstrs()
        if c.QCName == prefix or c.QCName.startswith(prefix + "[")
    ]
    removed = False
    if to_remove_quad:
        model.remove(to_remove_quad)
        removed = True
    return removed


def _cleanup_penalty_terms_for_structure(model: Model, structure_name: str):
    r"""
    ### Purpose:
    - Removes all slack variables and constraints previously created (by a prior call of
    `set_penalty_function_and_constraints`) for a given structure, so that rebuilding the penalty
    terms for that structure does not create duplicate variables/constraints under the same name.
    Covers both the target-structure naming (`p_L_`, `p_U_`, `c_L_`, `c_U_`) and the OAR naming
    (which reuses `p_L_`/`c_L_` since a structure is either a target or an OAR, never both).
    ### Inputs:
    - `model`: Model := the Gurobi model to clean up.
    - `structure_name`: str := the structure whose slack variables/constraints should be purged.
    ### Output:
    - None: the model is updated with the removals flushed (via `model.update()`).
    """
    removed = False
    removed |= _remove_variables_by_prefix(model, f"p_L_{structure_name}")
    removed |= _remove_variables_by_prefix(model, f"p_U_{structure_name}")
    removed |= _remove_constraints_by_prefix(model, f"c_L_{structure_name}")
    removed |= _remove_constraints_by_prefix(model, f"c_U_{structure_name}")
    if removed:
        model.update()


def _cleanup_hotspot_penalty_terms(model: Model):
    r"""
    ### Purpose:
    - Removes all hotspot slack variables/constraints (`p_H_*`/`c_H_*`) previously created, since
    the hotspot estimator naming is not keyed per-structure (only one target structure carries the
    hotspot estimator masks).
    """
    to_remove_vars = [v for v in model.getVars() if v.VarName.startswith("p_H_")]
    to_remove_constrs = [c for c in model.getQConstrs() if c.QCName.startswith("c_H_")]
    if to_remove_vars:
        model.remove(to_remove_vars)
    if to_remove_constrs:
        model.remove(to_remove_constrs)
    if to_remove_vars or to_remove_constrs:
        model.update()


def _get_mvar_by_name_indices(model: Model, prefix: str, n: int) -> Optional[MVar]:
    r"""
    ### Purpose:
    - Resolves `n` variables named `f"{prefix}[0]"` ... `f"{prefix}[{n-1}]"` into a live `MVar`, by
    generating the exact names on the fly and calling `model.getVarByName` for each (Gurobi has no
    bulk/pattern "get MVar by name" API - this per-name lookup, then `MVar.fromlist(...)`, is the
    standard idiom). Each `getVarByName` call is an O(1) hash-table lookup once the model has been
    `update()`-d, so this stays cheap even for structures with thousands of dose points.
    ### Inputs:
    - `model`: Model := the model to resolve variables on.
    - `prefix`: str := the base variable name (e.g. `f"p_L_{structure_name}"`).
    - `n`: int := how many indices to resolve (0 .. n-1).
    ### Output:
    - Optional[MVar] := the resolved `MVar`, or `None` if `n == 0` or any expected variable is
    missing (meaning this slack group was never created, e.g. its weight was 0 at build time).
    """
    if n <= 0:
        return None
    variables = [model.getVarByName(f"{prefix}[{i}]") for i in range(n)]
    if any(v is None for v in variables):
        return None
    return MVar.fromlist(variables)

def _create_hotspot_slacks_and_constraints(
    optimization_config: Optimization_Config,
    A_sparse: np.ndarray,
    c_MVar: MVar,
    t_MVar: MVar,
    model: Model,
):
    r"""
    ### Purpose:
    - Creates the hotspot slack variables and constraints (target structures only) for one
    `Optimization_Config`, one entry per hotspot mask. Only the Gurobi objects are created here (with
    a throwaway RHS of 0) - `update_penalty_weights_and_voxel_goals` re-derives their names from
    `optimization_config` and resolves/updates them by name afterwards, so nothing is returned here.
    As a side effect, caches `optimization_config.hotspot_num_dose_points` (one entry per hotspot
    mask, in the same order as `optimization_config.hotspot_masks`) so the voxel count - and hence
    the exact variable/constraint names - never has to be recomputed from the mask arrays again.
    """
    hotspot_num_dose_points = []
    for hotspot_mask in optimization_config.hotspot_masks:
        mask_array = hotspot_mask.imageArray.flatten()
        target_array = optimization_config.mask.imageArray.flatten()
        mask_in_target = np.ma.multiply(mask_array, target_array)[target_array].astype(bool)
        num_dose_points = int(np.sum(mask_in_target))
        hotspot_num_dose_points.append(num_dose_points)

        hotspot_suffix = hotspot_mask.name.split("hotspot_estimator_")[1]
        x_slack_hotspot = model.addMVar(
            shape=num_dose_points,
            name=f"p_H_{hotspot_suffix}")
        # Gotta filter out only the expressions that apply to hotspots.
        # so setting them to zero is not enough, we need to isolate them!
        dose_expression = A_sparse[mask_in_target, :] @ (c_MVar * t_MVar)
        model.addConstr(
            dose_expression - x_slack_hotspot <= np.zeros(num_dose_points),
            name=f"c_H_{hotspot_suffix}")

    optimization_config.hotspot_num_dose_points = hotspot_num_dose_points


def set_penalty_function_and_constraints(
    optimization_configs: List[Optimization_Config],
    dwellTimeVariables: List[DwellTime_Gurobi],
    catheter_vars: List[CatheterVar_Gurobi],
    model: Model,
    cleanup: bool = True,
):
    r"""
    ### Purpose:
    - Creates the slack variables and (quadratic) dose constraints for every structure, then
    delegates ALL weight/target-dose/objective logic to `update_penalty_weights_and_voxel_goals` - by
    name, exactly like any other call to it. This function's only unique responsibility is deciding
    WHICH slack variables need to exist (a structural decision, gated on `penalty_weight_* > 0` at
    build time) and creating them with the correct NAMES - Gurobi's `addMVar`/`addConstr` require some
    concrete bound/RHS value at call time, so throwaway placeholders (`GRB.INFINITY` bounds, RHS of 0)
    are used; these are immediately overwritten by the trailing call to
    `update_penalty_weights_and_voxel_goals`. Nothing returned by `addMVar`/`addConstr` is kept - not
    even locally - since every subsequent read/write goes through name-based lookup.

    Note: `A_sparse @ (c_MVar * t_MVar)` is bilinear in the decision variables, so every dose
    constraint built here (`c_L_*`, `c_U_*`, `c_H_*`) is a QUADRATIC constraint (`QConstr`), not a
    linear one - its right-hand side attribute is `QCRHS`, not `RHS`, and it must be looked up with
    `getQConstrByName`.

    ### Inputs:
    - `optimization_configs`: List[Optimization_Config] := List of optimization configs containing the
    penalty weights, target dose, mask, dwell_coef_dict and other attributes. As a side effect, this
    function sets `optimization_config.num_dose_points` (and `.hotspot_num_dose_points`, for target
    structures with hotspot masks) on each config - these two fields are what let
    `update_penalty_weights_and_voxel_goals` regenerate variable/constraint names without ever seeing
    `A_sparse` or the mask arrays again.
    - `dwellTimeVariables`: The list of the dwell times variables in the catheter table. it should be
    synched up with catheter_vars.
    - `catheter_vars`: List[CatheterVar_Gurobi] := the catheter variables to be used in the optimization.
    - `model`: Model := the Gurobi model to which the variables will be added.
    - `cleanup`: bool := if True (default), removes any slack variables/constraints previously created
    for the structures being (re)processed here, before adding the new ones. Gurobi does not do this
    automatically: calling `addVar`/`addConstr` with a name that already exists creates a *second*
    object rather than replacing the first.
    ### Output:
    - None: the model's variables/constraints/objective are created and set to match the current
    `optimization_configs` values.
    """
    t_MVar = MVar([dt._model_variable for dt in dwellTimeVariables])
    c_MVar = MVar([c._model_variable for c in catheter_vars for _ in c])

    hotspot_cleanup_done = False

    for optimization_config in tqdm(optimization_configs):
        structure_name = optimization_config.structure_name
        print(f"Setting constraints and penalty terms for structure {structure_name}")
        if not optimization_config.dwell_coef_dict:
            raise ValueError("The coefficint dictionary is empty. \
please run set_dwell_coef_dict_per_structure")

        for dt, dt_name in zip(t_MVar, list(optimization_config.dwell_coef_dict.keys())):
            if dt.VarName != dt_name:
                raise ValueError("The order of the dwell times is not matching the order \
of the corresponding dose rate coefficients.")

        if cleanup:
            _cleanup_penalty_terms_for_structure(model, structure_name)
            if optimization_config.penalty_weight_hotspot > 0 and not hotspot_cleanup_done:
                _cleanup_hotspot_penalty_terms(model)
                hotspot_cleanup_done = True

        A_sparse = np.column_stack(list(optimization_config.dwell_coef_dict.values()))
        num_dose_points = A_sparse.shape[0]
        optimization_config.num_dose_points = num_dose_points
        if num_dose_points == 0:
            continue

        if optimization_config.is_target:
            if optimization_config.penalty_weight_linear > 0 or optimization_config.penalty_weight_quadratic > 0:
                x_slack = model.addMVar(
                    shape=num_dose_points, lb=0.0, ub=GRB.INFINITY, name=f"p_L_{structure_name}")
                model.addConstr(
                    A_sparse @ (c_MVar * t_MVar) + x_slack >= np.zeros(num_dose_points),
                    name=f"c_L_{structure_name}")

            if optimization_config.penalty_weight_uniformity > 0:
                y_uniform = model.addMVar(
                    shape=num_dose_points, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_U_{structure_name}")
                model.addConstr(
                    A_sparse @ (c_MVar * t_MVar) + y_uniform == np.zeros(num_dose_points),
                    name=f"c_U_{structure_name}")

            if optimization_config.penalty_weight_hotspot > 0:
                _create_hotspot_slacks_and_constraints(
                    optimization_config=optimization_config,
                    A_sparse=A_sparse,
                    c_MVar=c_MVar,
                    t_MVar=t_MVar,
                    model=model,
                )
        # OAR constraints
        else:
            if optimization_config.penalty_weight_linear > 0 or optimization_config.penalty_weight_quadratic > 0:
                x_slack_oar = model.addMVar(
                    shape=num_dose_points, lb=0.0, ub=GRB.INFINITY, name=f"p_L_{structure_name}")
                model.addConstr(
                    A_sparse @ (c_MVar * t_MVar) - x_slack_oar <= np.zeros(num_dose_points),
                    name=f"c_L_{structure_name}")

    model.update()  # flush additions so update_penalty_weights_and_voxel_goals can find them by name
    update_penalty_weights_and_voxel_goals(
        model=model,
        optimization_configs=optimization_configs,
    )

_QCONSTR_NAME_RE = re.compile(r"^(c_L|c_U|c_H)_(.+)\[(\d+)\]$")


def update_penalty_weights_and_voxel_goals(
    model: Model,
    optimization_configs: List["Optimization_Config"],
):
    r"""
    ### Purpose:
    - Sets penalty weights and per-voxel target dose for every structure in `optimization_configs`,
    reading every numeric value (weights, `dose_voxel_goal`, `min_dose`, `max_dose`, `hotspot_threshold`,
    `num_dose_points`, `hotspot_num_dose_points`, `hotspot_masks`) straight from each
    `Optimization_Config`. NO handles, NO blueprint, NO cached Gurobi object references anywhere:
    every variable/constraint name is regenerated on the fly from `structure_name` + `num_dose_points`
    (and, for hotspots, each hotspot mask's name-derived suffix + its own voxel count), then resolved
    via `model.getVarByName` / a single pass over `model.getQConstrs()`.
    - Never calls `addVar`/`addConstr`/`remove`: only `Obj`, `QCRHS` and `UB` attributes are mutated,
    and the objective expression is rebuilt from the resolved `MVar`s. If a structure's weight was 0
    when `set_penalty_function_and_constraints` ran (so its slack variables were never created), the
    corresponding name lookup returns `None` and that term is silently skipped - it cannot be added
    retroactively without calling `set_penalty_function_and_constraints` again.

    ### Performance note:
    - The original implementation resolved quadratic constraints per-structure by scanning
    `model.getQConstrs()` once per structure (`O(num_structures * num_qconstrs)`). This version
    scans `model.getQConstrs()` exactly once for the whole model, parses each constraint's name to
    determine its type (`c_L`/`c_U`/`c_H`) and owning structure/hotspot-suffix, and drops it into a
    preallocated slot list resolved via `O(1)` dict lookups. Total constraint-resolution cost is now
    `O(num_qconstrs + num_structures)` instead of `O(num_structures * num_qconstrs)`. Variable
    lookups still go through `model.getVarByName` (already `O(1)` per call once the model has been
    `update()`-d), so `_get_mvar_by_name_indices` is unchanged.

    IMPORTANT: `c_L_*`/`c_U_*`/`c_H_*` are QUADRATIC constraints (bilinear `c_MVar * t_MVar` term), so
    their RHS is read/written via `QCRHS` (not `RHS`). RHS updates use the bulk
    `model.setAttr("QCRHS", constr_list, values_list)` form.

    ### Inputs:
    - `model`: Model := the Gurobi model whose variables/constraints follow the naming convention
    established by `set_penalty_function_and_constraints`.
    - `optimization_configs`: List[Optimization_Config] := the current optimization configs. Any of
    `penalty_weight_linear`, `penalty_weight_quadratic`, `penalty_weight_uniformity`,
    `penalty_weight_hotspot`, `hotspot_threshold`, `penalty_weight_variance_time`, `dose_voxel_goal`,
    `min_dose`, `max_dose` can be changed freely between calls.
    ### Output:
    - None: `model`'s objective, quadratic constraint QCRHS values, and variable bounds are updated
    in place. Call `model.optimize()` afterwards as usual.
    """
    penalty_terms = {
        "linear": 0,
        "quadratic": 0,
        "uniformity": 0,
        "hotspot": 0,
        "dwelltimes": 0,
    }

    # All dwell-time variables share the "dwell_" name prefix (see CatheterVar_Gurobi). A single scan
    # of the model's variables is enough to rebuild t_MVar - order doesn't matter here since it only
    # feeds a mean/variance computation.
    dwell_vars = [v for v in model.getVars() if v.VarName.startswith("dwell_")]
    t_MVar = MVar.fromlist(dwell_vars) if dwell_vars else None

    # --- Pass 1 over configs: build lookup structures + preallocate constraint slot lists ---
    configs_by_structure = {}
    # structure_name -> {"L": [None]*n, "U": [None]*n or None}
    constr_slots = {}
    # hotspot_suffix -> [None]*hs_num
    hotspot_slots = {}

    for optimization_config in optimization_configs:
        structure_name = optimization_config.structure_name
        num_dose_points = optimization_config.num_dose_points
        if not num_dose_points:
            continue

        configs_by_structure[structure_name] = optimization_config
        constr_slots[structure_name] = {
            "L": [None] * num_dose_points,
            "U": [None] * num_dose_points if optimization_config.is_target else None,
        }

        if optimization_config.is_target:
            hotspot_masks = optimization_config.hotspot_masks or []
            hotspot_num_dose_points = optimization_config.hotspot_num_dose_points or []
            for hotspot_mask, hs_num in zip(hotspot_masks, hotspot_num_dose_points):
                if not hs_num:
                    continue
                hotspot_suffix = hotspot_mask.name.split("hotspot_estimator_")[1]
                hotspot_slots[hotspot_suffix] = [None] * hs_num

    # --- Pass 2: single scan over all quadratic constraints in the model ---
    for constraint in model.getQConstrs():
        match = _QCONSTR_NAME_RE.match(constraint.QCName)
        if match is None:
            continue
        ctype, name_part, idx_str = match.groups()
        idx = int(idx_str)

        if ctype in ("c_L", "c_U"):
            slots = constr_slots.get(name_part)
            if slots is None:
                continue
            slot_list = slots["L"] if ctype == "c_L" else slots["U"]
            if slot_list is not None and idx < len(slot_list):
                slot_list[idx] = constraint
        else:  # ctype == "c_H"
            slot_list = hotspot_slots.get(name_part)
            if slot_list is not None and idx < len(slot_list):
                slot_list[idx] = constraint

    # --- Pass 3 over configs: resolve variables (O(1) getVarByName calls) and build the objective ---
    for structure_name, optimization_config in configs_by_structure.items():
        num_dose_points = optimization_config.num_dose_points
        min_dose = optimization_config.min_dose
        max_dose = optimization_config.max_dose
        voxel_goal = optimization_config.dose_voxel_goal
        voxel_goal_vec = np.full(num_dose_points, voxel_goal)

        linear_weight = optimization_config.penalty_weight_linear
        quadratic_weight = optimization_config.penalty_weight_quadratic
        uniformity_weight = optimization_config.penalty_weight_uniformity
        penalty_weight_variance_time = optimization_config.penalty_weight_variance_time
        penalty_weight_hotspot = optimization_config.penalty_weight_hotspot
        hotspot_threshold = optimization_config.hotspot_threshold

        slots = constr_slots[structure_name]
        constr_L_raw = slots["L"]
        constr_L = constr_L_raw if all(c is not None for c in constr_L_raw) else None

        # p_L_/c_L_ is shared naming for both the target dose-slack and the OAR dose-slack (a
        # structure is either a target or an OAR, never both, so there is no name collision).
        x_slack = _get_mvar_by_name_indices(model, f"p_L_{structure_name}", num_dose_points)

        if optimization_config.is_target:
            if x_slack is not None and constr_L is not None:
                x_slack.UB = np.full(num_dose_points, voxel_goal - min_dose)
                model.setAttr("QCRHS", constr_L, list(voxel_goal_vec))

                if linear_weight > 0:
                    penalty_terms["linear"] += (linear_weight / num_dose_points) * x_slack.sum()
                if quadratic_weight > 0:
                    penalty_terms["quadratic"] += (quadratic_weight / num_dose_points) * (x_slack @ x_slack)

            constr_U_raw = slots["U"]
            constr_U = constr_U_raw if constr_U_raw is not None and all(c is not None for c in constr_U_raw) else None
            y_uniform = _get_mvar_by_name_indices(model, f"p_U_{structure_name}", num_dose_points)
            if y_uniform is not None and constr_U is not None:
                y_uniform.UB = np.full(num_dose_points, voxel_goal - min_dose)
                model.setAttr("QCRHS", constr_U, list(voxel_goal_vec))

                uniformity_coeff = uniformity_weight / num_dose_points * 1e-3
                penalty_terms["uniformity"] += uniformity_coeff * (y_uniform @ y_uniform)

            # --- hotspot penalty (target structures only, one term per hotspot mask) ---
            hotspot_masks = optimization_config.hotspot_masks or []
            hotspot_num_dose_points = optimization_config.hotspot_num_dose_points or []
            for hotspot_mask, hs_num in zip(hotspot_masks, hotspot_num_dose_points):
                if not hs_num:
                    continue
                hotspot_suffix = hotspot_mask.name.split("hotspot_estimator_")[1]
                constr_H_raw = hotspot_slots.get(hotspot_suffix)
                constr_H = constr_H_raw if constr_H_raw is not None and all(c is not None for c in constr_H_raw) else None
                x_slack_hotspot = _get_mvar_by_name_indices(model, f"p_H_{hotspot_suffix}", hs_num)
                if x_slack_hotspot is None or constr_H is None:
                    continue

                model.setAttr("QCRHS", constr_H, [voxel_goal * hotspot_threshold] * hs_num)
                if penalty_weight_hotspot > 0:
                    hotspot_coeff = penalty_weight_hotspot / hs_num
                    penalty_terms["hotspot"] += hotspot_coeff * x_slack_hotspot.sum()

            # --- dwell-time-variance penalty (target structures only, no slack vars involved) ---
            if penalty_weight_variance_time > 0 and t_MVar is not None:
                mean_dwell_time = sum(t_MVar) / t_MVar.size
                penalty_terms["dwelltimes"] += (
                    penalty_weight_variance_time * 1e-3
                    * ((t_MVar - mean_dwell_time) @ (t_MVar - mean_dwell_time)) / t_MVar.size
                )
        else:
            if x_slack is not None and constr_L is not None:
                x_slack.UB = np.full(num_dose_points, max_dose - min_dose)
                model.setAttr("QCRHS", constr_L, list(voxel_goal_vec))

                if linear_weight > 0:
                    penalty_terms["linear"] += (linear_weight / num_dose_points) * x_slack.sum()
                if quadratic_weight > 0:
                    penalty_terms["quadratic"] += (quadratic_weight / num_dose_points) * (x_slack @ x_slack)

    model.setObjective(
        penalty_terms["linear"]
        + penalty_terms["quadratic"]
        + penalty_terms["uniformity"]
        + penalty_terms["hotspot"]
        + penalty_terms["dwelltimes"],
        GRB.MINIMIZE)
    model.update()


def set_dwell_coef_dict_per_structure(
    plan: BrachyPlan,
    dwellTimeVariables: List[DwellTime_Gurobi],
    optim_roi_bounds: List[List[float]] = None,
    multi_processing: bool = False,
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
        structure_mask = resample_crop_the_mask_or_contour_to_optimGrid(
            structure_mask=structure.mask,
            template_dose_obj=plan.combined_dose,
            optim_spacing=structure.optimization_config.spacing_mm,
            roi_bounds=optim_roi_bounds,
        )

        structure.optimization_config.mask = structure_mask
        if structure.optimization_config.hotspot_masks is not None:
            # consider multi processing this later
            hotspot_masks = []
            for hotspot_mask in structure.optimization_config.hotspot_masks:
                hotspot_mask_resampled = resample_crop_the_mask_or_contour_to_optimGrid(
                    structure_mask=hotspot_mask,
                    template_dose_obj=plan.combined_dose,
                    optim_spacing=structure.optimization_config.spacing_mm,
                    roi_bounds=optim_roi_bounds,
                )

                hotspot_masks.append(hotspot_mask_resampled)
            structure.optimization_config.hotspot_masks = hotspot_masks

        # Build dose rate matrix and dwell time vector for this structure
        dwell_vars, dose_rate_matrices = compute_dose_rate_matrices(
            dwellTimeVariables,
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


def remove_constraints(
    constraint_config_dict: Dict[str, Constraint_Config],
    model: Model,
):
    r"""
    ### Purpose:
    - To remove a set of constraints from the model by their name
    """
    for name_id in constraint_config_dict:
        # check if the constraint already exists, if yes remove it
        try:
            old_constraint = model.getConstrByName(name=name_id)
        except GurobiError:
            old_constraint = None
            print(f"Constraint {name_id} was not found in the model.")
        if old_constraint:
            model.remove(old_constraint)
    model.update()


def set_constraints(
    constraint_config_dict: Dict[str, Constraint_Config],
    model: Model,
):
    """
    ### Purpose:
    - To update the model with the new constraints. The type of the constraints can be
    "bound", "sum", "uniqueness", "continuity", "num_catheters", "collision", each is explained in
    Constraint_Config class. The target variables for the constraints can be "catheter" or "dwell".
    - Each constraint should have a unique name generated automatically by the Constraint_Config class.

    ### Inputs:
    - constraint_configs Dict[str, Constraint_Config]: See Constraint_Config for details.
    The key of this dictionary is the name of the constraint, which should be unique.
    - model (Model): The model containing the variables. The name of the variables in the constraint list
    should match the name of the variable. Otherwies, Error will be thrown.

    ### Outputs:
    - None: model is updated with the new constraints
    """
    for name_id, constraint in constraint_config_dict.items():
        # check if the constraint already exists, if yes remove it
        try:
            old_constraint = model.getConstrByName(name=name_id)
        except GurobiError:
            old_constraint = None
        if old_constraint:
            model.remove(old_constraint)
    model.update()

    for name_id, constraint in constraint_config_dict.items():
        if constraint.constraint_type == "bound":
            _set_bound_constraint(constraint, model)
        elif constraint.constraint_type == "sum":
            _set_sum_constraint(constraint, model)
        elif constraint.constraint_type == "uniqueness":
            _set_uniqueness_constraint(constraint, model)
        elif constraint.constraint_type == "num_catheters":
            _set_num_catheters_constraint(constraint, model)
        elif constraint.constraint_type == "continuity":
            _set_continuity_constraint(constraint, model)
        elif constraint.constraint_type == "collision":
            _set_collision_constraint(constraint, model)
    model.update()


def _set_bound_constraint(
    constraint: Constraint_Config,
    model: Model,
):
    r"""
    ### Purpose:
    - To set a bound constraint on a variable in the model.
    The variable can be either a catheter or dwell time variable.
    """
    var_name = f"{constraint.variable_type}_{constraint.variable_name_ids[0]}"
    variable = model.getVarByName(var_name)
    if not variable:
        raise ValueError(f"No variable with name {var_name} was found for constraint \
{constraint.name_id}. Ensure the constraint name is correct.")
    if constraint.minimum is not None:
        variable.LB = constraint.minimum
    if constraint.maximum is not None:
        variable.UB = constraint.maximum
    if constraint.equal is not None:
        model.addConstr(
            variable == constraint.equal,
            name=constraint.name_id)


def _set_sum_constraint(
    constraint: Constraint_Config,
    model: Model,
):
    r"""
    ### Purpose:
    - To set a sum constraint on a list of variables in the model.
    The variables can be either catheter or dwell time variables.
    """
    var_names = [f"{constraint.variable_type}_{name_id}" for name_id in constraint.variable_name_ids]
    variables = [model.getVarByName(var_name) for var_name in var_names]
    if not all(variables):
        missing_vars = [var_name for var_name, var in zip(var_names, variables) if not var]
        raise ValueError(f"No variable(s) with name(s) {missing_vars} were found for constraint {constraint.name_id}. \
Ensure the constraint name is correct.")
    if constraint.minimum is not None:
        model.addConstr(
            sum(variables) >= constraint.minimum,
            name=f"{constraint.name_id}")
    if constraint.maximum is not None:
        model.addConstr(
            sum(variables) <= constraint.maximum,
            name=f"{constraint.name_id}")
    if constraint.equal is not None:
        model.addConstr(
            sum(variables) == constraint.equal,
            name=constraint.name_id)


def _set_uniqueness_constraint(
    constraint: Constraint_Config,
    model: Model,
):
    r"""
    ### Purpose:
    - To set a uniqueness constraint on a list of variables in the model.
    The variables can be either catheter or dwell time variables.
    """
    if constraint.constraint_type != "uniqueness":
        raise ValueError(f"Wrong constraint type, the current type for {constraint.name_id}")
    var_names = [f"{constraint.variable_type}_{name_id}" for name_id in constraint.variable_name_ids]
    variables = [model.getVarByName(var_name) for var_name in var_names]
    if not all(variables):
        missing_vars = [var_name for var_name, var in zip(var_names, variables) if not var]
        raise ValueError(f"No variable(s) with name(s) {missing_vars} were found for constraint {constraint.name_id}. \
Ensure the constraint name is correct.")
    model.addConstr(
        sum(variables) <= constraint.maximum,
        name=constraint.name_id)


def _set_num_catheters_constraint(
    constraint: Constraint_Config,
    model: Model,
):
    r"""
    ### Purpose:
    - To set a constraint on the number of catheters in the model.
    The variables can be either catheter or dwell time variables.
    """
    var_names = [f"{constraint.variable_type}_{name_id}" for name_id in constraint.variable_name_ids]
    variables = [model.getVarByName(var_name) for var_name in var_names]
    if not all(variables):
        missing_vars = [var_name for var_name, var in zip(var_names, variables) if not var]
        raise ValueError(f"No variable(s) with name(s) {missing_vars} were found for constraint {constraint.name_id}. \
Ensure the constraint name is correct.")
    if constraint.equal is not None:
        model.addConstr(
            sum(variables) == constraint.equal,
            name=constraint.name_id)
    else:
        model.addRange(
            sum(variables),
            constraint.minimum,
            constraint.maximum,
            name=constraint.name_id)


def _set_continuity_constraint(
    constraint: Constraint_Config,
    model: Model,
):
    r"""
    ### Purpose:
    - To set a continuity constraint on a list of variables in the model.
    The variables can be either catheter or dwell time variables.
    """
    var_names = [f"{constraint.variable_type}_{name_id}" for name_id in constraint.variable_name_ids]
    parent_var_names = [f"{constraint.variable_type}_{name_id}" for name_id in constraint.parent_catheter_name_ids]
    variables = [model.getVarByName(var_name) for var_name in var_names]
    if not all(variables):
        missing_vars = [var_name for var_name, var in zip(var_names, variables) if not var]
        raise ValueError(f"No variable(s) with name(s) {missing_vars} were found for constraint {constraint.name_id}. \
Ensure the constraint name is correct.")
    parent_variables = [model.getVarByName(var_name) for var_name in parent_var_names]
    if not all(parent_variables):
        missing_parent_vars = [var_name for var_name, var in zip(parent_var_names, parent_variables) if not var]
        raise ValueError(f"No parent variable(s) with name(s) {missing_parent_vars} were found for constraint {constraint.name_id}. \
Ensure the constraint name is correct.")

    # All variables have the same parent.
    e_vec = model.addMVar(
        shape=len(variables),
        lb=0,
        ub=1,
        name=f"e_{constraint.name_id}",
        vtype=GRB.BINARY,)
    var_vec = MVar.fromlist(variables)
    sum_parents = sum(parent_variables)
    for i in range(len(variables)):
        model.addConstr(
            e_vec[i] * sum_parents == constraint.equal * var_vec[i],
            name=constraint.name_id,)


def _set_collision_constraint(
    constraint: Constraint_Config,
    model: Model,
):
    r"""
    ### Purpose:
    - To set a collision constraint on a list of variables in the model.
    The variables can be either catheter or dwell time variables.
    """
    var_names = [f"{constraint.variable_type}_{name_id}" for name_id in constraint.variable_name_ids]
    variables = [model.getVarByName(var_name) for var_name in var_names]
    if not all(variables):
        missing_vars = [var_name for var_name, var in zip(var_names, variables) if not var]
        raise ValueError(f"No variable(s) with name(s) {missing_vars} were found for constraint {constraint.name_id}. \
Ensure the constraint name is correct.")
    model.addConstr(
        sum(variables) <= constraint.equal,
        name=constraint.name_id)


def get_optimization_result_stats(
    catheter_table_optim: CatheterTableOptim_Gurobi,
) -> Dict[str, dict]:
    r"""
    ### Purpose:
    - To get all sorts of optimization result from the optimized plan.

    ### Inputs:
    - catheter_table_optim: CatheterTableOptim_Gurobi := the optimization object that has the optimized plan.

    ### Outputs:
    - A dictionary with the following information:
    - dvh_metrics: Dict[str, dict] := a dictionary containing the DVH metrics for each structure in the plan.
    - num_voxels_optimized: Dict[str, int] := a dictionary containing the number of voxels used for
    optimization in each structure.
    - catheter_table_stats: Dict[str, float] := a dictionary containing mean, std, median and IQR of dwell times
    number of dwell positions and catheters as well as fraction of used dwell positions and catheters.
    - optimization_timing: Dict[str, float] := a dictionary containing the build time and solve time for the model.
    """
    dvh_metrics = catheter_table_optim.plan.get_dvh_metrics()
    catheter_table_stats = catheter_table_optim.plan.catheter_table.get_stats()
    num_voxels_optimized = {
        optim_config.structure_name: optim_config.num_dose_points
        for optim_config in catheter_table_optim.plan.optimization_config_dict.values()
    }
    optimization_timing = {
        "build_time": catheter_table_optim.build_time,
        "solve_time": catheter_table_optim.solve_time,
    }
    return {
        "dvh_metrics": dvh_metrics,
        "catheter_table_stats": catheter_table_stats,
        "num_voxels_optimized": num_voxels_optimized,
        "optimization_timing": optimization_timing,
    }
