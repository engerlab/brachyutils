# from abc import ABC, abstractmethod
from typing import List, Callable, Any
from copy import deepcopy
from brachyutils.dose.dose_utils import BrachyDose
from pydantic import BaseModel, PrivateAttr
import numpy as np
from pathlib import Path
from opentps.core.data.images import ROIMask
from opentps.core.data import ROIContour
from gurobipy import Model, Var, GRB, Constr
# from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.types import BrachyPlan
class Optimization_Config(BaseModel):
    """
    ### Purpose:
    - This class holds the information regarding the optimization configuration per each structure.
    When loading the plan the optimization config is created for each structure in the plan.structure_list.

    ### Attributes:
    - structure_name: str := The name of the structure to which this optimization config applies.
    - spacing_mm: List[float] | float := The spacing of the optimization grid in mm. 
    - dose_voxel_goal: float := The dose goal for the structure in Gy.
    - penalty_weight_linear: float := Weight for linear penalty term in objective function. Default 1.
    - penalty_weight_quadratic: float := Weight for quadratic penalty term. Default 1.
    - penalty_weight_hotspot: float := Weight for hotspot penalty term. Default 0.
    - hotspot_threshold: float := If the average dose to the hot spot estimator volume goes above (target_dose * hotspot_threshold),
    penalty will be calculated for that hot spot estimator volume. Default 0.
    - penalty_weight_uniformity: float := Weight for dose uniformity penalty. Default 1.
    - mask_margin_mm: List[float] | float := Margin around structure for optimization in mm. Default 0.
    - min_dose: float := Minimum allowed dose in Gy. Default 0.
    - max_dose: float := Maximum allowed dose in Gy. Default 500.
    """
    structure_name:str = None
    spacing_mm:float | List[float]= None
    dose_voxel_goal:float = None
    penalty_weight_linear:float = 1
    penalty_weight_quadratic:float = 1
    penalty_weight_hotspot:float = 0
    hotspot_threshold:float = 0
    penalty_weight_uniformity:float = 1
    mask_margin_mm:float | List[float]= 0
    min_dose:float = 0
    max_dose:float = 500
    # may be needed later
    # self.index_range_constraints: List[int] = None

class DwellTimeVariable(BaseModel):
    """
    ### Purpose:
    - A class to represent a DwellTimeVariable in the dwell time optimization problem.
    ### Attributes:
    - name:str := references the catheter_number and dwell position number in the format
    catheter_{catheter_number}_dwell_{dwell_position_number}
    - dwell_time:float := The initial dwell_time of the DwellTimeVariable.
    - lower_bound:float := The lower bound of the DwellTimeVariable.
    - upper_bound:float := The upper bound of the DwellTimeVariable.
    - coordinates:List[float] := The coordinates of the dwell position for this DwellTimeVariable.
    - model_variable: Any := The variable in the optimization model corresponding to this DwellTimeVariable.
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "defer_build": True
        }

    name: str
    dwell_time: float = None
    lower_bound: float = None
    upper_bound: float = None
    coordinates: List[float] = None
    dose_rate_map: np.ndarray = None
    model_variable: Var = None
    def __init__(self, model: Any, **data):
        """
        ### Purpose:
        - A function to initialize the DwellTimeVariable.
        ### Inputs:
        - model: Any := The model object.
        - name:str := references the catheter_number and dwell position number in the format
        catheter_{catheter_number}_dwell_{dwell_position_number}
        - dwell_time:float := The initial dwell_time of the DwellTimeVariable.
        - lower_bound:float := The lower bound of the DwellTimeVariable.
        - upper_bound:float := The upper bound of the DwellTimeVariable.
        - coordinates:List[float] := The coordinates of the dwell position for this DwellTimeVariable.
        """
        super().__init__(**data)
        if isinstance(model, Model):
            self.model_variable = model.addVar(
                lb=self.lower_bound,
                ub=self.upper_bound,
                name=self.name,
                vtype=GRB.CONTINUOUS
            )
            model.update()
        else:
            raise ValueError("Model is not a Gurobi model. Please provide a Gurobi model.")

class DwellTimeOptimizer(BaseModel):
    r"""
    ### Purpose:
    - An abstract dwell time optimizer class to specify the common components of a dwell time optimizer class that
    easily integrates to BrachyUtils.
    ### Attributes:
    - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
    - dwellTimeVariables: The set of the dwellTimeVariables to be optimized. In HDR brachy, dwell times and catheter positions
    - constraints: A set of relationships between the dwellTimeVariables that should not be violated.
    In HDR brachy, we want all dwell times to be positive and sometimes have upper or lower bounds.
    - penalty_function: A function that states how good a set of dwellTimeVariables are.
    - solver:str := The name
    - model: The object that incorporates all the attributes above to output the optimal 
    dwell_time for each DwellTimeVariable.
    - roi_bounds: The coordinate bounds for the optimization region of interest (roi) from the plan.
    to consider voxels the dose rate maps. for each axis: 
    roi_bounds = [first dwell - margin : last dwell + margin]
    ### Functions:
    - get_model()
    - run()
    """
    model_config = {
        "arbitrary_types_allowed": True,
        # "defer_build": False
        }
    plan: Any
    solver: str = None
    dwellTimeVariables: List[Var] = None
    penalty_function: Callable = None
    model: Any = None
    roi_bounds: List[List[float]] = None # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    def __init__(
        self,
        roi_margin_mm: List[float] | float = 5.0,
        solver="gurobi",
        **data):
        r"""
        ### Purpose:
        - A function to initialize the optimizer.
        ### Parameters:
        - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
        - roi_margin_mm: The distance from the furthest dwell position along each axis
        """
        super().__init__(**data)
        roi_margin_mm = roi_margin_mm if isinstance(roi_margin_mm, list) else [roi_margin_mm] * 3
        self.solver = solver
        self.model = self.initialize_model(self.solver)
        self.dwellTimeVariables = self.set_dwellTimeVariables(plan=self.plan)
        self.roi_bounds: List[List[float]] = self.get_optimization_roi_bounds(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            roi_margin_mm=roi_margin_mm,
            )
        self.penalty_function = self.set_penalty_function_and_constraints(
            plan=self.plan,
            dwellTimeVariables=self.dwellTimeVariables,
            model=self.model)

    def initialize_model(
        self,
        solver: str,
        pth_logfile:str = None) -> Any:
        r"""
        ### Purpose:
        - A function to initialize the model. The model is the object that incorporates all the attributes above to output the optimal 
        dwell_time for each DwellTimeVariable.
        ### Inputs:
        - solver:str := The name of the solver to be used. Default is None.
        ### Outputs:
        - model: Any := The model object.
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
    ) -> List[Var]:
        r"""
        ### Purpose:
        - A function to get the dwellTimeVariables from the plan. The dwellTimeVariables are dwell times for each dwell positon
        inside the catehter table.
        ### Inputs:
        - plan: BrachyPlan := The plan should have a catheter table with at least one dwell position.
        - initial_dwell_time:float := The initial dwell_time of the DwellTimeVariable. Default is 0.
        - lower_bound:float := The lower bound of the DwellTimeVariable. Default is 0.
        - upper_bound:float := The upper bound of the DwellTimeVariable. Default is 100.
        ### Outputs:
        - DwellTimeVariable_list:List[DwellTimeVariable] := A list of dwellTimeVariables to be optimized. The dwellTimeVariables are the dwell times
        for each dwell position inside the catheter table.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please initialize the model first.")
        dwellTimeVariable_list = []
        dwell_counter = 0
        for catheter in plan.catheter_table:
            for dwell_position in catheter.dwells:
                dwellTimeVariable_list.append(
                    DwellTimeVariable(
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
        dwellTimeVariables: List[DwellTimeVariable],
        roi_margin_mm: List[float] = [5.0, 5.0, 5.0],
    ) -> List[List[float]]:
        r"""
        ### Purpose:
        - A function to get the coordinate bounds for the optimization region of
        interest (roi) from the plan.  The roi is the inclusion mask for the voxels
        to be included in the optimization. The roi is defined as the region around 
        the furthest dwell position along each axis plus the margin.
        ### Inputs:
        - plan: BrachyPlan := The plan should have a catheter table with at least one dwell position.
        - dwellTimeVariables:List[DwellTimeVariable] := The set of the dwellTimeVariables to be optimized.
        - roi_margin_mm:List[float] := The distance from the furthest dwell position along each axis
        to consider voxels the dose rate maps. for each axis:
            inclusion_space = [
                closest_dwell_position -relavance_distance :
                furthest_dwell_position + relavance_distance
                ]
        ### Outputs:
        - roi_optimization:ROIMask := The optimization region of interest (roi) from the plan.
        """
        # get the inclusion mask for the voxels to be included
        inclusion_boundaries = np.ones((3, 2))
        dwell_bounds = np.zeros((3, 2))
        for axis in [0, 1, 2]:
            dwell_bounds[axis, 0] = np.min(
                [dwelltime.coordinates[axis] for dwelltime in dwellTimeVariables]
            )
            dwell_bounds[axis, 1] = np.max(
                [dwelltime.coordinates[axis] for dwelltime in dwellTimeVariables]
            )
            inclusion_boundaries[axis, 0] = (
                dwell_bounds[axis, 0] - roi_margin_mm[axis]
            )
            inclusion_boundaries[axis, 1] = (
                dwell_bounds[axis, 1] + roi_margin_mm[axis]
            )
            # if the inclusion bound is outside the dose image, set it to the dose image bounds
            if (
                inclusion_boundaries[axis][0]
                < plan.combined_dose.dose_image.origin[axis]
            ):
                inclusion_boundaries[axis][0] = plan.combined_dose.dose_image.origin[axis]
            if (
                inclusion_boundaries[axis][1]
                > plan.combined_dose.dose_image.origin[axis]
                + plan.combined_dose.dose_image.gridSizeInWorldUnit[axis]
            ):
                inclusion_boundaries[axis][1] = (
                    plan.combined_dose.dose_image.origin[axis]
                    + plan.combined_dose.dose_image.gridSizeInWorldUnit[axis]
                )
        return inclusion_boundaries

    def set_penalty_function_and_constraints(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[DwellTimeVariable],
        model: Model,
    ) -> Callable:
        r"""
        ### Purpose:
        - A function to get the penalty function from the plan. The goal for the voxels inside the
        target volume is to reach the prescribed dose. the goal for the voxels in organs at risk is to
        reach zero. The dose rate maps are normalized by the prescribed dose by default. Only voxels
        that are close to the furthest dwell positions are considered.

        P = (1/prescribed_dose) * sum( p_linear_i(target) + p_quad_i(target) + p_hotspot_i(target))
            + sum( p_linear_i(oar) )

        where   p_linear_i(target) = dose_i - prescribed_dose_i if dose_i > prescribed_dose_i for i in all target volume voxels
                p_linear_i(oar) = dose_i for i in all oar voxels
                p_quad_i = (p_linear)^2
                p_hotspot_i = abs( mean(dose_i) - 2*prescribed_dose_i) if mean(dose_i) > 1.5*prescribed_dose_i
                dose_i := dose_rate_map_i * dwell_time_i

        ### Inputs:
        - plan: BrachyPlan := The plan should have a catheter table with at least one dwell position,
        a target volume defined, and the dose rate maps loaded.
        - dwellTimeVariables:List[DwellTimeVariable] := The set of the dwellTimeVariables to be optimized.
        - model:Model := The model object. Default is None.
        ### Outputs:
        - penalty_function:Callable := A function that states how good a set of dwellTimeVariables are.
        The penalty function is a function of the dose rate maps and the prescribed dose.
        """
        penalty_terms = {
            "linear":0,
            "quadratic":0,
            "hotspot":0,
            "uniformity":0,
        }
        num_dose_points_dict = {}
        for structure in plan.structure_list:
            if structure.optimization_config is None:
                continue
            # for now, skip the hotspot estimator structures
            if "hotspot_estimator" in structure.name.lower():
                continue

            structure_mask = structure.mask
            optim_spacing = structure.optimization_config.spacing_mm
            target_dose = structure.optimization_config.dose_voxel_goal
            linear_weight = structure.optimization_config.penalty_weight_linear
            quadratic_weight = structure.optimization_config.penalty_weight_quadratic
            uniformity_weight = structure.optimization_config.penalty_weight_uniformity
            min_dose = structure.optimization_config.min_dose
            structure_max_dose = structure.optimization_config.max_dose

            w = model.addVar(name=f"linear_weight_{structure.name}", vtype=GRB.CONTINUOUS)
            model.addConstr(w == linear_weight, name=f"linear_weight_{structure.name}")
            dose = 0
            for variable in dwellTimeVariables:
                cropped_resampled_dose_rate_map = crop_mask_resample_dose_rate_map(
                    dose_rate_map=variable.dose_rate_map,
                    template_dose_obj=plan.combined_dose,
                    roi_bounds=self.roi_bounds,
                    structure_mask=structure_mask,
                    optim_spacing=optim_spacing
                )
                dose += variable.model_variable * cropped_resampled_dose_rate_map

            dose = dose.flatten()
            num_dose_points = dose.shape[0]
            num_dose_points_dict[structure.name] = num_dose_points
            if num_dose_points == 0:
                continue
            for i, d in enumerate(dose):
                if structure.target_volume:
                    # slack variable for dose value
                    x = model.addVar(0, target_dose-min_dose, name=f"dose_slack_var_{i}")
                    model.addConstr(d + x >= target_dose,
                    name=f"dose_{structure.name}_slack_linear_constr_{i}")
                    # slack variable for dose uniformity
                    y = model.addVar(-GRB.INFINITY, target_dose-min_dose)
                    model.addConstr(d + y == target_dose,
                    name=f"dose_{structure.name}_slack_uniform_constr_{i}")
                    penalty_terms["linear"] += (
                        (linear_weight / num_dose_points) * x
                    )
                    penalty_terms["quadratic"] += (
                        ( quadratic_weight/ num_dose_points) * x * x
                    )
                    penalty_terms["uniformity"] += (
                        (uniformity_weight / (num_dose_points*1000)) * y * y
                    )
                else:
                    x = model.addVar(0, structure_max_dose-target_dose, name=f"dose_slack_var_{i}")
                    model.addConstr(d - x <= target_dose,
                    name=f"dose_{structure.name}_slack_linear_constr_{i}")
                    penalty_terms["linear"] += (
                        (linear_weight / num_dose_points) * x                        
                    )
                    penalty_terms["quadratic"] += (
                        (quadratic_weight / num_dose_points) * x * x
                    )

        model.setObjective(
            (
                penalty_terms["linear"]
                + penalty_terms["quadratic"]
                # + penalty_terms["hotspot"]
                + penalty_terms["uniformity"]
            ),
            GRB.MINIMIZE
        )
        model.update()

    def run(self):
        r"""
        ### Purpose:
        - A function to run the optimizer.
        """
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
        else:
            print("No optimal solution found.")

    def get_optimized_plan_from_model(
        self,
        inplace=True,
        ) -> BrachyPlan:
        r"""
        ### Purpose:
        - A function to get the optimized plan from the model after the optimizaton is done.
        By defailt, the plan is updated in place. If inplace is False, a new plan is returned.
        Note that plan is a very large object and it is not recommended to copy it.

        ### Inputs:
        - inplace:bool := If True, the plan is updated in place. If False, a new plan is returned.
        ### Outputs:
        - outplan:BrachyPlan := The optimized plan. The plan is updated in place by default.
        """
        if self.plan is None:
            raise ValueError("Plan is not set. Please set the plan first.")
        if self.model is None:
            raise ValueError("Model is not set. Please set the model first.")
        if self.dwellTimeVariables is None:
            raise ValueError("DwellTimeVariables are not set. Please set the DwellTimeVariables first.")
        # run the optimization
        self.run()
        for variable in self.dwellTimeVariables:
            # set the dwell time to the optimized value
            variable.dwell_time = variable.model_variable.X
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

def crop_mask_resample_dose_rate_map(
    dose_rate_map: np.ndarray,
    template_dose_obj: BrachyDose,
    roi_bounds: List[List[float]],
    structure_mask: ROIMask | ROIContour,
    optim_spacing: List[float]
    ) -> np.ndarray:
    r"""
    ### Purpose:
    - A function to crop the dose rate map to the roi bounds, mask it by the structure mask
    and resample it to the optimization spacing.
    ### Inputs:
    """
    from opentps.core.processing.imageProcessing.resampler3D import (
        crop3DDataAroundBox, resampleImage3DOnImage3D, resample
    )
    # create a dose object from the dose_rate_map tensor.
    # The coordinates of the dose object is the same as the combined_dose in the plan.
    masked_dose_rate_obj:BrachyDose = BrachyDose.dose_with_empty_grid_like(template_dose_obj)
    masked_dose_rate_obj.set_dose_array(dose_rate_map)
    # apply the optimization roi bounds to the dose rate image
    crop3DDataAroundBox(
        masked_dose_rate_obj.dose_image,
        roi_bounds)
    # get the structure mask from the contour
    if isinstance(structure_mask, ROIContour):
        structure_mask = structure_mask.getBinaryMask()
    # apply the structure mask to the dose rate map object                
    if not(structure_mask.hasSameGrid(masked_dose_rate_obj.dose_image)):
        structure_mask = resampleImage3DOnImage3D(
            structure_mask,
            masked_dose_rate_obj.dose_image,
            inPlace=False,
            fillValue=0
        )
    structure_mask = structure_mask.imageArray.astype(bool)
    masked_dose_rate_obj.dose_image.imageArray = (
        masked_dose_rate_obj.dose_image.imageArray * structure_mask
    )
    # resample the dose rate map to the optimization resolution
    if isinstance(optim_spacing, float):
        optim_spacing = [optim_spacing] * 3
    resample(
        masked_dose_rate_obj.dose_image,
        spacing = optim_spacing,
        inPlace=True)
    return masked_dose_rate_obj.dose_image.imageArray