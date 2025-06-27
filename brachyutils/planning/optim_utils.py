# from abc import ABC, abstractmethod
from typing import List, Callable, Any
from copy import deepcopy
import warnings
import time
from brachyutils.dose.dose_utils import BrachyDose
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
import numpy as np
from pathlib import Path
from opentps.core.data.images import ROIMask
from opentps.core.data import ROIContour
from gurobipy import Model, Var, GRB, MConstr, MVar
# from brachyutils.planning.plan_utils import BrachyPlan
from brachyutils.types import BrachyPlan
from abc import ABC, abstractmethod
from amplpy import AMPL

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
    - dose_rate_map: np.ndarray := The dose rate map to be cropped and resampled.
    - template_dose_obj: BrachyDose := The template dose object to use for cropping and resampling.
    - roi_bounds: List[List[float]] := The bounds of the region of interest (roi) to crop the dose rate map.
    - structure_mask: ROIMask | ROIContour := The structure mask to apply to the dose rate map.
    - optim_spacing: List[float] := The spacing of the optimization grid in mm.
    ### Outputs:
    - np.ndarray := The cropped, masked and resampled dose rate map.
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
    return masked_dose_rate_obj.get_dose_array().astype(float)

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

class BrachyDwellTime(BaseModel, ABC):
    """
    ### Purpose:
    - An abstract class (solver independent) to represent a DwellTimeVariable in the dwell time optimization problem.
    This class is used to define the properties of a dwell time variable, such as its name, initial dwell time,
    lower and upper bounds, coordinates, dose rate map, and the variable in the optimization model.
    It is used to create instances of DwellTimeVariable for each dwell position in the catheter table.
    """
    
    name: str = Field(
        pattern=r"catheter_\d+_dwell_\d+",
        description="Name of the DwellTimeVariable in the format catheter_{catheter_number}_dwell_{dwell_position_number}")
    dwell_time: float = Field(ge=0, description="Initial dwell time of the DwellTimeVariable in seconds.")
    lower_bound: float = Field(ge=0, description="Lower bound of the DwellTimeVariable in seconds.")
    upper_bound: float = Field(ge=0, description="Upper bound of the DwellTimeVariable in seconds.")
    coordinates: List[float] = Field(default=None, description="Coordinates of the dwell position for this DwellTimeVariable.")
    dose_rate_map: np.ndarray = Field(default=None, description="Dose rate map for this DwellTimeVariable.")

    _model_variable: Any = PrivateAttr(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        )

    @abstractmethod
    def build_backend_variable(self, model: Any) -> None:
        r"""
        ### Purpose:
        - A function to build the backend variable in the optimization model by adding the
        dwell time attributes like the lower and upper bounds, name, and type to the optimizatio model.
        ### Inputs:
        - model: Any := The model object.
        """
        pass
    
    @abstractmethod
    def set_bounds(
        self, *,
        lower_bound: float | None = None,
        upper_bound: float | None = None) -> None:
        r"""
        ### Purpose:
        - A function to set the lower and upper bounds for the DwellTimeVariable.
        This function should update the underlying model variable's bounds.
        ### Inputs:
        - lower_bound: float | None := The lower bound to set for the DwellTimeVariable. Default is None.
        - upper_bound: float | None := The upper bound to set for the DwellTimeVariable. Default is None.
        """
        pass

class DwellTimeOptimizer_ABC(ABC):
    r"""
    ### Purpose:
    - An abstract dwell time optimizer class to specify the common components of a dwell time optimizer class that
    easily integrates to BrachyUtils.
    ### Attributes:
    - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
    - solver: The name of the solver to be used for optimization.
    - dwellTimeVariables: The set of the dwellTimeVariables to be optimized. In HDR brachy, dwell times and catheter positions.
    - model: The object that incorporates all the attributes above to output the optimal dwell_time for each DwellTimeVariable.
    - roi_bounds: The coordinate bounds for the optimization region of interest (roi) from the plan.
    - roi_margin_mm: The distance from the furthest dwell position along each axis to consider voxels the dose rate maps.
    ### Functions:
    - initialize_model: A function to initialize the optimization model.
    - set_dwellTimeVariables: A function to set the dwellTimeVariables for the optimization.
    - get_optimization_roi_bounds: A function to get the optimization region of interest bounds.
    - set_penalty_function_and_constraints: A function to set the penalty function and constraints for the optimization.
    - run: A function to run the optimization.
    - get_optimized_plan_from_model: A function to get the optimized plan from the model.
    - bound_dwell_time: A function to bound the dwell time of a DwellTimeVariable.
    """
    @abstractmethod
    def __init__(self):
        r"""
        ### Purpose:
        - A function to initialize the optimizer.
        ### Parameters:
        - roi_margin_mm: The distance from the furthest dwell position along each axis
        to consider voxels the dose rate maps. for each axis:
            roi_bounds = [first dwell - margin : last dwell + margin]
        """
        self.plan: Any
        self.solver: str = None
        self.dwellTimeVariables: List[BrachyDwellTime] = None
        self.model: Any = None
        self.roi_bounds: List[List[float]] = None  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.roi_margin_mm: List[float] | float = 3.0

    @abstractmethod
    def initialize_model(
        self,
        solver: str,
        pth_logfile: str = None
    ) -> Any:
        pass

    @abstractmethod
    def set_dwellTimeVariables(
        self,
        plan: BrachyPlan,
        initial_dwell_time: float = 0.0,
        lower_bound: float = 0.0,
        upper_bound: float = 100,
    ) -> List[Any]:
        pass

    @abstractmethod
    def get_optimization_roi_bounds(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[Any],
        roi_margin_mm: List[float] = [5.0, 5.0, 5.0],
    ) -> List[List[float]]:
        pass

    @abstractmethod
    def set_penalty_function_and_constraints(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[Any],
        model: Any,
    ) -> Callable:
        pass

    # @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_optimized_plan_from_model(
        self,
        inplace=True,
    ) -> BrachyPlan | None:
        pass

    @abstractmethod
    def bound_dwell_time(
        self,
        name: str,
        lower_bound: float = None,
        upper_bound: float = None
    ) -> None:
        pass

class DwellTime_Gurobi(BrachyDwellTime):
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
    - _model_variable: Any := The variable in the optimization model corresponding to this DwellTimeVariable.
    - dose_rate_map:np.ndarray := The dose rate map for this DwellTimeVariable.
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
        ### Purpose:
        - A function to set the lower and upper bounds for the DwellTimeVariable.
        This function should update the underlying model variable's bounds.
        ### Inputs:
        - lower_bound: float | None := The lower bound to set for the DwellTimeVariable. Default is None.
        - upper_bound: float | None := The upper bound to set for the DwellTimeVariable. Default is None.
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

class BrachyOptim_Gurobi(DwellTimeOptimizer_ABC):
    r"""
    ### Purpose:
    - A class using Gurobi to do dwell time optimization.
    ### Attributes:
    - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
    - dwellTimeVariables: The set of the dwellTimeVariables to be optimized. In HDR brachy, dwell times and catheter positions
    - constraints: A set of relationships between the dwellTimeVariables that should not be violated.
    In HDR brachy, we want all dwell times to be positive and sometimes have upper or lower bounds.
    - solver:str := The name
    - model: The object that incorporates all the attributes above to output the optimal 
    dwell_time for each DwellTimeVariable.
    - roi_margin_mm: The distance from the furthest dwell position along each axis to consider voxels the dose rate maps.
    - roi_bounds: The coordinate bounds for the optimization region of interest (roi) from the plan.
    to consider voxels the dose rate maps. for each axis: 
    roi_bounds = [first dwell - margin : last dwell + margin]
    ### Functions:
    - get_model()
    - run()
    """
    def __init__(
        self,
        plan:BrachyPlan,
        roi_margin_mm: List[float] | float = 5.0,
        solver="gurobi"):
        r"""
        ### Purpose:
        - A function to initialize the optimizer.
        ### Parameters:
        - plan: The brachytherapy plan to be optimized. Note that the plan will be modified in place.
        - roi_margin_mm: The distance from the furthest dwell position along each axis
        """
        super().__init__()
        self.plan = plan
        self.roi_margin_mm = roi_margin_mm if isinstance(roi_margin_mm, list) else [roi_margin_mm] * 3
        self.solver = solver
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
        dwellTimeVariables: List[DwellTime_Gurobi],
        model: Model):
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
        - model: Model := The Gurobi optimization model

        ### Outputs:
        None - sets up the model objective function and constraints directly
        """
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
        ) -> BrachyPlan | None:
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
        if self.model.status != GRB.OPTIMAL:
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
        ### Purpose:
        - A function to set the lower and upper bounds for a specific dwell time variable.
        ### Inputs:
        - name:str := The name of the dwell time variable to set the bounds for.
        - lower_bound:float := The lower bound to set for the dwell time variable. Default is None.
        - upper_bound:float := The upper bound to set for the dwell time variable. Default is None.
        ### Outputs:
        - None
        """
        for variable in self.dwellTimeVariables:
            if variable.name == name:
                variable.set_bounds(lower_bound=lower_bound, upper_bound=upper_bound)
                break
        self.model.update()

class DwellTime_AMPL(BrachyDwellTime):
    r"""
    ### Purpose:
    - A class to represent a DwellTimeVariable in the dwell time optimization problem using AMPL.
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
        ### Purpose:
        - A function to set the lower and upper bounds for the DwellTimeVariable.
        This function should update the underlying model variable's bounds.
        ### Inputs:
        - lower_bound: float | None := The lower bound to set for the DwellTimeVariable. Default is None.
        - upper_bound: float | None := The upper bound to set for the DwellTimeVariable. Default is None.
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

class BrachyOptim_AMPL(DwellTimeOptimizer_ABC):
    """
    ### Purpose:
    A class to solve dwell time optimization problems using AMPL. AMPL, allows for using a variety
    of solvers, for now we use it for HiGHS, but it can be used with other solvers as well.

    ### Attributes:
    - plan: BrachyPlan := The brachytherapy plan to be optimized. Note that the plan will be modified in place.
    - solver: str := The name of the solver to be used for optimization.
    - dwellTimeVariables: List[DwellTimeVariable] := The set of the dwellTimeVariables to be optimized.
    - model: AMPL := The AMPL model object that incorporates all the attributes above to output the optimal dwell_time for each DwellTimeVariable.
    - roi_bounds: List[List[float]] := The coordinate bounds for the optimization region of interest (roi) from the plan.
    - roi_margin_mm: List[float] | float := The distance from the furthest dwell position along each axis to consider voxels the dose rate maps.
    ### Functions:
    - initialize_model: A function to initialize the AMPL model.
    - set_dwellTimeVariables: A function to set the dwellTimeVariables for the optimization.
    - get_optimization_roi_bounds: A function to get the optimization region of interest bounds.
    - set_penalty_function_and_constraints: A function to set the penalty function and constraints for the optimization.
    - run: A function to run the optimization.
    - get_optimized_plan_from_model: A function to get the optimized plan from the model.
    - bound_dwell_time: A function to bound the dwell time of a DwellTimeVariable.
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

    def initialize_model(self, solver, pth_logfile = None):
        if pth_logfile is None:
            pth_logfile = Path("temp_data/ampl_model.log").resolve()
        pth_logfile.parent.mkdir(parents=True, exist_ok=True)
        
        if solver == "highs":
            model = AMPL()
            model.option["solver"] = solver
            
            # Configure verbose output
            if self.verbose:
                model.option["display_1col"] = 20  # Display up to 20 columns
                model.option["display_eps"] = 1e-6  # Display precision
                model.option["display_round"] = 6   # Rounding precision
                
                # HiGHS-specific options for verbose output
                model.option["highs_options"] = "output_flag=true log_to_console=true"
                
                # Set log file
                model.option["log_file"] = str(pth_logfile)
                print(f"AMPL log file: {pth_logfile}")
                print(f"Using solver: {solver}")
            
            return model
        else:
            raise ValueError(f"Solver {solver} is not supported. Please use 'highs'.")

    def set_dwellTimeVariables(
        self,
        plan: BrachyPlan,
        initial_dwell_time: float = 0.0,
        lower_bound: float = 0.0,
        upper_bound: float = 100,
    ) -> List[Any]:
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
                    DwellTime_AMPL(
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
        dwellTimeVariables: List[Any],
        roi_margin_mm: List[float] = [5.0, 5.0, 5.0],
    ) -> List[List[float]]:
        pass

    def set_penalty_function_and_constraints(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[DwellTime_AMPL],
        model: AMPL,
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
        
        model.eval(f"param total_dwells := {total_dwells};")
        model.eval("set ALL_DWELLS := 1 .. total_dwells;")
        model.eval("var t_vec {ALL_DWELLS};")
        
        # Link individual dwell variables to the global vector
        for i, d_var in enumerate(dwellTimeVariables):
            model.eval(f"subject to t_def_{i+1}: t_vec[{i+1}] = {d_var._model_variable.name()};")
        
        # Initialize objective function components
        objective_terms = []
        structure_counter = 0
        
        structures_with_config = [s for s in plan.structure_list if s.optimization_config is not None]
        if self.verbose:
            print(f"Processing {len(structures_with_config)} structures with optimization configs...")
        
        for structure in plan.structure_list:
            if structure.optimization_config is None:
                continue

            structure_counter += 1
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
                
            # create the dose rate matrix A (n x m) for this structure
            A = np.column_stack(dose_rate_matrices)
            num_dose_points = A.shape[0]
            num_dwells = len(dwell_vars)
            
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

            if structure.target_volume:
                # Create structure-specific slack variables for underdosing
                model.eval(f"var x_slack_{struct_id} {{{{D_{struct_id}}}}} >= 0 <= target_dose_{struct_id} - min_dose_{struct_id};")
                # For uniformity
                model.eval(f"var y_slack_{struct_id} {{{{D_{struct_id}}}}} >= -Infinity <= target_dose_{struct_id} - min_dose_{struct_id};")

                # Create structure-specific constraints
                model.eval(
                    f"""
                    subject to dose_constraint_{struct_id} {{i in D_{struct_id}}}:
                        sum{{j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j] + x_slack_{struct_id}[i] >= target_dose_{struct_id};
                    """)
                model.eval(
                    f"""
                    subject to uniformity_constraint_{struct_id} {{i in D_{struct_id}}}:
                        sum{{j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j] + y_slack_{struct_id}[i] = target_dose_{struct_id};
                    """)

                # Add penalty terms to objective
                linear_term = f"({linear_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]"
                quadratic_term = f"({quadratic_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]^2"
                uniformity_term = f"({uniformity_weight / (num_dose_points * 1000)}) * sum{{i in D_{struct_id}}} y_slack_{struct_id}[i]^2"
                objective_terms.extend([linear_term, quadratic_term, uniformity_term])
                
            elif "hotspot_estimator:" in structure.name.lower():
                # slack variable for hotspot estimator
                model.eval(f"var x_slack_{struct_id} {{{{D_{struct_id}}}}} >= 0 <= {hotspot_threshold} * target_dose_{struct_id} - min_dose_{struct_id};")
                model.eval(
                    f"""
                    subject to hotspot_constraint_{struct_id} {{i in D_{struct_id}}}:
                        sum{{j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j] - x_slack_{struct_id}[i] <= {hotspot_threshold} * target_dose_{struct_id};
                    """)
                
                # Add hotspot penalty term
                hotspot_term = f"({hotspot_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]"
                objective_terms.append(hotspot_term)
                
            else:
                # OAR (Organ at Risk) constraints and penalties
                model.eval(f"param structure_max_dose_{struct_id} := {structure_max_dose};")
                model.eval(f"var x_slack_{struct_id} {{{{D_{struct_id}}}}} >= 0 <= structure_max_dose_{struct_id} - target_dose_{struct_id};")
                model.eval(
                    f"""
                    subject to oar_constraint_{struct_id} {{i in D_{struct_id}}}:
                        sum{{j in T_{struct_id}}} A_{struct_id}[i,j] * t_vec[j] - x_slack_{struct_id}[i] <= target_dose_{struct_id};
                    """)
                
                # Add OAR penalty terms
                linear_term = f"({linear_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]"
                quadratic_term = f"({quadratic_weight / num_dose_points}) * sum{{i in D_{struct_id}}} x_slack_{struct_id}[i]^2"
                objective_terms.extend([linear_term, quadratic_term])

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
        - A function to run the optimizer.
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
        
        # Solve with options
        solve_options = [
            "outlev=1" if self.verbose else "outlev=0",
            "logfile=" + str(self.model.option["log_file"]),
            "time_limit=60",
        ]
        self.model.solve(" ".join(solve_options))

        solve_time = time.time() - start_time
        
        # Get solve results
        solve_result = self.model.get_value("solve_result")
        solve_message = self.model.get_value("solve_message")
        
        print(f"\n=== Solve Results ===")
        print(f"Solve time: {solve_time:.2f} seconds")
        print(f"Solve result: {solve_result}")
        print(f"Solve message: {solve_message}")
        
        if solve_result == "solved":
            print("✓ Optimal solution found.")
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
        if self.plan is None:
            raise ValueError("Plan is not set. Please set the plan first.")
        if self.model is None:
            raise ValueError("Model is not set. Please set the model first.")
        if self.dwellTimeVariables is None:
            raise ValueError("DwellTimeVariables are not set. Please set the DwellTimeVariables first.")

        self.run()
        if self.model.get_value("solve_result") != "solved":
            warnings.warn(
                "No optimal solution found. Return None.",
                stacklevel=2)
            return None
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
        for variable in self.dwellTimeVariables:
            if variable.name == name:
                variable.set_bounds(
                    model=self.model,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound
                )
                break