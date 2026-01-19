from typing import List, Any, Dict
from brachyutils.dose.dose_utils import BrachyDose
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, model_validator
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import SimpleITK as sitk
# import gurobipy as gb
from opentps.core.data.images import ROIMask, DoseImage
from opentps.core.data import ROIContour
from brachyutils.types import BrachyPlan
from abc import ABC, abstractmethod
from opentps.core.processing.imageProcessing.sitkImageProcessing import image3DToSITK
from opentps.core.processing.imageProcessing.resampler3D import (
    resampleImage3DOnImage3D, crop3DDataAroundBox, resampleImage3D
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import compute_new_origin_for_resampling

def resample_crop_the_mask_or_contour_to_optimGrid(
    structure_mask: ROIMask | ROIContour,
    template_dose_obj: BrachyDose,
    optim_spacing: List[float] = None,
    roi_bounds: List[List[float]] = None,
    sitk_interpolator_contour=sitk.sitkLinear,
    shift_origin: bool = True
) -> ROIMask:
    r"""
    ### Purpose:
    - To prepare the structure mask or contour for optimization by resampling it 
    to the optimization grid. Shift the origin when resampling to align with the new spacing.
    ### Inputs:
    - structure_mask: ROIMask | ROIContour := The structure mask or contour to be resampled. If 
    ROIContour is provided, it will be converted to ROIMask first. If ROIMask is provided,
    it will resampled to dose grid if needed.
    - template_dose_obj: BrachyDose := The template dose object to use for resampling.
    - sitk_interpolator: sitk interpolator type := The sitk interpolator to use for resampling
    the structure mask. Default is sitk.sitkLinear.
    ### Outputs:
    - ROIMask := The resampled structure mask
    """
    # get the structure mask from the contour
    if isinstance(structure_mask, ROIContour):
        structure_mask = structure_mask.getBinaryMask(
            origin=template_dose_obj.dose_image.origin,
            spacing=template_dose_obj.dose_image.spacing,
            gridSize=template_dose_obj.dose_image.gridSize
        )
    need_resampling = False
    # apply the structure mask to the dose rate map object
    if not(structure_mask.hasSameGrid(template_dose_obj.dose_image)):
        origin_for_resampling = template_dose_obj.dose_image.origin
        spacing_for_resampling = template_dose_obj.dose_image.spacing
        need_resampling = True
    if optim_spacing is not None:
        spacing_for_resampling = [optim_spacing]*3 if isinstance(optim_spacing, float) else optim_spacing
        if shift_origin:
            origin_for_resampling = compute_new_origin_for_resampling(
                image3DToSITK(template_dose_obj.dose_image), 
                new_spacing=spacing_for_resampling
            )
        need_resampling = True
    if need_resampling:
        structure_mask = resampleImage3D(
            structure_mask,
            spacing=spacing_for_resampling,
            inPlace=False,
            fillValue=0,
            origin=origin_for_resampling,
            sitk_interpolator=sitk_interpolator_contour
        )
    # crop the structure mask to the roi bounds
    if roi_bounds is not None:
        crop3DDataAroundBox(
            structure_mask,
            roi_bounds)
    return structure_mask

def resample_mask_crop_the_doseRateMap_to_optimGrid(
    dose_rate_map: np.ndarray,
    template_dose_obj: BrachyDose,
    roi_bounds: List[List[float]]=None,
    structure_mask: ROIMask=None,
    optim_spacing: List[float]=None, 
    sitk_interpolator_dose=sitk.sitkLinear,
    shift_origin: bool = True
    ) -> np.ndarray:
    r"""
    ### Purpose:
    - A function to resample to the optimization spacing the dose rate map to the, mask it by the structure mask
    and crop it to the roi bounds the optimization spacing.
    ### Inputs:
    - dose_rate_map: np.ndarray := The dose rate map to be cropped and resampled.
    - template_dose_obj: BrachyDose := The template dose object to use for cropping and resampling.
    - roi_bounds: List[List[float]] := The bounds of the region of interest (roi) to crop the dose rate map.
    - structure_mask: ROIMask := The structure mask to apply to the dose rate map. it has to be in the same grid as the template_dose_obj.
    - optim_spacing: List[float] := The spacing of the optimization grid in mm.
    - sitk_interpolator_dose: sitk interpolator type := The sitk interpolator to use for resampling the dose rate map. Default is sitk.sitkLinear.
    - shift_origin: bool := Whether to shift the origin of the resampled dose rate map to align with the new spacing. Default is False.
    ### Outputs:
    - np.ndarray := The cropped and resampled dose rate map and mask.
    """
    # create a dose object from the dose_rate_map tensor.
    # The coordinates of the dose object is the same as the combined_dose in the plan.
    # dose_rate_obj:BrachyDose = BrachyDose.dose_with_empty_grid_like(template_dose_obj)
    # dose_rate_obj.set_dose_array(dose_rate_map)
    dose_rate_img = DoseImage(
        imageArray=dose_rate_map.swapaxes(0, 2),
        origin=template_dose_obj.dose_image.origin,
        spacing=template_dose_obj.dose_image.spacing
    )

    # # resample the dose rate map to the optimization resolution
    if optim_spacing is not None:
        if isinstance(optim_spacing, float):
            optim_spacing = [optim_spacing] * 3
        if shift_origin:
            origin_for_resampling = compute_new_origin_for_resampling(
                image3DToSITK(dose_rate_img), 
                new_spacing=optim_spacing
            )
        else:
            origin_for_resampling = None
        resampleImage3D(
            dose_rate_img,
            spacing=optim_spacing,
            inPlace=True,
            origin=origin_for_resampling,
            sitk_interpolator=sitk_interpolator_dose
            )
    # crop the dose rate map to the roi bounds
    if roi_bounds is not None:
        crop3DDataAroundBox(
        dose_rate_img,
        roi_bounds)
    # by now the structure mask is in the same grid as the template dose object
    # apply the structure mask to the dose rate map
    if structure_mask is not None:
        non_zero_dose_rate = dose_rate_img.imageArray[structure_mask.imageArray==True]
    else:
        non_zero_dose_rate = dose_rate_img.imageArray.swapaxes(0, 2).flatten()
    return non_zero_dose_rate

def process_variable(
    variable,
    # structure_name,
    structure_mask,
    plan,
    optim_spacing,
    roi_bounds,
    shift_origin:bool=True
    ):
    r"""
    ### Purpose:
    - A helper function to process a dwell time variable to get its dose rate matrix for a given structure.
    This function is used in multi-threaded processing of dwell time variables.
    ### Inputs:
    - variable: BrachyDwellTime := The dwell time variable to process.
    - structure_name: str := The name of the structure to process.
    - structure_mask: ROIMask := The structure mask to use for processing. it has to be
    in the same grid as the plan.combined_dose.
    - plan: BrachyPlan := The brachytherapy plan to use for processing.
    - optim_spacing: List[float] := The spacing of the optimization grid in mm.
    - roi_bounds: List[List[float]] := The bounds of the region of interest (roi) to crop the dose rate map.
    - shift_origin: bool := Whether to shift the origin of the resampled dose rate map to align with the new spacing. Default is True.
    ### Outputs:
    - Tuple[BrachyDwellTime, np.ndarray] | None := A tuple of the dwell time variable and its dose rate matrix for the given structure.
    If the variable is not relevant for the structure (e.g., hotspot estimator), returns None.
    """
    dwell_var = variable._model_variable

    valid_dose_points = resample_mask_crop_the_doseRateMap_to_optimGrid(
        dose_rate_map=variable.dose_rate_map,
        template_dose_obj=plan.combined_dose,
        roi_bounds=roi_bounds,
        structure_mask=structure_mask,
        optim_spacing=optim_spacing,
        sitk_interpolator_dose=sitk.sitkLinear,
        # Using Linear instead of NearestNeighbor since NN does a bad job when downsampling
        # sitk_interpolator_contour=sitk.sitkLinear, #sitkNearestNeighbor # sitkLinear
        shift_origin=shift_origin
    )

    return dwell_var, valid_dose_points

def compute_dose_rate_matrices(
        dwellTimeVariables: List[Any],
        plan: BrachyPlan,
        structure_name: str = None,
        structure_mask: ROIMask = None,
        optim_spacing: List[float] = None,
        roi_bounds: List[List[float]] = None,
        max_workers:int=16,
        shift_origin:bool=False,
        multi_processing:bool=True):
    r"""
    ### Purpose:
    - A function to compute the resampled, masked and cropped dose rate matrices
    for a list of dwell time variables for a given structure. This function can 
    use multi-processing to speed up the computation.
    ### Inputs:
    - dwellTimeVariables: List[BrachyDwellTime] := The list of dwell time variables to process.
    - plan: BrachyPlan := The brachytherapy plan to use for processing.
    - structure_name: str := The name of the structure to process.
    - structure_mask: ROIMask := The structure mask to use for processing. it has
    to be in the same grid as the plan.combined_dose.
    - optim_spacing: List[float] := The spacing of the optimization grid in mm.
    - roi_bounds: List[List[float]] := The bounds of the region of interest (
    roi) to crop the dose rate map.
    - max_workers: int := The maximum number of workers to use for multi-processing. Default is 8.
    - shift_origin: bool := Whether to shift the origin of the resampled dose rate map to align with the new spacing. Default is False.
    - multi_processing: bool := Whether to use multi-processing. Default is True.
    ### Outputs:
    - Tuple[List[Any], List[np.ndarray]] := A tuple of two lists:
        - The first list contains the model variables of the dwell time variables.
        - The second list contains the processed dose rate matrices for each dwell time variable.
    """
    if structure_name is None:
        structure_name = "No"
    dose_rate_matrices_chaos = []
    dwell_vars_chaos = []
    if multi_processing:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_variable,
                    variable=variable,
                    structure_mask=structure_mask,
                    plan=plan,
                    optim_spacing=optim_spacing,
                    roi_bounds=roi_bounds, 
                    shift_origin=shift_origin
                ): variable
                for variable in dwellTimeVariables
            }

            for future in tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{structure_name} mask is applied to all dose rate maps"):
                result = future.result()
                if result is not None:
                    dwell_var, valid_dose_points = result
                    dwell_vars_chaos.append(dwell_var)
                    dose_rate_matrices_chaos.append(valid_dose_points)
    
        dwell_vars = []
        dose_rate_matrices = []
        # sort the dwell_vars and dose_rate_matrices according to the original dwellTimeVariables order
        for var in dwellTimeVariables:
            for var_mat in zip(dwell_vars_chaos, dose_rate_matrices_chaos):
                if var_mat[0].VarName == var.name: # XXX this line will cause error for AMPL and gurobi
                    dwell_vars.append(var_mat[0])  # as they have different name attributes for their model variables
                    dose_rate_matrices.append(var_mat[1])
    else:
        for var in dwellTimeVariables:
            dwell_var, valid_dose_points = process_variable(
                variable=var,
                structure_mask=structure_mask,
                plan=plan,
                optim_spacing=optim_spacing,
                roi_bounds=roi_bounds, 
                shift_origin=shift_origin
                )
            dwell_vars.append(dwell_var)
            dose_rate_matrices.append(valid_dose_points)

    return dwell_vars, dose_rate_matrices

class Optimization_Config(BaseModel):
    """
    ### Purpose:
    - This class holds the information regarding the optimization configuration per each structure.
    When loading the BrachyPlan the optimization config is created for each structure in the plan.structure_list.
    Some attributes are unique to target structures (CTV/PTV) and some are common to all structures.
    target attributes: 
        - penalty_weight_hotspot
        - hotspot_threshold
        - catheter_recommendaion
        - penalty_weight_variance_time
        - penalty_weight_uniformity
    ### Attributes:
    - structure_name: str := The name of the structure to which this optimization config applies.
    - is_target: bool := If true, we're looking at a target structure.
    - spacing_mm: List[float] | float := The spacing of the optimization grid in mm. 
    - dose_voxel_goal: float := The dose goal for every voxel in the structure in Gy.
    - penalty_weight_linear: float := Weight for linear penalty term in objective function. Default 1.
    - penalty_weight_quadratic: float := Weight for quadratic penalty term. Default 1.
    - penalty_weight_hotspot: float := Weight for hotspot penalty term. Default 0.
    - hotspot_threshold: float := If the average dose to the hot spot estimator volume goes above (target_dose * hotspot_threshold),
    penalty will be calculated for that hot spot estimator volume. Default 0.
    - penalty_weight_uniformity: float := Weight for dose uniformity penalty. Default 1.
    - mask_margin_mm: List[float] | float := Margin around structure for optimization in mm. Default 0.
    - min_dose: float := Minimum allowed dose in Gy. Default 0.
    - max_dose: float := Maximum allowed dose in Gy. Default 500.
    - catheter_recommendaion: bool := If True, catheter positions will be optimized as well. Default False.
    - dwell_coef_dict: Dict[str, np.array] := A dictionary mapping the name of the dwell position to the cropped, masked
    and flattend dose rate map corresponding to that dwell positition.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        )

    structure_name:str = None
    is_target:bool = False
    spacing_mm:float | List[float]= None
    dose_voxel_goal:float = None
    penalty_weight_linear:float = 0
    penalty_weight_quadratic:float = 0
    penalty_weight_hotspot:float = 0
    hotspot_threshold:float = 0
    penalty_weight_uniformity:float = 0
    penalty_weight_variance_time:float = 0
    mask_margin_mm:float | List[float]= 0
    min_dose:float = 0
    max_dose:float = 500
    catheter_recommendaion: bool = False
    dwell_coef_dict:Dict[str, np.array] = None
    mask:ROIMask = None
    # may be needed later
    # self.index_range_constraints: List[int] = None
    @model_validator(mode="after")
    def validate_target_only_fields(self):
        if not self.is_target:
            assert self.penalty_weight_hotspot == 0, "only target structure can have penalty_weight_hotspot"
            assert self.hotspot_threshold == 0, "only target structure can have hotspot_threshold"
            assert self.catheter_recommendaion == False, "only target structure can have catheter_recommendaion"
            assert self.penalty_weight_variance_time == 0, "only target structure can have penalty_weight_variance_time"
            assert self.penalty_weight_uniformity == 0, "only target structure can have penalty_weight_uniformity"
        return self

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
        description="Name of the DwellTimeVariable in the format catheter_{catheter_number+1}_dwell_{dwell_position_number+1}")
    dwell_time: float = Field(ge=0, description="Initial dwell time of the DwellTimeVariable in seconds.")
    lower_bound: float = Field(ge=0, description="Lower bound of the DwellTimeVariable in seconds.")
    upper_bound: float = Field(ge=0, description="Upper bound of the DwellTimeVariable in seconds.")
    coordinates: List[float] | None = Field(default=None, description="Coordinates of the dwell position for this DwellTimeVariable.")
    dose_rate_map: BrachyDose | np.ndarray | None = Field(default=None, description="Dose rate map for this DwellTimeVariable.")

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


class BrachyDwellTimeOptim(ABC):
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
    - solution_found: A boolean indicating whether a solution was found.
    - solve_time: The time taken to solve the optimization problem.
    ### Functions:
    - initialize_model: A function to initialize the optimization model (prepare the solver and the log files).
    - set_dwellTimeVariables: A function to create the DwellTimeVariable objects based on the treatment plan. 
    - set_penalty_function_and_constraints: A function to set the penalty function and constraints for the optimization.
    - run: A function to run the optimization model inside the solver and capture the solve time.
    - get_optimized_plan_from_model: A function to get the optimized BrachyPlan from the model.
    - bound_dwell_time: A function to bound the dwell time of a DwellTimeVariable.
    """
    @abstractmethod
    def __init__(self):
        r"""
        ### Purpose:
        - A function to initialize the optimizer object with the plan and solver.
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
        self.solution_found: bool = False
        self.solve_time: float = 0.0

    @abstractmethod
    def initialize_model(
        self,
        solver: str,
        pth_logfile: str = None
    ) -> Any:
        r"""
        ### Purpose:
        - A function to initialize the model. The model is the object that incorporates all
        the attributes above to output the optimal dwell_time for each DwellTimeVariable.
        ### Inputs:
        - solver:str := The name of the solver to be used. Default is None.
        - pth_logfile:str := The path to the log file for the solver. Default is None.
        ### Outputs:
        - model: Any := The model object.
        """
        pass

    @abstractmethod
    def set_dwellTimeVariables(
        self,
        plan: BrachyPlan,
        initial_dwell_time: float = 0.0,
        lower_bound: float = 0.0,
        upper_bound: float = 100,
    ) -> List[BrachyDwellTime]:
        r"""
        ### Purpose:
        - A function to get the dwellTimeVariables from the plan.
        The dwellTimeVariables are dwell times for each dwell positon inside the catehter table.
        ### Inputs:
        - plan: BrachyPlan := The plan should have a catheter table with at least one dwell position.
        - initial_dwell_time:float := The initial dwell_time of the DwellTimeVariable. Default is 0.
        - lower_bound:float := The lower bound of the DwellTimeVariable. Default is 0.
        - upper_bound:float := The upper bound of the DwellTimeVariable. Default is 100.
        ### Outputs:
        - DwellTimeVariable_list:List[DwellTimeVariable] := A list of dwellTimeVariables to be optimized. The dwellTimeVariables are the dwell times
        for each dwell position inside the catheter table.
        """
        pass

    @abstractmethod
    def set_penalty_function_and_constraints(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[Any],
        model: Any,
    ) -> None:
        r"""
        - A function to set up the optimization model's objective function and constraints based 
        on the plan.
        """
        pass

    @abstractmethod
    def run(self):
        r"""
        - A function to run the underlying mathematical solver and capture the solution 
        status and solve time.
        """
        pass

    @abstractmethod
    def get_optimized_plan_from_model(
        self,
        inplace=True,
    ) -> BrachyPlan | None:
        r"""
        ### Purpose:
        - A function to get the optimized plan from the model after the optimization is done.
        - This method calls `self.run()` to perform the optimization.
        - If no optimal solution is found, the method returns `None`.
        By default, the plan is updated in place. If inplace is False, a new plan is returned.
        Note that plan is a very large object and it is not recommended to copy it.
        ### Inputs:
        - inplace:bool := If True, the plan is updated in place. If False, a new plan is returned.
        ### Outputs:
        - outplan:BrachyPlan | None := The optimized plan if an optimal solution is found, otherwise None. The plan is updated in place by default.
        """
        pass

    @abstractmethod
    def bound_dwell_time(
        self,
        name: str,
        lower_bound: float = None,
        upper_bound: float = None
    ) -> None:
        r"""
        ### Purpose:
        - A function to set the lower and upper bounds for a specific dwell time variable.
        - After changing the bounds, the method updates the solver model to ensure the new
        bounds are applied.
        ### Inputs:
        - name:str := The name of the dwell time variable to set the bounds for.
        - lower_bound:float := The lower bound to set for the dwell time variable. Default is None.
        - upper_bound:float := The upper bound to set for the dwell time variable. Default is None.
        ### Outputs:
        - None
        """
        pass

def get_optimization_roi_bounds(
    plan: BrachyPlan,
    dwellTimeVariables: List[BrachyDwellTime],
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
            closest_dwell_position - roi_margin_mm[axis] :
            furthest_dwell_position + roi_margin_mm[axis]
            ]
    ### Outputs:
    - inclusion_boundaries:List[List[float]] := The min and max of the roi along each axis after
    applying the margin.
    """
    # get the inclusion mask for the voxels to be included
    inclusion_boundaries = np.ones((3, 2))
    dwell_bounds = np.zeros((3, 2))
    for axis in [0, 1, 2]:
        dwell_coord_axis = [dwelltime.coordinates[axis] for dwelltime in dwellTimeVariables]
        dwell_bounds[axis, 0] = np.min(dwell_coord_axis)
        dwell_bounds[axis, 1] = np.max(dwell_coord_axis)
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
