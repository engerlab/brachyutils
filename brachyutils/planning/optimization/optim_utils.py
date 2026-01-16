from typing import List, Any, Tuple
from brachyutils.dose.dose_utils import BrachyDose
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import computed_field

import tqdm
import SimpleITK as sitk
# import gurobipy as gb
from opentps.core.data.images import ROIMask
from opentps.core.data import ROIContour
from brachyutils.types import BrachyPlan
from abc import ABC, abstractmethod
from opentps.core.processing.imageProcessing.sitkImageProcessing import image3DToSITK

from ai_assisted_brachy.utils.utils import compute_new_origin_for_resampling


def scale_to_objective(
    plan: BrachyPlan,
    objective_to_scale_to: dict[str, float],
    check_scaling: bool = False
    ) -> BrachyPlan:
    r"""
    ### Purpose:
    - A function to scale the dwell times in the plan to match a target objective value.
    ### Inputs:
    - plan: BrachyPlan := The brachytherapy plan to be scaled.
    - objective_to_scale_to: dict[str, float] := A dictionary mapping structure names to target objective values.
    - check_scaling: bool := Whether to check the scaling after applying it. Default is False.
    ### Outputs:
    - outplan: BrachyPlan := The scaled brachytherapy plan.
    """

    plan.update_plan_from_catheter_table()
    
    current_dvh_metrics = plan.get_dvh_metrics(
        # This is essential not to return percentages but absolute values
        return_percentage=False, 
        # For V metrics, since we are counting voxels which resolution is in the mm range,
        # and that in OpenTPS DVH calculation the is interpolation of cm3 volumes, reducing 
        # the bin size helps to have an accurate number of mm3 voxels. If this bin size is
        # lets say 10-7, the the number of voxels should be exact. However here we use a smaller
        # as a trade off between accuracy and computation time. Using a bin_size too large 
        # will break this scaling function.
        bin_size=0.0001,
        query_dvh_metrics=list(objective_to_scale_to.keys())
    )
    assert len(objective_to_scale_to) == 1, "Only one structure scaling is supported at a time."
    metric_to_scale = list(objective_to_scale_to.keys())[0]
    target_metric_value = objective_to_scale_to[metric_to_scale]
    if metric_to_scale not in current_dvh_metrics:
        raise ValueError(f"Structure {metric_to_scale} not found in plan DVH metrics.")
    current_metric_value = current_dvh_metrics[metric_to_scale]
    if metric_to_scale.startswith("D"):
        # Simple scaling since we directly scale the dose
        scale_factor = target_metric_value / current_metric_value if current_metric_value != 0 else 1.0
    else:
        assert metric_to_scale.startswith("V"), "Metric to scale must start with 'D' or 'V'."
        # More complicated scaling since we have to find the number of voxels missing 
        # to reach the target volume, find the smallest dose in that voxel and use this 
        # dose to estimate the scale factor
        current_metric_value *= 1000. # From cm3 to mm3
        target_metric_value *= 1000. # From cm3 to mm3
        volume_diff = target_metric_value - current_metric_value
        voxels_cubic_size_mm = np.prod(plan.combined_dose.dose_image.spacing)
        number_of_voxels_diff =int(np.round(np.abs(volume_diff) / voxels_cubic_size_mm))

        strucname_used_for_scaling = metric_to_scale.split("(")[1][:-1]  # e.g., V100%(PTV) -> PTV

        if plan.phantom.cached_structure_masks is None:
            # WARNING: this can be slow if there are many structures in the plan
            # TODO: consider caching only the needed structure
            plan.phantom.cache_structure_set_as_masks()
        structure_mask = plan.phantom.cached_structure_masks[strucname_used_for_scaling].imageArray.astype(bool)
        masked_dose = plan.combined_dose.dose_image.imageArray[structure_mask] # this is flattened: dim (1, num_dose_points in structure)
        # Getting the current number of voxels used to compute the metric
        current_number_voxels_considered = current_metric_value / voxels_cubic_size_mm
        current_number_voxels_considered = int(np.round(current_number_voxels_considered))
        dose_points_sorted = np.sort(masked_dose)[::-1] # descending order from hotest voxels so that Vx is first chunk of voxels 

        current_max_dose_considered_for_metric = dose_points_sorted[int(current_number_voxels_considered)-1]

        if volume_diff == 0:
            scale_factor = 1.0
        else:
            # Uncertainty linked to open TPS interpolation in DVH calculation in cc which
            # can lead to many many 1mm3 voxels difference in the DVH calculation
            uncertainty_in_numvoxels_in_dvh_cacl = 0.01
            if volume_diff > 0:
                # need to increase volume by number_of_voxels_diff voxels
                # We check the dose that is ordered in the number_of_voxels_diff voxel after the current metric 
                new_min_dose_to_scale_up = dose_points_sorted[
                    min(
                        current_number_voxels_considered-1 + 
                        # We want to make sure we remain above threshold
                        int(number_of_voxels_diff * (1 + uncertainty_in_numvoxels_in_dvh_cacl)),
                        len(dose_points_sorted)-1
                    )]
                assert new_min_dose_to_scale_up < current_max_dose_considered_for_metric, \
                    "New min dose to scale up must be less than current max dose considered for metric."
                scale_factor = current_max_dose_considered_for_metric/new_min_dose_to_scale_up
                assert scale_factor > 1.0, "Scale factor must be greater than 1.0 when increasing volume for V dvh metric."

            else:
                new_max_dose_to_scale_down = dose_points_sorted[
                    max(
                    current_number_voxels_considered-1 - 
                    # We want to make sure we remain above threshold
                    int(number_of_voxels_diff * (1 - uncertainty_in_numvoxels_in_dvh_cacl)),
                    0
                    )]
                assert new_max_dose_to_scale_down > current_max_dose_considered_for_metric, \
                    "New max dose to scale down must be greater than current max dose considered for metric."
                scale_factor = current_max_dose_considered_for_metric/new_max_dose_to_scale_down
                assert scale_factor < 1.0, "Scale factor must be less than 1.0 when decreasing volume for V dvh metric."

    # Apply scaling to dwell times
    print(f"Applying scale factor {scale_factor:.4f} to dwell times to reach target {metric_to_scale} of {target_metric_value:.2f} from {current_metric_value:.2f}...")
    for catheter in plan.catheter_table:
        for dwell_position in catheter.dwells:
            new_dwell_time = dwell_position.time * scale_factor
            dwell_position.time = new_dwell_time
    
    if check_scaling:
        plan.update_plan_from_catheter_table()
        new_dvh_metrics = plan.get_dvh_metrics(
            return_percentage=False, bin_size=0.0001, query_dvh_metrics=list(objective_to_scale_to.keys())
        )
        # Putting everything in mm, if V metric, it has already been scaled above
        ogval = current_metric_value if metric_to_scale.startswith("V") else current_metric_value * 1000.
        newval = new_dvh_metrics[metric_to_scale] * 1000.
        targetval = target_metric_value if metric_to_scale.startswith("V") else target_metric_value * 1000.
        print(f"""Scaling plan from {ogval:.2f} to {newval:.2f} with scale factor {scale_factor:.4f} for dwell times to match \
            target {metric_to_scale} of {targetval:.2f}""")

        # Tolerance is below 1mm3 or greater than target
        assert np.isclose(newval, targetval, atol=1) or newval >= targetval, \
            f"Scaling failed to reach target {metric_to_scale}. Current value: {newval:.2f}, Target value: {targetval:.2f}"

        # Checking we got closer to the target
        assert abs(newval - targetval) <= abs(ogval - targetval), \
            f"Failed to get closer to target {metric_to_scale}. Before: {ogval:.2f}, After: {newval:.2f}, Target: {targetval:.2f}"

    return plan


def crop_resample_dose_rate_map_and_mask(
    dose_rate_map: np.ndarray,
    template_dose_obj: BrachyDose,
    roi_bounds: List[List[float]],
    structure_mask: ROIMask | ROIContour,
    optim_spacing: List[float], 
    sitk_interpolator_dose=sitk.sitkLinear,
    sitk_interpolator_contour=sitk.sitkLinear, 
    shift_origin: bool = False
    ) -> Tuple[BrachyDose, ROIMask]:
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
    - sitk_interpolator_dose: sitk interpolator type := The sitk interpolator to use for resampling the dose rate map. Default is sitk.sitkLinear.
    - sitk_interpolator_contour: sitk interpolator type := The sitk interpolator to use for resampling the structure mask. Default is sitk.sitkLinear.
    - shift_origin: bool := Whether to shift the origin of the resampled dose rate map to align with the new spacing. Default is False.
    ### Outputs:
    - np.ndarray := The cropped and resampled dose rate map and mask.
    """
    from opentps.core.processing.imageProcessing.resampler3D import (
        crop3DDataAroundBox, resampleImage3DOnImage3D, resampleImage3D
    )
    # create a dose object from the dose_rate_map tensor.
    # The coordinates of the dose object is the same as the combined_dose in the plan.
    dose_rate_obj:BrachyDose = BrachyDose.dose_with_empty_grid_like(template_dose_obj)
    # XXX the max sometimes changes here!
    dose_rate_obj.set_dose_array(dose_rate_map)
    # apply the optimization roi bounds to the dose rate image

    crop3DDataAroundBox(
        dose_rate_obj.dose_image,
        roi_bounds)
   
    # # resample the dose rate map to the optimization resolution
    if isinstance(optim_spacing, float):
        optim_spacing = [optim_spacing] * 3
    if shift_origin:
        origin_for_resampling = compute_new_origin_for_resampling(
            image3DToSITK(dose_rate_obj.dose_image), 
            new_spacing=optim_spacing
        )
    else:
        origin_for_resampling = None
    resampleImage3D(
        dose_rate_obj.dose_image,
        spacing=optim_spacing,
        inPlace=True,
        origin=origin_for_resampling,
        sitk_interpolator=sitk_interpolator_dose
        )

    # get the structure mask from the contour
    if isinstance(structure_mask, ROIContour):
        structure_mask = structure_mask.getBinaryMask()
    # apply the structure mask to the dose rate map object                
    if not(structure_mask.hasSameGrid(dose_rate_obj.dose_image)):
        structure_mask = resampleImage3DOnImage3D(
            structure_mask,
            dose_rate_obj.dose_image,
            inPlace=False,
            fillValue=0, 
            sitk_interpolator=sitk_interpolator_contour
        )
    return dose_rate_obj, structure_mask

def process_variable(variable, structure_name, structure_mask, plan, optim_spacing, roi_bounds, shift_origin:bool=False):

    if "hotspot_estimator:" in structure_name.lower():
        if not "all_dwell_pairs" in structure_name.lower():
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
        dose_rate_obj, structure_for_masking = crop_resample_dose_rate_map_and_mask(
            dose_rate_map=variable.dose_rate_map,
            template_dose_obj=plan.combined_dose,
            roi_bounds=roi_bounds,
            structure_mask=structure_mask,
            optim_spacing=optim_spacing,
            sitk_interpolator_dose=sitk.sitkLinear,
            # Using Linear instead of NearestNeighbor since NN does a bad job when downsampling
            sitk_interpolator_contour=sitk.sitkNearestNeighbor,
            shift_origin=shift_origin
        )
        masked_dose_array = dose_rate_obj.dose_image.imageArray.astype(float)
        structure_mask = structure_for_masking

    structure_for_masking = structure_mask.imageArray.astype(bool)
    valid_dose_points = masked_dose_array[structure_for_masking == 1].flatten()

    return dwell_var, valid_dose_points

def compute_dose_rate_matrices(
        dwellTimeVariables, structure, structure_mask, plan, 
        optim_spacing, roi_bounds, max_workers:int=8, shift_origin:bool=False):
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
                roi_bounds, 
                shift_origin
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

class Optimization_Config(BaseModel):
    """
    ### Purpose:
    - This class holds the information regarding the optimization configuration per each structure.
    When loading the BrachyPlan the optimization config is created for each structure in the plan.structure_list.

    ### Attributes:
    - structure_name: str := The name of the structure to which this optimization config applies.
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
    - num_dose_points: int := Number of dose points in the structure after applying the optimization mask and resampling.
    This is computed during the optimization setup.
    ### Computed Fields:
    - linear_coeff: float := The linear coefficient for the optimization objective function.
    - quadratic_coeff: float := The quadratic coefficient for the optimization objective function.
    - uniformity_coeff: float := The uniformity coefficient for the optimization objective function.
    ### Functions:
    - to_dict() -> dict := A function to convert the Optimization_Config object to a dictionary
    """
    structure_name:str = None
    spacing_mm:float | List[float]= None
    dose_voxel_goal:float = None
    penalty_weight_linear:float = 0
    penalty_weight_quadratic:float = 0
    penalty_weight_hotspot:float = 0
    hotspot_threshold:float = 0
    penalty_weight_uniformity:float = 0
    mask_margin_mm:float | List[float]= 0
    min_dose:float = 0
    max_dose:float = 500
    num_dose_points: int = None
    # may be needed later
    # self.index_range_constraints: List[int] = None

    @computed_field
    def linear_coeff(self) -> float:
        r"""
        ### Purpose:
        - A computed field to get the linear coefficient for the optimization objective function.
        ### Outputs:
        - linear_coeff: float := The linear coefficient for the optimization objective function.
        """
        assert self.num_dose_points is not None, "num_dose_points must be set before accessing linear_coeff"
        return self.penalty_weight_linear / self.num_dose_points
    
    @computed_field
    def quadratic_coeff(self) -> float:
        r"""
        ### Purpose:
        - A computed field to get the quadratic coefficient for the optimization objective function.
        ### Outputs:
        - quadratic_coeff: float := The quadratic coefficient for the optimization objective function.
        """
        assert self.num_dose_points is not None, "num_dose_points must be set before accessing quadratic_coeff"
        return self.penalty_weight_quadratic / self.num_dose_points
    
    @computed_field
    def uniformity_coeff(self) -> float:
        r"""
        ### Purpose:
        - A computed field to get the uniformity coefficient for the optimization objective function.
        ### Outputs:
        - uniformity_coeff: float := The uniformity coefficient for the optimization objective function.
        """
        assert self.num_dose_points is not None, "num_dose_points must be set before accessing uniformity_coeff"
        return self.penalty_weight_uniformity / (self.num_dose_points * 1000)

    @computed_field
    def hotspot_coeff(self) -> float:
        r"""
        ### Purpose:
        - A computed field to get the hotspot coefficient for the optimization objective function.
        ### Outputs:
        - hotspot_coeff: float := The hotspot coefficient for the optimization objective function.
        """
        assert self.num_dose_points is not None, "num_dose_points must be set before accessing hotspot_coeff"
        return self.penalty_weight_hotspot / self.num_dose_points
    
    def to_dict(self) -> dict:
        r"""
        ### Purpose:
        - A function to convert the Optimization_Config object to a dictionary.
        ### Outputs:
        - config_dict: dict := The dictionary representation of the Optimization_Config object.
        """
        return {
            "structure_name": self.structure_name,
            "spacing_mm": self.spacing_mm,
            "dose_voxel_goal": self.dose_voxel_goal,
            "penalty_weight_linear": self.penalty_weight_linear,
            "penalty_weight_quadratic": self.penalty_weight_quadratic,
            "penalty_weight_hotspot": self.penalty_weight_hotspot,
            "hotspot_threshold": self.hotspot_threshold,
            "penalty_weight_uniformity": self.penalty_weight_uniformity,
            "mask_margin_mm": self.mask_margin_mm,
            "min_dose": self.min_dose,
            "max_dose": self.max_dose,
            "num_dose_points": self.num_dose_points
        }

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
    - get_optimization_roi_bounds: A function to get the optimization region of interest bounds (furhter dwells +/- margin).
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
    def get_optimization_roi_bounds(
        self,
        plan: BrachyPlan,
        dwellTimeVariables: List[Any],
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
        - roi_optimization:ROIMask := The optimization region of interest (roi) from the plan.
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
