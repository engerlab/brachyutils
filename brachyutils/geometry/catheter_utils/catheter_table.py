import copy
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple, Literal
from pathlib import Path
from pydantic import BaseModel, computed_field, ConfigDict, model_validator, Field
import json
import SimpleITK as sitk
from opentps.core.processing.imageProcessing.sitkImageProcessing import imageToSITK
from opentps.core.data.images import ROIMask

from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.pw_linear_interpolator import PiecewiseLinear3D
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.spline_interpolator import NeedleSplineCreator
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.catheter_setup import (
    get_rotation_from_position, CatheterSetUp, dilate_mask_in_mm
)
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.catheter_api import (
    dicom_to_catheter_table, _update_catheter_table, CreatedSetUp
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from itertools import chain

from brachyutils.dose.dose_utils import BrachyDose

class ExportConfig_Dose(BaseModel):
    """XXX : move this back to where it came from!
    Configuration for exporting dose data from the plan.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True  # Enables auto-docs from Field desc [web:48]
    )
    dir_export: str | Path = Field(None, description="Directory where dose files are exported.")
    name_combined: str = Field("combined", description="File name for combined dose output.")
    file_extension: Literal[".seq.nrrd", ".3ddose"] = Field(
        ".seq.nrrd", description="Allowed file extensions for dose files."
    )
    write_dose_rate_maps: bool = Field(
        False, description="Whether to write individual dose rate maps to files."
    )
    multi_processing: bool = Field(
        True, description="Enable multiprocessing for export (yes/no toggle)."
    )
    @computed_field
    def pth_combined(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.name_combined+self.file_extension)


class DwellPosition(BaseModel):
    r"""
    ### Purpose:
    - This class holds the information regarding a dwell position.

    ### Attributes:
    - index: int := the index of the dwell position along the catheter
    - angle := angle of the IMBT shield
    - position:dict: np.array := dwell position in the patient coordinate system [x, y, z]
    - relativePos: float := distance along the catheter from the reference point. increments of x mm
    - rotation: np.array := rotation of the dwell position in the patient coordinate system [x, y, z]
    - time: float := dwell time for this dwell position
    - weight: float := ratio of this dwell time over the sum of all dwell times in all catheters.
    - parent_catheter: 
    ### Functions:
    - to_dict() -> dict := convert the dwell position to a dictionary.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: int
    angle: int = 0
    position: List[float] | np.array
    relativePos: float
    rotation: List[float] | np.array = None
    time: float = 0.0
    catheter_index: int = None
    gen_dose_rate: bool = True
    dose_rate: 'BrachyDose' = None

    @model_validator(mode="after")
    def validate_dwell_position(self):
        self.position = np.array(self.position)
        if self.rotation is not None:
            self.rotation = np.array(self.rotation)
        return self

    @computed_field
    def name_id(self) -> str:
        return f"{self.catheter_index+1}_{self.index+1}_{self.angle}"

    def weight(self, total_time: float) -> float:
        r"""
        ### Purpose:
        - To calculate the weight of the dwell position relative to a total time.
        The total time could come from the catheter or the treatment plan.
            
        ### Inputs:
        - self := the DwellPosition object.
        - total_time:float=None := the total time of the catheter or the treatment plan.
        if this is not provided, the weight of the dwell position will be returned.
        
        ### Outputs:
        - float := the weight of the dwell position.
        """
        return self.time / total_time

    def to_dict(self, total_time:float=None) -> dict:
        r"""
        ### Purpose:
        - To convert the dwell position to a dictionary.
        ### Inputs:
        - self := the DwellPosition object.
        ### Outputs:
        - dict := the dictionary containing the dwell position.
        """
        if total_time is None:
            total_time = self.time
        return {
            "index": int(self.index),
            "name_id": self.name_id,
            "catheter_index": self.catheter_index,
            "angle": float(self.angle),
            "position": list(self.position),
            "relativePos": float(self.relativePos),
            "rotation": list(self.rotation),
            "time": float(self.time),
            "weight": float(self.weight(total_time)),
        }

    def get_position(self) -> List[float]:
        r"""
        ### Purpose:
        - To get the position of the dwell position.
        
        ### Inputs:
        - self := the DwellPosition object.
        
        ### Outputs:
        - List[float] := the position of the dwell position.
        """
        return self.position

    def isin_mask(self, mask:Union[ROIMask, sitk.Image]) -> bool:
        r"""
        ### Purpose:
        - To check if the dwell position is inside a given mask.
        
        ### Inputs:
        - self := the DwellPosition object.
        - mask:Union[ROIMask, sitk.Image] := the mask to check if the dwell position is inside.

        ### Outputs:
        - bool := True if the dwell position is inside the mask, False otherwise.
        """
        if isinstance(mask, ROIMask):
            mask = imageToSITK(mask)
        index = mask.TransformPhysicalPointToIndex(self.position)
        in_mask = mask.GetPixel(index) > 0
        return in_mask

class Catheter(BaseModel):
    r"""
    ### Purpose:
    - This class holds the information regarding a catheter. The first catheter is placed at the 
    tip position and extends back towards the insertion point. The last dwell position is placed
    at the last_dwell_coordinate.
    
    To initiate a catheter, you can provide either of the following:
    1. tip_position and last_dwell_coordinate
    2. digitization points
    3. fit_function
    4. dwells

    ### Attributes:
    - index:int := the index of the catheter.
    - tip_position: The coordinate position of the tip of the catheter.
    - digitization_points:List[np.array] := the list of digitization points of the catheter.
    - dwells:List[DwellPosition] := the list of dwell positions of the catheter.
    - afterloader_channel_number:int := the afterloader channel number of the catheter.
    - channel_total_time:float := the total time of the catheter.
    - step_size: float := distance between the subsequent dwell positions.
    - fit_function:PiecewiseLinear3D := a line that connects the dwell positions together.
    - insert_position:list := The coordinates on patient body or insertion grid where the 
    catheter was inserted from.
    - gen_dose_rates: bool := whether the catheter needs to be generated for dose calculation or not.
    ### Functions:
    - to_dict() -> dict := convert the catheter to a dictionary.
    - add_dwell(dwell:DwellPosition) -> None := add a dwell position to the catheter.
    """
    index: int
    dwells: List[DwellPosition] = None
    channel_length: Optional[float] = None
    # in case dwells are missing and fit, tip, last dwell position and step size is provided
    fit_function:Any = None
    tip_position: List[float] = None
    last_dwell_coordinate: List[float] = None
    step_size: float = 5.0
    # in case dwells and fit is missing and digitization points are provided.
    # we assume tip is the first digitization point and last dwell is the last digitization point.
    digitization_points: List[List[float]] = None
    # auxiliary attributes
    afterloader_channel_number: Optional[int] = None # if none, will be set to index
    insert_position: Optional[List[float]] = None
    # points: Optional[List[List[float]]] = None  # to keep compatibility with previous versions
    gen_dose_rates: bool = True

    @computed_field
    def name_id(self) -> str:
        return f"{self.index+1}"

    @computed_field
    def channel_total_time(self) -> float:
        r"""
        ### Purpose:
        - To calculate the total time of the catheter by summing over indivual dwell times.
        
        ### Inputs:
        - self := the Catheter object.
        
        ### Outputs:
        - float := the total time of the catheter.
        """
        return np.sum([dwell.time for dwell in self.dwells])

    @model_validator(mode="after")
    def validate_catheter(self):
        r"""
        Validate the catheter object and set necessary attributes based on provided inputs.
        This method ensures that the catheter has valid configuration by:
        1. Setting digitization_points from points if not already provided
        2. Setting afterloader_channel_number to index if not provided
        3. Initializing dwells through one of several methods (in priority order):
            - Converting existing dwells from dict format to DwellPosition objects
            - Creating dwells from a provided fit_function
            - Creating fit_function and dwells from digitization_points
            - Creating fit_function and dwells from tip_position and last_dwell_coordinate
        4. Setting tip_position and last_dwell_coordinate from the first and last dwell positions
        Raises:
             ValueError: If neither dwells, fit_function, points, nor tip/last_dwell coordinates are provided.
             ValueError: If no dwell positions are found after initialization.
        Attributes modified:
             digitization_points: Set from points if None
             afterloader_channel_number: Set to index if None
             fit_function: Generated if not provided and dwells can be created from points
             dwells: List of DwellPosition objects representing dwell positions along the catheter
             tip_position: Set to the position of the first dwell
             last_dwell_coordinate: Set to the position of the last dwell
        """
        # Set digitization_points from points if digitization_points is None
        # if self.points is not None and self.digitization_points is None:
        #     self.digitization_points = self.points

        # Set afterloader_channel_number to index if not provided
        if self.afterloader_channel_number is None:
            self.afterloader_channel_number = self.index

        # Initialize dwells based on available inputs
        if self.dwells:
            # Convert dict dwells to DwellPosition objects if needed
            if isinstance(self.dwells[0], dict):
                self.dwells = [DwellPosition(**dwell) for dwell in self.dwells]
        elif self.fit_function is not None:
            # Create dwells from fit function
            self.dwells = self.get_dwells_from_fit(
                fit_function=self.fit_function,
                step_size=self.step_size,
            )
        elif self.digitization_points:
            # Create fit and dwells from points
            self.fit_function = self.get_fit_from_points(points=self.digitization_points)
            self.dwells = self.get_dwells_from_fit(
                fit_function=self.fit_function,
                step_size=self.step_size,
            )
        elif (self.tip_position is not None and self.last_dwell_coordinate is not None):
            # Create fit and dwells from tip and last dwell coordinates
            self.fit_function = self.get_fit_from_points(
                points=[self.tip_position, self.last_dwell_coordinate]
            )
            self.dwells = self.get_dwells_from_fit(
                fit_function=self.fit_function,
                step_size=self.step_size,
            )
        else:
            raise ValueError("""Either provide dwells, fit_function, points or 
            tip and last dwell coordinate coordinates to create a catheter.""")

        # Set tip_position and last_dwell_coordinate from dwells
        if self.dwells and len(self.dwells) > 0:
            self.tip_position = self.dwells[0].position
            self.last_dwell_coordinate = self.dwells[-1].position
        else:
            raise ValueError("No dwell positions found in the catheter. Please provide valid dwells.")
        return self

    def __getitem__(self, indices: int| slice) ->  Union[DwellPosition, List[DwellPosition]] :
        r"""
        ### Purpose:
        - To get a subset of the catheter table.

        ### Inputs:
        - self := the CatheterTable object.
        - indices: int | slice := the index or slice to get.

        ### Outputs:
        - List[DwellPosition] := the list of dwell positions in the catheter table.
        TODO: Maybe we want to return a new catheter object if a slice is provided?
        """
        if isinstance(indices, slice):
            return self.dwells[indices],
        elif isinstance(indices, int):
            if indices < 0 or indices >= len(self.dwells):
                return None
            return self.dwells[indices]
    
    def __iter__(self):
        for dwell in self.dwells:
            yield dwell

    def __len__(self):
        return len(self.dwells)

    def to_dict(self, total_time=None) -> dict:
        r"""
        ### Purpose:
        - To convert the catheter to a dictionary.

        ### Inputs:
        - self := the Catheter object.
        - total_time:float=None := the total time to be used in weight calculation for
        each dwell position. if None, channel_total_time will be used.

        ### Outputs:
        - dict := the dictionary containing the catheter.
        """
        if total_time is None:
            total_time = self.channel_total_time
        return {
            "index": int(self.index),
            "dwells": [dwell.to_dict(total_time) for dwell in self.dwells],
            # "fit_function": self.fit_function,
            "tip_position": [float(x) for x in self.tip_position],
            "last_dwell_coordinate": [float(x) for x in self.last_dwell_coordinate],
            "step_size": float(self.step_size),
            "digitization_points": [[float(x) for x in point] for point in self.digitization_points] if self.digitization_points else None,
            "afterloader_channel_number": int(self.afterloader_channel_number) if self.afterloader_channel_number is not None else None,
            "insert_position": [float(x) for x in self.insert_position] if self.insert_position else None,
            "channel_total_time": float(self.channel_total_time),
            "channel_length": float(self.channel_length) if self.channel_length is not None else None
        }

    def add_dwell(self, dwell:DwellPosition) -> None:
        r"""
        ### Purpose:
        - Insert a dwell position to the catheter and update the necessary attributes.

        ### Inputs:
        - self := the Catheter object.
        - dwell:DwellPosition := the dwell position to be added.
        """
        raise NotImplementedError("This function is not implemented yet.")

    def get_dwells_from_fit(
        fit_function:PiecewiseLinear3D | NeedleSplineCreator,
        step_size: float = 5.0,
        # kwargs: Dict[str, Any] = None,
        ) -> List[DwellPosition]:
        r"""
        ### Purpose:
        - To generate dwell positions from a 3D fit function. The fit could be a spline or a
        pieacewise linear function.

        ### Inputs:
        - fit_function:Any := the fit function to be used.
        - step_size:float= 5.0 := the step size in mm between the dwell positions.

        ### Outputs:
        - List[DwellPosition] := the list of dwell positions.
        """
        dwell_positions: List[dict] = []
        if isinstance(fit_function, PiecewiseLinear3D):
            # the tip is the first point in the first segment
            previous_pt = fit_function.point_pairs[0][0]
            t_used = 0.0
            dwell_index = 1
            while t_used < 0.9999:
                point, t, distance_prev_current = fit_function.step_in_pw_line(
                    previous_pt, step_size, bound_min=t_used
                )
                # distance_prev_current = distance(previous_pt, point)
                if distance_prev_current < 0.99 * step_size:
                    # Not creating dwell position for the point that hits the 1 bound
                    # in the step_in_pw_line function if step size is not respected.
                    # ie. Not adding the last dwell position as the last digi point
                    # if distance between the two last dwell would be < step size.
                    # Giving a 1% error on the step size.
                    break
                dwell_positions.append(
                    {
                        "index":dwell_index,
                        # angle:kwargs.get("angle"),
                        "position":point,
                        "relativePos":dwell_index * step_size,
                        "rotation":None,
                        # time:kwargs.get("time"),
                    }
                )
                previous_pt = point
                t_used = t
                dwell_index += 1

            # generate the rotations for the dwell positions
            for i in range(len(dwell_positions)):
                dwell_positions[i]["rotation"] = get_rotation_from_position(i, dwell_positions)

        elif isinstance(fit_function, NeedleSplineCreator):
            raise NotImplementedError("This function is not implemented. Do it if you need it.")

        else:
            raise ValueError("fit_function should be either PiecewiseLinear3D or NeedleSplineCreator")

        # generate the dwell positions and return them
        return [DwellPosition(**dwell) for dwell in dwell_positions]

    def get_dwell_positions_as_list(self) -> List[List[float]]:
        r"""
        ### Purpose:
        - To get the dwell positions as a list of lists.

        ### Inputs:
        - self := the Catheter object.

        ### Outputs:
        - List[List[float]] := the list of dwell positions.
        """
        return [dwell.get_position() for dwell in self.dwells]
    
    def get_fit_from_points(points:List[List[float]]) -> PiecewiseLinear3D:
        r"""
        ### Purpose:
        - To generate a spline from a list of points.

        ### Inputs:
        - points:List[List[float]] := the list of points to generate the spline from.

        ### Outputs:
        - List[List[float]] := the list of points on the spline.
        """
        return PiecewiseLinear3D(points=points)

    def remove_outside_mask(self, mask:Union[ROIMask, sitk.Image]) -> None:
        r"""
        ### Purpose:
        - To filter out the dwell positions that are outside a given mask.

        ### Inputs:
        - self := the Catheter object.
        - mask:Union[ROIMask, sitk.Image] := the mask to filter the dwell positions.

        ### Outputs:
        - None
        """
        if isinstance(mask, ROIMask):
            mask = imageToSITK(mask)
        filtered_dwells = []
        dwell_idx = 0
        for dwell in self.dwells:
            if dwell.isin_mask(mask):
                new_dwell = copy.deepcopy(dwell)
                new_dwell.index = dwell_idx
                filtered_dwells.append(new_dwell)
                dwell_idx += 1
        self.dwells = filtered_dwells

    def remove_inside_mask(self, mask:Union[ROIMask, sitk.Image]) -> None:
        r"""
        ### Purpose:
        - To remove the dwell positions that are inside a given mask.

        ### Inputs:
        - self := the Catheter object.
        - mask:Union[ROIMask, sitk.Image] := the mask to remove the dwell positions.

        ### Outputs:
        - None
        """
        if isinstance(mask, ROIMask):
            mask = imageToSITK(mask)
        filtered_dwells = []
        dwell_idx = 0
        for dwell in self.dwells:
            if not dwell.isin_mask(mask):
                new_dwell = copy.deepcopy(dwell)
                new_dwell.index = dwell_idx
                filtered_dwells.append(new_dwell)
                dwell_idx += 1
        self.dwells = filtered_dwells

class CatheterTable(BaseModel):
    r"""
    ### Purpose:
    - This class holds the information regarding the catheter table.
    catheter table could be created from multiple sources:
        1. from a dicom file
            1.1. from the delivered dwell positions
            1.2. from the digitization points
        2. from a list of catheter dictionaries
        3. from a json file
        4. from a CatheterSetUp object XXX clean this
        5. from a CreatedSetUp object XXX clean this
    ### Attributes:
    - catheters_dict : Dict[Catheter] := the dictionary or list of catheter objects in the catheter table.
    it could also be a string, Path, CatheterSetup, CreatedSetup. We will convert it all to a dictionary.
    - from_delivered_dwellpositions: bool := whether the catheter table was created from delivered 
    dwell positions. only applicable if the catheter table is created from a dicom file.
    If False, the catheter table will be created from the digitization points.
    - step_size: float := the step size in mm between the dwell positions on the catheter table.
    - treatment_time: float = None := the total treatment time of the catheter table.
    this attributed is computed from the catheter list.
    - non_zero_dwell_positions := The dictionary mapping each catheter to the list of 
    dwell positions that were actually used for plan delivery.
    - num_catheters: int = None := the number of catheters in the catheter table.
    - num_dwell_positions: int = None := the number of dwell positions in the catheter table.
    ### Functions:
    - load_from_json(pth_json:Path) -> list
    - load_from_dicom(pth_dicom:Path) -> list
    """
    # TODO: unify writing to file (json, dicom, slicer markup). decide based on extension.
    ##########
    ## To enable using CatheterSetUp and CreatedSetUp as a data types, default is False.
    # This is not a parameter to be provided.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ##########

    catheters_dict: Union[
        List[Catheter], List[dict], str, Path, CatheterSetUp, CreatedSetUp,
        Dict[Catheter], Dict[dict]
    ]
    step_size: float = 5.0
    from_delivered_dwellpositions: bool = False
    _cached_combined_dose: 'BrachyDose' = None
    _time_diffs:Dict[str, float] = None

    @computed_field
    def all_dwells(self) -> List[DwellPosition]:
        r"""
        ### Purpose:
        - returns a list of all the dwell positions in this catheter table.
        """
        return list(chain.from_iterable(self))

    @computed_field
    def catheters_list(self) -> List[Catheter]:
        r"""
        ### Purpose:
        - returns a list of catheters from self.catheters_dict
        """
        return list(self.catheters_dict.values())

    @computed_field
    def treatment_time(self) -> float:
        r"""
        ### Purpose:
        - To calculate the total treatment time.
        
        ### Inputs:
        - catheter_table:CatheterTable := the catheter table object.
        
        ### Outputs:
        - float := the total treatment time.
        """
        return np.sum([catheter.channel_total_time for catheter in self.catheters_list])

    @computed_field
    def num_catheters(self) -> int:
        r"""
        ### Purpose:
        - To calculate the number of catheters in the catheter table.
        
        ### Inputs:
        - catheter_table:CatheterTable := the catheter table object.
        
        ### Outputs:
        - int := the number of catheters in the catheter table.
        """
        return len(self.catheters_dict)
    
    @computed_field
    def num_dwell_positions(self) -> int:
        r"""
        ### Purpose:
        - To calculate the number of dwell positions in the catheter table.
        
        ### Inputs:
        - catheter_table:CatheterTable := the catheter table object.
        
        ### Outputs:
        - int := the number of dwell positions in the catheter table.
        """
        return np.sum([len(catheter.dwells) for catheter in self.catheters_list])

    @computed_field
    def non_zero_dwell_positions(
        self,
        ) -> Dict[str, List[List[float]]]:
        r"""
        ### Purpose:
        - To get the non zero dwell positions in the catheter table.
        
        ### Inputs:
        - catheter_table:CatheterTable := the catheter table object.
        
        ### Outputs:
        - Dict[str, List[List[float]]] := the dictionary mapping each catheter to the list of 
        dwell positions that were actually used for plan delivery.
        """
        non_zero_dwell_positions = {}
        for catheter in self.catheters_list:
            dwell_positions = []
            for dwell in catheter.dwells:
                if dwell.time > 0.0:
                    dwell_positions.append(dwell.position)
            non_zero_dwell_positions[f"Needle_{catheter.index}"] = dwell_positions
        return non_zero_dwell_positions

    @computed_field
    def combined_dose(self) -> 'BrachyDose':
        """
        ### Purpose:
        - To calculate the combined dose by multiplying the dose rates with the dwell times.
        if this value has already been cached without change to the catheter table, then
        the cache will be returned.
        We require strict name matching between the _time_diffs and dwell.name_id
        ### Inputs:
        - self._cached_combined_dose: The combined dose caclualted previously, which will 
        be returned if no change to the catheter table has been made.
        - self._time_diffs: a dictionary of time differences for each dwell in the plan. 
        This is used to update the combined dose if the dwell times are updated without
        having to reload the dose rate maps. The keys of the dictionary should be in
        the format "{catheter.index+1}{dwell.index+1}{dwell.angle" and the values
        should be the time differences in seconds. If None, the combined dose will
        be calculated using the current dwell times in the plan.
        ### Outputs:
        - self._cached_combined_dose
        also resets self._time_diffs to None for future.
        """
        from brachyutils.dose.dose_utils import BrachyDose
        all_dwells: List[DwellPosition] = self.all_dwells
        dwells_with_doserate = [dwell for dwell in all_dwells if dwell.dose_rate is not None]
        
        if not dwells_with_doserate:
            # return self._cached_combined_dose
            raise ValueError("No dose rate found in this catheter table")

        # Initialize combined dose if not cached
        if self._cached_combined_dose is None:
            self._cached_combined_dose = BrachyDose.dose_with_empty_grid_like(
            dwells_with_doserate[0].dose_rate
            )

        # Calculate combined dose with or without time diffs
        for dwell in dwells_with_doserate:
            dwell_time = (
            self._time_diffs.get(dwell.name_id, 0) 
            if self._time_diffs is not None 
            else dwell.time
            )
            if dwell_time != 0:
                self._cached_combined_dose.dose_image.imageArray += (
                    dwell.dose_rate.dose_image.imageArray * dwell_time)
        # reset the time diffs for future
        self._time_diffs = None
        return self._cached_combined_dose

    @model_validator(mode="after")
    def validate_catheter_table(self):
        r"""
        ### Purpose:
        - To initialize the CatheterTable object.
        
        ### Inputs:
        - catheters_dict: List[Catheter] | List[dict] | str | Path | CatheterSetUp | CreatedSetUp :=
        the list of catheter objects in the catheter table or the path to a json or dicom file.
        - step_size: float := the step size in mm between the dwell positions on the catheter table.
        - from_delivered_dwellpositions: bool := if true, the dwell positions inside the delivered dwell positions will be used.
        ### Outputs:
        - CatheterTable := the catheter table object.
        """
        if (isinstance(self.catheters_dict, str) or
            isinstance(self.catheters_dict, Path)
            ):
            catheter_file = Path(self.catheters_dict)

            if not catheter_file.exists():
                raise ValueError(f"catheter file {catheter_file} does not exist.")
            if str(catheter_file).endswith(".mrk.json"):
                # if the file is a slicer markup file, load it as a json file
                raise NotImplementedError("this feature is not implemented yet.")

            if str(catheter_file).endswith(".json"):
                cat_dict = load_from_json(catheter_file)
            elif str(catheter_file).endswith(".dcm"):
                cat_dict = load_from_dicom(
                    pth_dicom=catheter_file,
                    from_delivered_dwellpositions=self.from_delivered_dwellpositions,
                )
            self.catheters_dict = cat_dict["catheter_list"]
            self.step_size = cat_dict["step_size"]

        elif isinstance(self.catheters_dict, CatheterSetUp):
            # if the catheters_dict is a CatheterSetUp object, convert it to a CatheterTable
            cat_setup = self.catheters_dict
            updated_catheter_dict = _update_catheter_table(
                catheter_table = cat_setup.catheter_table,
                digitization_points=cat_setup.digitization_points,
                fit_function=cat_setup.piece_wise_lines,
                tips=cat_setup.get_tips_coords(),
                step_size=cat_setup.step_size,
            )
            self.catheters_dict = updated_catheter_dict["catheter_list"]
            self.step_size = updated_catheter_dict["step_size"]

        elif isinstance(self.catheters_dict, CreatedSetUp):
            created_setup = self.catheters_dict
            updated_catheter_dict = created_setup.to_brachyutils_CatheterTable_format()
            self.catheters_dict = updated_catheter_dict["catheter_list"]
            self.step_size = updated_catheter_dict["step_size"]
            self.non_zero_dwell_positions = created_setup.get_non_zero_dwell_positions()

        elif isinstance(self.catheters_dict, list):
            # if catheter dict is a list, convert it to a dict
            real_dict = defaultdict(Catheter)
            for cat in self.catheters_dict:
                real_dict[cat.name_id] = cat
            self.catheters_dict = real_dict

        # check if the values are dicts or catheter\
        self.catheters_dict = {
            key: Catheter(val) if isinstance(val, dict) else val
            for key, val in self.catheters_dict.items()
        }

        return self

    def __iter__(self):
        for catheter in self.catheters_list:
            yield catheter

    def __len__(self):
        return len(self.catheters_dict)

    def __getitem__(self, indices: int | slice | str) ->  Union[Catheter, "CatheterTable"] :
        r"""
        ### Purpose:
        - To get a subset of the catheter table.

        ### Inputs:
        - self := the CatheterTable object.
        - indices: int | slice | str := the index, slice or key string to get the 
        catheters by. index and slice finds the catheters by their catheter.index,
        while if a string is provided, it'll get the catheters by name_id.
        note that catheter.name_id = str(catheter.index +1) 

        ### Outputs:
        - List[Catheter] := the list of catheters in the catheter table.
        """
        if isinstance(indices, str):
            return self.catheters_dict.get(indices, None)

        if isinstance(indices, slice):
            indices = list(range(*slice.indices(len(self.catheters_dict))))
            name_ids = [str(index+1) for index in indices]
            caths_found = {name_id: self.catheters_dict.get(name_id, None) for name_id in name_ids}
            return CatheterTable(
                catheters_dict=caths_found,
                step_size=self.step_size,
                from_delivered_dwellpositions=self.from_delivered_dwellpositions,
            )
        elif isinstance(indices, int):
            if indices < 0 or indices >= len(self.catheters_dict):
                return None
            return self.catheters_dict.get(f"{indices+1}", None)

    def __add__(self, other: "CatheterTable") -> "CatheterTable":
        r"""
        ### Purpose:
        - To add two catheter tables together.

        ### Inputs:
        - self := the first CatheterTable object.
        - other := the second CatheterTable object.

        ### Outputs:
        - CatheterTable := the combined CatheterTable object.
        """
        if not isinstance(other, CatheterTable):
            raise ValueError("other should be a CatheterTable object.")
        if self.step_size != other.step_size:
            raise ValueError("Cannot add two catheter tables with different stepsizes.")
        return CatheterTable(
            catheters_dict=self.catheters_dict | other.catheters_dict,
            stepsize=self.stepsize,
            from_delivered_dwellpositions=self.from_delivered_dwellpositions,
        )

    def __iadd__(self, other: "CatheterTable") -> "CatheterTable":
        r"""
        ### Purpose:
        - To add another catheter table to the current catheter table.

        ### Inputs:
        - self := the current CatheterTable object.
        - other := the CatheterTable object to be added.

        ### Outputs:
        - CatheterTable := the updated CatheterTable object.
        """
        if not isinstance(other, CatheterTable):
            raise ValueError("other should be a CatheterTable object.")
        self.catheters_dict = self.catheters_dict | other.catheters_dict
        if self.step_size != other.step_size:
            raise ValueError("Cannot add two catheter tables with different stepsizes.")
        return self

    def __delitem__(self, indicies: int | slice | str):
        r"""
        ### Purpose:
        - To delete a few catheters from the catheter table.

        ### Inputs:
        - self := the CatheterTable object.
        - indicies: int | slice | str:= the index, slice or the specific name id
        of the catheter to be deleted.

        ### Outputs:
        - None
        """
        if isinstance(indices, str):
            del self.catheters_dict.get(indices)
        if isinstance(indices, slice):
            indices = list(range(*slice.indices(len(self.catheters_dict))))
            name_ids = [str(index+1) for index in indices]
            for name_id in name_ids:
                del self.catheters_dict[name_id]

        elif isinstance(indices, int):
            if indices < 0 or indices >= len(self.catheters_dict):
                return None
            del self.catheters_dict[f"{indices+1}"]

    def __sub__(self, other: "CatheterTable") -> "CatheterTable":
        r"""
        ### Purpose:
        - To take the difference between self and other catheter table. If a catheter in the
        other catheter table does not exist in self (name_id) not found, the entire catheter 
        will be included in the difference. If the other catheter exists in self, the dwell times,
        position and rotation of that catheter is subtracted from from the self catheter.
        This subtraction excludes the dose rates.
        gen_dose_rate is set to true if the position, rotation or angle has changed.

        ### Inputs:
        - self := the current CatheterTable object.
        - other := the CatheterTable object to be subtracted.
        ### Outputs:
        - CatheterTable := A catheter table with the differences.
        """
        if not isinstance(other, CatheterTable):
            raise ValueError("other should be a CatheterTable object.")
        dict_catheter_diffs = defaultdict(Catheter)
        list_dwell_diffs = []
        
        for name_id, other_catheter in other.catheters_dict.items():
            catheter = self[name_id]
            if catheter is None:
                dict_catheter_diffs[name_id] = other_catheter
            else:
                for dwell in catheter.dwells:
                    for other_dwell in other_catheter.dwells:
                        if dwell.index == other_dwell.index:
                            diff_time = dwell.time - other_dwell.time
                            diff_angle = dwell.angle - other_dwell.angle
                            diff_relativePos = dwell.relativePos - dwell.relativePos
                            diff_position = np.zeros(3)
                            diff_rotation = np.zeros(3)
                            for i in range(3):
                                diff_position[i] = dwell.position[i] - other_dwell.position[i]
                                diff_rotation[i] = dwell.rotation[i] - other_dwell.rotation[i]
                            # gen dose rate if any attribute other than the dwell time has changed.
                            diff_gen_doserate = False
                            if (np.any(diff_position !=0) or np.any(diff_rotation !=0)
                                or diff_angle != 0):
                                diff_gen_doserate = True
                            dwell_diff = DwellPosition(
                                index=dwell.index,
                                angle=diff_angle,
                                position=diff_position,
                                relativePos=diff_relativePos,
                                rotation=diff_rotation,
                                time=diff_time,
                                gen_dose_rate=diff_gen_doserate,
                                catheter_index=catheter.index
                            )
                            list_dwell_diffs.append(dwell_diff)

                    dict_catheter_diffs[catheter.name_id] = Catheter(
                        index=catheter.index,
                        dwells=list_dwell_diffs,
                        )
        return CatheterTable(
            catheters_list=dict_catheter_diffs,
            from_delivered_dwellpositions=self.from_delivered_dwellpositions
        )

    def append(self, catheter: Catheter) -> None:
        r"""
        ### Purpose:
        - To append a catheter to the catheter table. Appending
        will over-write the catheter.index based on the largest
        index in self. If you would like to preserve the catheter.index
        use __setitem__

        ### Inputs:
        - self := the CatheterTable object.
        - catheter: Catheter := the catheter to be appended.

        ### Outputs:
        - None
        """
        if not isinstance(catheter, Catheter):
            raise ValueError("catheter should be a Catheter object.")
        new_index = len(self.catheters_dict)
        catheter.index = new_index
        self.catheters_list[catheter.name_id] = catheter

    def __setitem__(self, name_id: str, new_catheter: dict | Catheter) -> None:
        r"""
        ### Purpose:
        - To add a new catheter to the catheter table based on its name_id.
        the name_id = index+1.
        """
        if new_catheter.name_id != name_id:
            raise ValueError("The name_id of the new catheter does not \
match its index, be sure that the name_id == new_catheter.index +1")
        if not (isinstance(new_catheter, dict) or isinstance(new_catheter, Catheter)):
            raise ValueError("The new_catheter should of type dict or Catheter")

        self.catheters_dict[name_id] = (
            new_catheter if isinstance(new_catheter, Catheter)
            else Catheter(new_catheter))

    def get_dwells_by_name_ids(self, name_ids: List[str]) -> List[DwellPosition]:
        r"""
        ### Purpose:
        - To return dwell positions that have the queried name ids.
        The name ids are in the format {catheter.index+1}_{dwell_index+1}_{angle}
        ### Inputs:
        - name_ids := The list of name ids to be returned.
        ### Outputs:
        - out_dwells : List[DwellPosition] := The dwell positions with the matching name ids
        """
        out_dwells = []
        for name_id in name_ids:
            for dwell in self.all_dwells:
                if dwell.name_id == name_id:
                    out_dwells.append(dwell)
        return out_dwells

    def get_catheters_by_ids(self, name_ids: List[str]) -> List[Catheter]:
        r"""
        ### Purpose:
        - To return the catheters that have the queried name ids.
        The name ids are in the format {catheter.index+1}
        ### Inputs:
        - name_ids := The list of name ids to be returned.
        ### Outputs:
        - out_catheters : List[Catheter] := The catheters with the matching name ids        
        """
        out_catheters = []
        for name_id in name_ids:
            for cath in self.catheters_list:
                if cath.name_id == name_id:
                    out_catheters.append(cath)
        return out_catheters

    def reset_index(self) -> None:
        r"""
        ### Purpose:
        - To reset the index of the catheters in the catheter table.

        ### Inputs:
        - self := the CatheterTable object.

        ### Outputs:
        - None
        """
        for i, catheter in enumerate(self.catheters_list):
            catheter.index = i

    def get_catheters_for_dose_gen(self):
        r"""
        ### Purpose:
        - To get a catheter table with only the catheters that are needed for dose rate generation
        """
        dose_gen_list = [cat for cat in self if cat.gen_dose_rates]
        return CatheterTable(
            catheters_list=dose_gen_list,
            step_size=self.step_size,
            from_delivered_dwellpositions=self.from_delivered_dwellpositions
        )

    def to_dict(self) -> dict:
        r"""
        ### Purpose:
        - To convert the catheter table to a dictionary.
        ### Inputs:
        - self := the CatheterTable object.
        ### Outputs:
        - dict := the dictionary containing the catheter table.
        """
        treatment_t = self.treatment_time
        return {
            "catheter_list": [
                catheter.to_dict(total_time=treatment_t) 
                for catheter in self.catheters_list
                ],
            "step_size": float(self.step_size),
            "treatment_time": float(treatment_t)
        }

    def info(self) -> None:
        r"""
        ### Purpose:
        - To print the information about the catheter table.
        """
        print("Catheter table info is as follows:")
        print(f"Number of catheters: {len(self.catheters_list)}")
        print(f"Total treatment time: {self.treatment_time}")
        for catheter in self.catheters_list:
            print(f"Catheter ID: {catheter.index}")
            print(f"Number of dwell positions: {len(catheter.dwells)}")
            print(f"Total channel time: {catheter.channel_total_time}")

    def write_to_json(self, pth_json: Path) -> None:
        r"""
        ### Purpose:
        - Write the catheter table to a json file.
        
        ### Inputs:
        - pth_json: Path := the path to the json file where the catheter table will be written.
        
        ### Outputs:
        - Void := will write the catheter table to a json file.
        """
        pth_json = Path(pth_json)
        pth_json.parent.mkdir(parents=True, exist_ok=True)
        with open(pth_json, "w") as json_file:
            json.dump(self.to_dict(), json_file, indent=4
            )

    def write_to_slicer_markup(
        self,
        pth_mrk_json: Path | str,
        remove_text: bool = True,
        one_markup_per_catheter: bool = False,
        ) -> None:
        r"""
        ### Purpose:
        - Write the catheter table to a json file in the slicer markup format.
        
        ### Inputs:
        - pth_json: Path := the path to the json file where the catheter table will be written.
        
        ### Outputs:
        - Void := will write the catheter table to a json file in the slicer markup format.
        """
        from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.utils import create_marker_pts_from_catheter_table
        pth_mrk_json = Path(pth_mrk_json)
        if not str(pth_mrk_json).endswith(".mrk.json"):
            raise ValueError("The output file name should end with .mrk.json")
        pth_mrk_json.parent.mkdir(parents=True, exist_ok=True)

        # point_list = [catheter.digitization_points for catheter in self]
        
        create_marker_pts_from_catheter_table(
            output_path=str(pth_mrk_json),
            catheter_table=self.to_dict(),
            one_markup_per_catheter=one_markup_per_catheter,
            remove_text=remove_text,
        )

    def get_dwell_positions_as_list(self) -> List[List[float]]:
        r"""
        ### Purpose:
        - To get the dwell positions as a list from all catheters in the catheter table.
        
        ### Inputs:
        - self := the CatheterTable object.
        
        ### Outputs:
        - List[List[float]] := the list of dwell positions from all catheters.
        """
        dwell_positions = []
        for catheter in self.catheters_list:
            dwell_positions.extend(catheter.get_dwell_positions_as_list())
        return dwell_positions

    def remove_inside_mask(self, mask:Union[ROIMask, sitk.Image], margin_mm: float = 0.0) -> None:
        r"""
        ### Purpose:
        - To filter out the dwell positions that are inside a given mask.

        ### Inputs:
        - self := the CatheterTable object.
        - mask:Union[ROIMask, sitk.Image] := the mask to filter the dwell positions.

        ### Outputs:
        - None
        """
        if isinstance(mask, ROIMask):
            mask = imageToSITK(mask)
        if margin_mm > 0.0:
            mask = dilate_mask_in_mm(mask, margin_mm, voxel_based=False)

        for catheter in self.catheters_list:
            catheter.remove_inside_mask(mask)
        
    def remove_outside_mask(self, mask:Union[ROIMask, sitk.Image], margin_mm: float = 0.0) -> None:
        r"""
        ### Purpose:
        - To filter out the dwell positions that are outside a given mask.

        ### Inputs:
        - self := the CatheterTable object.
        - mask:Union[ROIMask, sitk.Image] := the mask to filter the dwell positions.

        ### Outputs:
        - None
        """
        if isinstance(mask, ROIMask):
            mask = imageToSITK(mask)
        if margin_mm > 0.0:
            mask = dilate_mask_in_mm(mask, margin_mm, voxel_based=False)

        for catheter in self.catheters_list:
            catheter.remove_outside_mask(mask)

    def load_dose_rates(
        self,
        dir_dose_rate: str| Path,
        load_uncertainty:bool=False,
        multi_processing: bool = True,
        combined_dose_only: bool = False,
        dose_dtype=np.float32,
        ):
        r"""
        ### Purpose:
        - To load the dose rates into the CatheterTable object given a folder with
        patient's dose rate files and the catheter table loaded into the BrachyPlan object.
        In addition, combined dose is calculated as a linear combination of the dose rates
        and dwell times.
        ### Inputs:
        - `dir_dose_rate` :=  path to the directory containing the dose rate files. we assume
        that the name of the dose rate files end as "run_X_X_X.seq.nrrd".
        where the X corresponds to the catheter index+1, dwell index+1, and angle in increasing order.
        - `load_uncertainty`:= If true, uncertainty is loaded from the dose file, else it'll be set to 1. 
        - `multi_processing` := if True, the dose rate files will be loaded in parallel. By default,
        we use 8 cores for parallel processing.
        - `combined_dose_only`:bool = False := flag to keep only the combined dose in memory after loading.
        ### Outputs:
        - Void := will update the BrachyPlan.dose_rate_dict attribute
        """
        if self.num_dwell_positions == 0:
            raise ValueError("Cannot load dose rates since there is no catheters or dwells in this catheter table.")

        dir_dose_rate = Path(dir_dose_rate).resolve()
        if not dir_dose_rate.exists():
            raise ValueError(f"directory of dose rates does not exist: {dir_dose_rate}")

        # figure out which dwells we want to load dose rates for
        all_dwells = self.all_dwells
        new_dose_rate_files = [
            dir_dose_rate/f"run_{x.name_id}.seq.nrrd" 
            for x in all_dwells if x.gen_dose_rate]
        # check if the paths are correct
        for pth in new_dose_rate_files:
            if not pth.exists():
                raise ValueError(f"Dose rate path ({pth}) does not exist. Either run export or set gen_dose_rate \
to False for the corresponding dwell position.")
        from brachyutils.dose.dose_utils import BrachyDose
        dose_rate_dict = defaultdict(BrachyDose)
        if multi_processing:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(_load_single_dose_rate, pth, load_uncertainty, dose_dtype): pth
                    for pth in new_dose_rate_files
                    }
                for action in tqdm(
                    as_completed(futures),
                    desc="Loading dose rate maps",
                    total=len(new_dose_rate_files)):
                    try:
                        dose_rate = action.result()
                        dwell_id = dose_rate.path.name.split(".")[0].split("run_")[1]
                        dose_rate_dict[dwell_id] = dose_rate
                    except:
                        failed_path = futures[action]
                        raise ValueError(f"Failed loading f{failed_path}")
        else:
            for pth in tqdm(
                new_dose_rate_files,
                desc="Loading dose rate maps",
                total=len(new_dose_rate_files)):
                dose_rate = _load_single_dose_rate(
                    pth_dose_rate=pth,
                    load_uncertainty=load_uncertainty,
                    dtype=dose_dtype)
                dwell_id = dose_rate.path.name.split(".")[0].split("run_")[1]
                dose_rate_dict[dwell_id] = dose_rate

        for dwell in all_dwells:
            dwell.dose_rate = dose_rate_dict.get(dwell.name_id, None)
            dwell.gen_dose_rate = False

        # run combined dose to fill out the cached combined dose 
        self.combined_dose
        if load_uncertainty:
            self._calculate_combined_uncertainty()
        if combined_dose_only:
            for dwell in all_dwells:
                del dwell.dose_rate

    def _calculate_combined_uncertainty(self):
        r"""
        ### Purpose:
        - To calculate the combined uncertainty of the combined dose map based on the
        dose rates and dwell times.
        ### Inputs:
        - self._cached_combined_dose := the BrachyDose
        ### Outputs:
        - Void := will update the self.combined_dose.uncertainty_image
        """
        if self._cached_combined_dose is None:
            raise ValueError("combined dose is not calculated yet")

        treatment_time = self.catheter_table.treatment_time
        all_dwells = self.all_dwells
        dwells_with_doserate = [dwell for dwell in all_dwells if dwell.dose_rate is not None]
        # sanity check the dwell times matching the treatment time
        sanity_time = 0
        for dwell in dwells_with_doserate:
            sanity_time += dwell.time
        if sanity_time != treatment_time:
            raise ValueError(f"The treatment time is {treatment_time}, which does not \
agree with the sum of dwells times that have dose rates ({sanity_time})")
        for dwell in dwells_with_doserate:
            self._cached_combined_dose.uncertainty_image.imageArray.fill(0)
            self.combined_dose.uncertainty_image.imageArray += (
                dwell.dose_rate.uncertainty_image.imageArray * (dwell.time/treatment_time)
                )**2
        self.combined_dose.uncertainty_image.imageArray = np.sqrt(
        self.combined_dose.uncertainty_image.imageArray)

    def export_dose(
        self,
        export_config_dose: ExportConfig_Dose
    ):
        r"""
        ### Purpose:
        - to export combined dose map and if needed the dose rate maps to a given directory.
        exporting dose rate maps is optional.
        ### Inputs:
        - export_config_dose: The dose export configuration. Look at ExportConfig_Dose for more info 
        ### Outputs:
        - None := will export the dose map into the specified export directory.
        ### Dependencies:
        - _write_single_dose_rate()
        - multiprocessing
        """
        dir_export = Path(export_config_dose.dir_export)
        # write combined dose
        self.combined_dose.write_brachydose_to_file(
            export_config_dose.pth_combined
        )
        if export_config_dose.write_dose_rate_maps:
            all_dwells = self.all_dwells
            dose_rate_dict = {
                dwell.name_id: dwell.dose_rate 
                for dwell in all_dwells 
                if dwell.dose_rate is not None}

            if export_config_dose.multi_processing:
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {
                        executor.submit(
                            _write_single_dose_rate,
                            dose_rate_dict.get(dose_rate_name),
                            dir_export,
                            export_config_dose.file_extension,
                            f"run_{dose_rate_name}"):
                            dose_rate_name for dose_rate_name in dose_rate_dict
                        }
                    for action in tqdm(as_completed(futures), desc="Writing dose rate maps"):
                        try:
                            action.result()
                        except:
                            failed_path = futures[action]
                            raise ValueError(f"Failed writing {failed_path}")
            else:
                for dwell_name in tqdm(dose_rate_dict, desc="Writing dose rate maps"):
                    _write_single_dose_rate(
                        dose_rate=self.dose_rate_dict.get(dwell_name),
                        dir_export=dir_export,
                        file_name=f"run_{dwell_name}",
                        dose_extension=export_config_dose.file_extension)
        print(f"Dose exported to {dir_export}")

    def merge(self, new_catheter_table:"CatheterTable"):
        r"""
        ### Purpose:
        - Given a new catheter table, merge it with self.
        If a catheter with a specific index does not exist, it'll be added as it is.
        If a catheter with a specific index exists, then the previous catheter will be updated.
        Same logic applies to the dwells except that dose rates are kept only if dwell time is updated. 
        otherwise the dwells inside a catheter are replaced.

        This requires the indecies in the new catheter table to be aware of self. Idealy, we should
        be using dictionaries, but it's too late at this point.
        Also, this function assumes that the catheter.index attribute in self matches the the index
        of the catheter in self.catheters_list, while this assumption is not required for the 
        new_catheter_table (I know, horrible design choices!)

        ### Inputs:
        - new_catheter_table:CatheterTable := The new catheter table used
        to update self.
        ### Output:
        None := update self.
        """
        catheter_table_diff = self - new_catheter_table
        self._time_diffs = {}
        # identify common catheters
        # name_ids_diff = [cath.name_id for cath in catheter_table_diff]
        name_ids_self = [cath.name_id for cath in self]
        # XXX: debug this with the finalized catheter update logic!
        for catheter_diff in catheter_table_diff:
            # if the catheter is not in the current catheter table, add the entire catheter with all its dwells.
            if catheter_diff.name_id not in name_ids_self:
                self.append(new_catheter_table[catheter_diff.index])
                continue
            for dwell_diff in catheter_diff.dwells:
                # if that dwell is not in the current cathetr, add it.
                if self[catheter_diff.index][dwell_diff.index] is None:
                    self[catheter_diff.index][dwell_diff.index] = new_catheter_table[catheter_diff.index][dwell_diff.index]
                    continue
                else:
                    if np.any(dwell_diff.position != 0) or np.any(dwell_diff.rotation != 0):
                        # if the postion or rotation of the dwell has changed,
                        # update it in the catheter table and set gen_dose_rates to True for that catheter.
                        self[catheter_diff.index][dwell_diff.index].position = new_catheter_table[catheter_diff.index][dwell_diff.index].position
                        self[catheter_diff.index][dwell_diff.index].rotation = new_catheter_table[catheter_diff.index][dwell_diff.index].rotation
                        self[catheter_diff.index].gen_dose_rates = True
                    elif dwell_diff.time != 0:
                        self.time_diffs[
                            dwell_diff.name_id
                            ] = dwell_diff.time
                    else:
                        continue

def load_delivered_cathetertable_from_dicom(pth_dicom: Path) -> list:
    r"""
    Purpose:
        - Load the catheter table from a dicom file.
    Inputs:
        - pth_dicom: Path := the path to the dicom file containing the catheter table.
    Outputs:
        - Void := will update the catheter table based on the dicom file.
    """
    import pydicom

    plan = pydicom.dcmread(pth_dicom)
    catheter_table = []
    # loop through the channels
    for catheter_dcm in plan.ApplicationSetupSequence[0].ChannelSequence:
        control_points = []
        catheter_time = (
            float(catheter_dcm.ChannelTotalTime)
            if hasattr(catheter_dcm, "ChannelTotalTime")
            else 0
        )
        channel_final_time_weight = (
            float(catheter_dcm.FinalCumulativeTimeWeight)
            if hasattr(catheter_dcm, "FinalCumulativeTimeWeight")
            else 0
        )
        # loop through the control points.
        # Each dwell position has 2 control points, get them all.
        for control_point_dcm in catheter_dcm.BrachyControlPointSequence:
            if control_point_dcm.CumulativeTimeWeight is None:
                continue

            cumulative_time_weight = (
                float(control_point_dcm.CumulativeTimeWeight)
                if hasattr(control_point_dcm, "CumulativeTimeWeight")
                else 0
            )
            control_points.append(
                {
                    "index": (
                        int(control_point_dcm.ControlPointIndex)
                        if hasattr(control_point_dcm, "ControlPointIndex")
                        else None
                    ),
                    "angle": (
                        control_point_dcm.ControlPointShieldAngle
                        if hasattr(control_point_dcm, "ControlPointShieldAngle")
                        else 0
                    ),
                    "position": (
                        np.array(
                            control_point_dcm.ControlPoint3DPosition,
                            dtype=np.float32,
                        )
                        if hasattr(control_point_dcm, "ControlPoint3DPosition")
                        else None
                    ),
                    "relativePos": (
                        float(control_point_dcm.ControlPointRelativePosition)
                        if hasattr(
                            control_point_dcm, "ControlPointRelativePosition"
                        )
                        else None
                    ),
                    "rotation": (
                        np.array(
                            control_point_dcm.ControlPointOrientation,
                            dtype=np.float32,
                        )
                        if hasattr(control_point_dcm, "ControlPointOrientation")
                        else np.array([0, 0, 0], dtype=np.float32)
                    ),
                    "cumulative_weight": cumulative_time_weight,
                    # "total rerence air kerma": total_reference_air_kerma,
                }
            )
        catheter_table.append(
            {
                "index": int(catheter_dcm.ChannelNumber) - 1,
                "points": [],
                "channel_total_time": catheter_time,
                "channel_final_time_weight": channel_final_time_weight,
                "control_points": control_points,
            }
        )

    # # Convert control points to dwell positions:
    # # after extracting the final cummulative time weight of the catheters,
    # # the time of the catheter, and the cummulative time weight of the control points,
    # # we need to calculate the dwell time and time weight of the dwell positions.
    # # the formula is:
    # #     time_weight = (cumulative_time_weight - previous_cumulative_time_weight) / channel_final_time_weight
    # #     dwell time = time_weight * channel_total_time
    # #     dwell weight = dwell time / sum(channel_total_time)
    # get total treatment time
    treatment_time = np.sum(
        [catheter["channel_total_time"] for catheter in catheter_table]
    )
    final_catheter_table = []
    empty_catheter_counter = 0
    # loop through the catheters
    for catheter in catheter_table:
        dwells = []
        # loop through the control points
        # each dwell position has 2 control points:
        #   arrive time and depart time for the source
        for idx, control_point in enumerate(catheter["control_points"]):
            # if idx == len(catheter["control_points"]) - 1:
            #     break
            if idx % 2 == 1:
                continue
            dwell_time_weight = (
                catheter["control_points"][idx + 1]["cumulative_weight"]
                - control_point["cumulative_weight"]
            ) / catheter["channel_final_time_weight"]
            dwell_time = dwell_time_weight * catheter["channel_total_time"]
            dwell_weight = dwell_time / treatment_time
            dwells.append(
                {
                    "index": int(control_point["index"] / 2),
                    "angle": float(control_point["angle"]),
                    "position": [
                        control_point["position"][0],
                        control_point["position"][1],
                        control_point["position"][2],
                    ],
                    "relativePos": int(control_point["relativePos"]),
                    "rotation": [
                        control_point["rotation"][0],
                        control_point["rotation"][1],
                        control_point["rotation"][2],
                    ],
                    "time": dwell_time,
                    "weight": dwell_weight,
                }
            )
        # do not add empty catheters here
        if not dwells:
            empty_catheter_counter += 1
            continue
        catheter["dwells"] = dwells
        if (
            np.all([np.all(catheter["dwells"][i]["rotation"] == [0,0,0])
                    for i in range(len(catheter["dwells"]))])
            and len(catheter["dwells"]) > 1
        ):
            for i in range(len(catheter["dwells"])):
                catheter["dwells"][i]["rotation"] = get_rotation_from_position(i, catheter["dwells"])
        catheter["index"] -= empty_catheter_counter
        final_catheter_table.append(catheter)
    return {
        "catheter_list": final_catheter_table,
        "treatment_time": treatment_time,
        "step_size": float(
            final_catheter_table[0]["dwells"][1]["relativePos"] 
            - final_catheter_table[0]["dwells"][0]["relativePos"]
            )
    }

def _load_single_dose_rate(
    pth_dose_rate:Path,
    load_uncertainty=False,
    dtype=np.float32
    )->'BrachyDose':
        from brachyutils.dose.dose_utils import BrachyDose
        return BrachyDose(
            pth_dose_file=pth_dose_rate,
            load_uncertainty=load_uncertainty,
            dtype=dtype)

def _write_single_dose_rate(
    dose_rate:'BrachyDose',
    dir_export: str | Path = None,
    dose_extension: str = None,
    file_name: str = None,
    ):
    r"""
    ### Purpose:
    to write out a single dose rate map and uncertainty to a file.
    ### Inputs:
    - dose_rate:= The BrachyDose object for the dose rate data.
    - dir_export:= the directory to which the dose rate maps will be exported
    - file_name:= The name of the file inside dir_export. Following the RapidBrachy standard, it should be
    "run_{catheter.index+1}{dwell.index+1}{angle}.seq.nrrd". if none, dose_rate.path.name is used.
    - dose_extension := the type of dose rate map to be exported. options are ".3ddose", ".minidos", or ".nrrd"
    ### Output:
    - Void := dose file is written to dir_export+f"/{file_name}.{dose_type}
    """
    if file_name is None:
        file_name = dose_rate.path.name.split(".")[0]
    if dose_extension is None:
        dose_extension = ".seq.nrrd"
    dir_export = Path(dir_export)
    pth_out = dir_export/(file_name+dose_extension)
    dose_rate.write_brachydose_to_file(pth_dose_file=pth_out)

def load_from_dicom(
    pth_dicom: Path,
    from_delivered_dwellpositions: bool = False,
    ) -> Tuple[Dict, Dict]:
    r"""
    ### Purpose:
    - Load the catheter table from a dicom file.

    ### Inputs:
    - pth_dicom: Path := the path to the dicom file containing the catheter table.
    - from_delivered_dwellpositions: bool := if true, the dwell positions inside the 
    catheters_dict will only be the ones with non-zero dwell times. If false, the
    dwell positions will be created from the digitization points.
    ### Outputs:
    cat_dict := a dictionary containing the following keys:
        - catheters_dict
        - step_size
    """
    
    if from_delivered_dwellpositions:
        catheter_table_dict = load_delivered_cathetertable_from_dicom(pth_dicom=pth_dicom)
    else:
        catheter_table_dict, _ = dicom_to_catheter_table(dir_dicom=pth_dicom.parent)

    # add catheter index to the dwells
    for catheter in catheter_table_dict["catheter_list"]:
        for dwell in catheter["dwells"]:
            dwell["catheter_index"] = catheter["index"]

    return catheter_table_dict

def load_from_json(pth_json: Path) -> list:
    r"""
    ### Purpose:
    - Load the catheter table from a json file.
    
    ### Inputs:
    - pth_json: Path := the path to the json file containing the catheter table.
    
    ### Outputs:
    - Void := will update the catheter table based on the json file.
    """
    raw_catheter_table: list = []
    with open(pth_json, "r") as json_file:
        cat_table = json.load(json_file)
        if isinstance(cat_table, list):
            catheter_table_list = cat_table
            step_size = catheter_table_list[0].get("step_size", None)
        elif isinstance(cat_table, dict):
            catheter_table_list = cat_table.get("catheter_list", None)
            if catheter_table_list is None:
                catheter_table_dict = cat_table.get("catheters_dict", None)
                catheter_table_list = list(catheter_table_dict.values())
            step_size = cat_table.get("step_size", None)
            non_zero_dwell_positions = cat_table.get("non_zero_dwell_positions", None)
        else:
            raise ValueError(f"contents of the catheter file {pth_json} should be a list or dictionary")
        if catheter_table_list is None:
            raise ValueError(f"catheter list is missing from file {pth_json}")

        for catheter_dict in catheter_table_list:
            raw_catheter_table.append(Catheter(**catheter_dict))
        return {
            "catheter_list":raw_catheter_table,
            "step_size":step_size,
            "non_zero_dwell_positions": non_zero_dwell_positions
            }
