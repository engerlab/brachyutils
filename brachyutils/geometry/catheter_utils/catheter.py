import numpy as np
from pydantic import BaseModel, computed_field, model_validator
from typing import List, Union, Any, Optional
import SimpleITK as sitk
import copy
from brachyutils.geometry.catheter_utils.dwell_position import DwellPosition
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.pw_linear_interpolator import PiecewiseLinear3D
from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.digitization.spline_interpolator import NeedleSplineCreator
from opentps.core.data.images import ROIMask
from opentps.core.processing.imageProcessing.sitkImageProcessing import imageToSITK

from brachyutils.geometry.catheter_utils.patch_ai_assisted_brachy.catheter_setup import (
    get_rotation_from_position
)


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
    digitization_points: Optional[List[List[float]]] = None
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

    @computed_field
    def num_dwell_positions(self) -> int:
        r"""
        ### Purpose:
        - To calculate the number of dwell position in a catheter
        ### Inputs:
        - self := the Catheter object.
        ### Outputs:
        - int := the number of dwell positions in the catheter.
        """
        return len(self.dwells)

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
            points=[self.tip_position, self.last_dwell_coordinate]
            self.fit_function = self.get_fit_from_points(
                points=points
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
            "tip_position": [round(float(x), 3) for x in self.tip_position],
            "last_dwell_coordinate": [round(float(x), 3) for x in self.last_dwell_coordinate],
            "step_size": round(float(self.step_size), 3),
            "digitization_points": [[round(float(x), 3) for x in point] for point in self.digitization_points] if self.digitization_points else None,
            "afterloader_channel_number": int(self.afterloader_channel_number) if self.afterloader_channel_number is not None else None,
            "insert_position": [round(float(x), 3) for x in self.insert_position] if self.insert_position else None,
            "channel_total_time": round(float(self.channel_total_time), 3),
            "channel_length": round(float(self.channel_length), 3) if self.channel_length is not None else None
        }

    def add_dwell(self, dwell:DwellPosition) -> None:
        r"""
        ### Purpose:
        - Insert a dwell position to the catheter and update the necessary attributes.
        XXX need to check if relativePos is correct. for now, just append the dwell 
        ### Inputs:
        - self := the Catheter object.
        - dwell:DwellPosition := the dwell position to be added.
        """
        self.dwells.append(dwell)

    def get_dwells_from_fit(
        self,
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
            dwell_index = 0
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
                        "catheter_index": self.index
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
    
    def get_fit_from_points(self, points:List[List[float]]) -> PiecewiseLinear3D:
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
