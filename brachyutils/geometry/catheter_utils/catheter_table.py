import copy
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, computed_field, ConfigDict
import json
import SimpleITK as sitk
from opentps.core.processing.imageProcessing.sitkImageProcessing import imageToSITK
from opentps.core.data.images import ROIMask
from brachyutils.geometry.phantom_utils import BrachyPhantom

from brachyutils.geometry.catheter_utils.digitization.pw_linear_interpolator import PiecewiseLinear3D
from brachyutils.geometry.catheter_utils.digitization.spline_interpolator import NeedleSplineCreator
from brachyutils.geometry.catheter_utils.catheter_setup import (
    get_rotation_from_position, CatheterSetUp, dilate_mask_in_mm
)
from catheter_api import (
    dicom_to_catheter_table, _update_catheter_table, CreatedSetUp
)

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
    
    ### Functions:
    - to_dict() -> dict := convert the dwell position to a dictionary.
    """
    index: int
    angle: float = 0.0
    position: List[float] | Dict[str, float]
    relativePos: float
    rotation: List[float] | Dict[str, float]
    time: float = 0.0
    # weight: float = None

    def __init__(self, **data):
        r"""
        ### Purpose:
        - To initialize the DwellPosition object.
        ### Inputs:
        - **data: dict := the dictionary containing the dwell position attributes.
        """
        super().__init__(**data)
        # convert position and rotation to lists if they are dictionaries
        if isinstance(self.position, dict):
            self.position = list(self.position.values())
        if isinstance(self.rotation, dict):
            self.rotation = list(self.rotation.values())
    
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
    points: Optional[List[List[float]]] = None  # to keep compatibility with previous versions

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

    def __init__(self, **data):
        r"""
        ### Purpose:
        - To initialize the Catheter object.
        
        ### Inputs:
        - **data: dict := the dictionary containing the catheter attributes.
        """
        super().__init__(**data)
        # Set digitization_points from points if digitization_points is None
        if self.points is not None and self.digitization_points is None:
            self.digitization_points = self.points

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
            "index": self.index,
            "dwells": [dwell.to_dict(total_time) for dwell in self.dwells],
            # "fit_function": self.fit_function,
            "tip_position": self.tip_position,
            "last_dwell_coordinate": self.last_dwell_coordinate,
            "step_size": self.step_size,
            "digitization_points": self.digitization_points,
            "afterloader_channel_number": self.afterloader_channel_number,
            "insert_position": self.insert_position,
            "channel_total_time": self.channel_total_time,
            "channel_length": self.channel_length
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

    @classmethod
    def get_dwells_from_fit(
        cls,
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
    
    @classmethod
    def get_fit_from_points(cls, points:List[List[float]]) -> PiecewiseLinear3D:
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
    - catheter_list : List[Catheter] := the list of catheter objects in the catheter table.
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
    ##########
    ## To enable using CatheterSetUp and CreatedSetUp as a data types, default is False.
    # This is not a parameter to be provided.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ##########

    catheter_list: List[Catheter] | List[dict] | str | Path | CatheterSetUp | CreatedSetUp
    step_size: float = 5.0
    from_delivered_dwellpositions: bool = False

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
        return np.sum([catheter.channel_total_time for catheter in self.catheter_list])

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
        return len(self.catheter_list)
    
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
        return np.sum([len(catheter.dwells) for catheter in self.catheter_list])

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
        for catheter in self.catheter_list:
            dwell_positions = []
            for dwell in catheter.dwells:
                if dwell.time > 0.0:
                    dwell_positions.append(dwell.position)
            non_zero_dwell_positions[f"Needle_{catheter.index}"] = dwell_positions
        return non_zero_dwell_positions

    def __init__(self, **data):
        r"""
        ### Purpose:
        - To initialize the CatheterTable object.
        
        ### Inputs:
        - **data: dict := the dictionary containing the catheter table attributes.
        """
        super().__init__(**data)
        if (isinstance(self.catheter_list, str) or
            isinstance(self.catheter_list, Path)
            ):
            catheter_file = Path(self.catheter_list)

            if not catheter_file.exists():
                raise ValueError(f"catheter file {catheter_file} does not exist.")
            if str(catheter_file).endswith(".mrk.json"):
                # if the file is a slicer markup file, load it as a json file
                raise NotImplementedError("this feature is not implemented yet.")

            if str(catheter_file).endswith(".json"):
                cat_dict = self.load_from_json(catheter_file)
            elif str(catheter_file).endswith(".dcm"):
                cat_dict = self.load_from_dicom(
                    pth_dicom=catheter_file,
                    from_delivered_dwellpositions=self.from_delivered_dwellpositions,
                )
            self.catheter_list = cat_dict["catheter_list"]
            self.step_size = cat_dict["step_size"]

        elif isinstance(self.catheter_list, CatheterSetUp):
            # if the catheter_list is a CatheterSetUp object, convert it to a CatheterTable
            cat_setup = self.catheter_list
            updated_catheter_dict = _update_catheter_table(
                catheter_table = cat_setup.catheter_table,
                digitization_points=cat_setup.digitization_points,
                fit_function=cat_setup.piece_wise_lines,
                tips=cat_setup.get_tips_coords(),
                step_size=cat_setup.step_size,
            )
            self.catheter_list = updated_catheter_dict["catheter_list"]
            self.step_size = updated_catheter_dict["step_size"]

        elif isinstance(self.catheter_list, CreatedSetUp):
            created_setup = self.catheter_list
            updated_catheter_dict = created_setup.to_brachyutils_CatheterTable_format()
            self.catheter_list = updated_catheter_dict["catheter_list"]
            self.step_size = updated_catheter_dict["step_size"]
            self.non_zero_dwell_positions = created_setup.get_non_zero_dwell_positions()

        if isinstance(self.catheter_list[0], dict):
            self.catheter_list = [
                Catheter(**catheter_dict) for catheter_dict in self.catheter_list
            ]

    def __iter__(self):
        for catheter in self.catheter_list:
            yield catheter

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
                for catheter in self.catheter_list
                ],
            "step_size": self.step_size,
            "treatment_time": treatment_t
        }
    def info(self) -> None:
        r"""
        ### Purpose:
        - To print the information about the catheter table.
        """
        print("Catheter table info is as follows:")
        print(f"Number of catheters: {len(self.catheter_list)}")
        print(f"Total treatment time: {self.treatment_time}")
        for catheter in self.catheter_list:
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

    def write_to_slicer_markup(self, pth_mrk_json: Path | str, **kwargs) -> None:
        r"""
        ### Purpose:
        - Write the catheter table to a json file in the slicer markup format.
        
        ### Inputs:
        - pth_json: Path := the path to the json file where the catheter table will be written.
        
        ### Outputs:
        - Void := will write the catheter table to a json file in the slicer markup format.
        """
        from brachyutils.geometry.catheter_utils.utils import create_slicer_markup_points
        pth_mrk_json = Path(pth_mrk_json)
        if not str(pth_mrk_json).endswith(".mrk.json"):
            raise ValueError("The output file name should end with .mrk.json")
        pth_mrk_json.parent.mkdir(parents=True, exist_ok=True)

        point_list = [catheter.digitization_points for catheter in self]
        
        create_slicer_markup_points(
            output_path=str(pth_mrk_json),
            point_list=point_list,
            color=kwargs.get("color", None),
            remove_text=kwargs.get("remove_text", True),
        )

    @classmethod
    def load_from_json(cls, pth_json: Path) -> list:
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

    @classmethod
    def load_from_dicom(
        cls,
        pth_dicom: Path,
        from_delivered_dwellpositions: bool = False,
        ) -> Tuple[Dict, Dict]:
        r"""
        ### Purpose:
        - Load the catheter table from a dicom file.

        ### Inputs:
        - pth_dicom: Path := the path to the dicom file containing the catheter table.
        - from_delivered_dwellpositions: bool := if true, the dwell positions inside the 
        catheter_list will only be the ones with non-zero dwell times. If false, the
        dwell positions will be created from the digitization points.
        ### Outputs:
        cat_dict := a dictionary containing the following keys:
            - catheter_list
            - step_size
        """
        
        if from_delivered_dwellpositions:
            catheter_table_dict = load_delivered_cathetertable_from_dicom(pth_dicom=pth_dicom)
        else:
            catheter_table_dict, _ = dicom_to_catheter_table(dir_dicom=pth_dicom.parent)
        return catheter_table_dict

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
        for catheter in self.catheter_list:
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

        for catheter in self.catheter_list:
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

        for catheter in self.catheter_list:
            catheter.remove_outside_mask(mask)

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