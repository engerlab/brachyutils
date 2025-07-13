import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, model_validator, computed_field
import json
from opentps.core.processing.imageProcessing.sitkImageProcessing import imageToSITK
from brachyutils.geometry.phantom_utils import BrachyPhantom

from ai_assisted_brachy.catheter.digitization.pw_linear_interpolator import PiecewiseLinear3D
from ai_assisted_brachy.catheter.digitization.spline_interpolator import NeedleSplineCreator
from ai_assisted_brachy.catheter.catheter_setup import get_rotation_from_position
from ai_assisted_brachy.catheter.catheter_api import dicom_to_catheter_table, CatheterSetUp
from ai_assisted_brachy.catheter.catheter_api import ct_to_catheter_table

class DwellPosition(BaseModel):
    r"""
    ### Purpose:
    - This class holds the information regarding a dwell position.

    ### Attributes:
    - index: int := the index of the dwell position along the catheter
    - angle := angle of the IMBT shield
    - position:dict: np.array := dwell position in the patient coordinate system [x, y, z]
    - relativePos: int := dwell coordinate along the catheter from the reference point. increments of 5 mm
    - rotation: np.array := rotation of the dwell position in the patient coordinate system [x, y, z]
    - time: float := dwell time for this dwell position
    - weight: float := ratio of this dwell time over the sum of all dwell times in all catheters.
    
    ### Functions:
    - to_dict() -> dict := convert the dwell position to a dictionary.
    """
    index: int
    angle: float = 0.0
    position: List[float] | Dict[str, float]
    relativePos: int
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
            "relativePos": int(self.relativePos),
            "rotation": list(self.rotation),
            "time": float(self.time),
            "weight": float(self.weight(total_time)),
        }

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
    - points:List[np.array] := the list of digitization points of the catheter.
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
    # in case dwells are missing and fit, tip, last dwell position and step size is provided
    fit_function:Any = None
    tip_position: List[float] = None
    last_dwell_coordinate: List[float] = None
    step_size: float = 5.0
    # in case dwells and fit is missing and digitization points are provided.
    # we assume tip is the first digitization point and last dwell is the last digitization point.
    points: List[List[float]] = None
    # auxiliary attributes
    afterloader_channel_number: Optional[int] = None # if none, will be set to index
    insert_position: Optional[List[float]] = None

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
        # Set afterloader_channel_number to index if not provided
        if self.afterloader_channel_number is None:
            self.afterloader_channel_number = self.index

        # Initialize dwells based on available inputs
        if self.dwells is not None:
            # Convert dict dwells to DwellPosition objects if needed
            if isinstance(self.dwells[0], dict):
                self.dwells = [DwellPosition(**dwell) for dwell in self.dwells]
        elif self.fit_function is not None:
            # Create dwells from fit function
            self.dwells = self.get_dwells_from_fit(
                fit_function=self.fit_function,
                step_size=self.step_size,
            )
        elif self.points is not None:
            # Create fit and dwells from points
            self.fit_function = self.get_fit_from_points(points=self.points)
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
        if self.dwells is not None and len(self.dwells) > 0:
            self.tip_position = self.dwells[0].position
            self.last_dwell_coordinate = self.dwells[-1].position

    # @model_validator(mode="before")
    # def finish_initialization(cls, all_inputs):
    #     r"""
    #     ### Purpose:
    #     - To conver the list of dwell dictionaries to a list of DwellPosition objects.
    #     - extract the channel_total_time from the dwells if it is not provided.
    #     """
    #     # load in the dwell positions directry
    #     if all_inputs.get("dwells", None) is not None:
    #         if isinstance(all_inputs["dwells"][0], dict):
    #             all_inputs["dwells"] = [
    #                 DwellPosition(**dwell) for dwell in all_inputs["dwells"]
    #                 ]
    #     # create dwells from fit, tip, last dwell position and step size
    #     elif all_inputs.get("fit_function", None) is not None:
    #         all_inputs["dwells"] = cls.get_dwells_from_fit(
    #             fit_function=all_inputs["fit_function"],
    #             step_size=all_inputs.get("step_size",5.0),
    #             )
    #     # create the fit and digitization from points
    #     elif all_inputs.get("points", None) is not None:
    #         all_inputs["fit_function"] = cls.get_fit_from_points(
    #             points=all_inputs["points"],
    #         )
    #         all_inputs["dwells"] = cls.get_dwells_from_fit(
    #             fit_function=all_inputs["fit_function"],
    #             step_size=all_inputs.get("step_size",5.0),
    #         )
    #     elif (all_inputs.get("tip_position", None) is not None
    #           and all_inputs.get("last_dwell_coordinate", None) is not None
    #     ):
    #         all_inputs["fit_function"] = cls.get_fit_from_points(
    #             points=[all_inputs["tip_position"], all_inputs["last_dwell_coordinate"]],
    #         )
    #         all_inputs["dwells"] = cls.get_dwells_from_fit(
    #             fit_function=all_inputs["fit_function"],
    #             step_size=all_inputs.get("step_size",5.0),
    #         )
    #     else:
    #         raise ValueError("""Either provide dwells, fit_function, points or
    #         tip and last dwell coordinate coordinates to the create a catheter.""")

    #     all_inputs["tip_position"] = all_inputs["dwells"][0].position
    #     all_inputs["last_dwell_coordinate"] = all_inputs["dwells"][-1].position

    #     return all_inputs

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
            "points": self.points,
            "afterloader_channel_number": self.afterloader_channel_number,
            "insert_position": self.insert_position,
            "channel_total_time": total_time,
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

class CatheterTable(BaseModel):
    r"""
    ### Purpose:
    - This class holds the information regarding the catheter table.

    ### Attributes:
    - catheter_list : List[Catheter] := the list of catheter objects in the catheter table.
    - step_size: float := the step size in mm between the dwell positions on the catheter table.
    - treatment_time: float = None := the total treatment time of the catheter table.
    this attributed is computed from the catheter list.
    - delivered_dwell_coordinates := The dictionary mapping each catheter to the list of 
    dwell positions that were actually used for plan delivery.
    - channel_length: float = None := the length of the catheter channel in mm.
    - num_catheters: int = None := the number of catheters in the catheter table.
    - num_dwell_positions: int = None := the number of dwell positions in the catheter table.
    ### Functions:
    - load_from_json(pth_json:Path) -> list
    - load_from_dicom(pth_dicom:Path) -> list
    """
    catheter_list: List[Catheter] | List[dict] | str | Path
    step_size: float = 5.0
    # brachy_source:Any = None
    channel_length: float = None
    delivered_dwell_coordinates: Dict[str, List[List[float]]] = None

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

    @model_validator(mode="before")
    def finish_initialization(cls, all_inputs):
        r"""
        ### Purpose:
        - To handle the different types of inputs for the catheter list.
        if a file path or a string is provided, load the catheter table from the json or dicom file.
        """
        delivered_dwell_coordinates = None
        if (isinstance(all_inputs["catheter_list"], str) or
            isinstance(all_inputs["catheter_list"], Path)
            ):
            catheter_file = Path(all_inputs["catheter_list"])

            if not catheter_file.exists():
                raise ValueError(f"catheter file {catheter_file} does not exist.")
            if str(catheter_file).endswith(".mrk.json"):
                # if the file is a slicer markup file, load it as a json file
                raise NotImplementedError("this feature is not implemented yet.")

            if str(catheter_file).endswith(".json"):
                cat_dict = cls.load_from_json(catheter_file)

            elif str(catheter_file).endswith(".dcm"):
                cat_dict, delivered_dwell_coordinates = cls.load_from_dicom(pth_dicom=catheter_file)
            elif catheter_file.is_dir():
                cat_dict = cls.load_from_dicom(pth_dicom=catheter_file, from_ct=True)

            all_inputs["catheter_list"] = cat_dict["catheter_list"]
            all_inputs["step_size"] = cat_dict["step_size"]
            all_inputs["channel_length"] = cat_dict["channel_length"]
            if delivered_dwell_coordinates is not None:
                all_inputs["delivered_dwell_coordinates"] = delivered_dwell_coordinates

        if isinstance(all_inputs["catheter_list"][0], dict):
            all_inputs["catheter_list"] = [
                Catheter(**catheter_dict) for catheter_dict in all_inputs["catheter_list"]
            ]
        return all_inputs

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
        return {
            "catheter_list": [
                catheter.to_dict(total_time=self.treatment_time) 
                for catheter in self.catheter_list
                ],
            "step_size": self.step_size,
            "channel_length": self.channel_length,
            "treatment_time": self.treatment_time
        }
    def info(self) -> None:
        r"""
        ### Purpose:
        - To print the information about the catheter table.
        """
        # print(self.to_dict())
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
        from ai_assisted_brachy.preprocessing.utils import create_slicer_markup_points
        pth_mrk_json = Path(pth_mrk_json)
        if not str(pth_mrk_json).endswith(".mrk.json"):
            raise ValueError("The output file name should end with .mrk.json")
        pth_mrk_json.parent.mkdir(parents=True, exist_ok=True)

        point_list = [catheter.points for catheter in self]
        
        create_slicer_markup_points(
            output_path=str(pth_mrk_json),
            point_list=point_list,
            color=kwargs.get("color", None),
            remove_text=kwargs.get("remove_text", True),
        )

    def get_delivered_catheter_table(self) -> "CatheterTable":
        r"""
        ### Purpose:
        - To get the catheter table with the dwell positions that were used for the treatment.
        ### Input:
        - self: an instant of CatheterTable object
        - delivered_dwell_coordinates: A dictonary mapping the catheters as keys (Needle_#) to the 
        list of dwell position coordinates [x, y, z] that had non zero dwell time in 
        the catheter table.
        ### Output:
        - delivered_catheter_table: CatheterTable := a catheter table where all the dwell positions
        were used in the clinic.
        """
        if self.delivered_dwell_coordinates is None:
            raise ValueError("delivered_dwell_coordinates is None. Please provide the delivered dwell coordinates.")

        delivered_catheter_list = []        
        for catheter, delivered_cat in zip(
            self.catheter_list, list(self.delivered_dwell_coordinates.values())):
            if len(delivered_cat) == 0:
                continue
            new_dwells = []
            for i, coordinate in enumerate(delivered_cat):
                dwell_position = list(
                    filter(
                        lambda dp : np.isclose(dp.position, coordinate).all(),
                        catheter.dwells)
                )
                if len(dwell_position) == 0:
                    raise ValueError("The delivered coordinate was not found the entire catheter table")
                if len(dwell_position) > 1:
                    raise ValueError("The delivered coordinate was found in multiple dwell positions")

                dwell_position = dwell_position[0]
                new_dwells.append(
                    DwellPosition(
                        index=i,
                        angle=dwell_position.angle,
                        position=dwell_position.position,
                        rotation=dwell_position.rotation,
                        relativePos=dwell_position.relativePos,
                        time=dwell_position.time,
                    )
                )

            delivered_catheter_list.append(
                Catheter(
                    index=catheter.index,
                    dwells=new_dwells,
                    fit_function=catheter.fit_function,
                    tip_position=catheter.tip_position,
                    last_dwell_coordinate=new_dwells[-1].position,
                    step_size=catheter.step_size,
                    points=catheter.points,
                    afterloader_channel_number=catheter.afterloader_channel_number,
                    insert_position=catheter.insert_position,
                    )
            )

        return CatheterTable(
            catheter_list=delivered_catheter_list,
            step_size=self.step_size,
            channel_length=self.channel_length,
            delivered_dwell_coordinates=self.delivered_dwell_coordinates,
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
                channel_length = catheter_table_list[0].get("channel_length", None)
            elif isinstance(cat_table, dict):
                catheter_table_list = cat_table.get("catheter_list", None)
                step_size = cat_table.get("step_size", None)
                channel_length = cat_table.get("channel_length", None)
                delivered_dwell_coordinates = cat_table.get("delivered_dwell_coordinates", None)
            else:
                raise ValueError(f"contents of the catheter file {pth_json} should be a list or dictionary")
            if catheter_table_list is None:
                raise ValueError(f"catheter list is missing from file {pth_json}")

            for catheter_dict in catheter_table_list:
                raw_catheter_table.append(Catheter(**catheter_dict))
            return {
                "catheter_list":raw_catheter_table,
                "step_size":step_size,
                "channel_length":channel_length,
                "delivered_dwell_coordinates": delivered_dwell_coordinates
                }

    @classmethod
    def load_from_dicom(cls, pth_dicom: Path, from_ct: bool = False) -> Tuple[Dict, Dict]:
        r"""
        ### Purpose:
        - Load the catheter table from a dicom file.

        ### Inputs:
        - pth_dicom: Path := the path to the dicom file containing the catheter table.
        - from_ct: bool = False := if True, catheters will be contoured on CT images, then digitized.

        ### Outputs:
        - catheter_table_dict := the dictionary containing the catheter table.
        - delivered_dwell_coordinates := maps "Needle_#" to the list of dwell position coordinates
        that were used to deliver a plan.
        """
        if from_ct:
            phantom = BrachyPhantom(dir_dicom=pth_dicom)
            catheter_table_dict = cls.load_from_phantom(image=phantom)
            catheter_setup = None
        else:
            catheter_table_dict, catheter_setup = dicom_to_catheter_table(dir_dicom=pth_dicom.parent)

        if catheter_setup is not None:
            delivered_dwell_coordinates = catheter_setup.non_zero_dwell_positions
        else:
            delivered_dwell_coordinates = None

        return catheter_table_dict, delivered_dwell_coordinates

    @classmethod
    def load_from_phantom(cls, image: Path | str | BrachyPhantom) -> dict:
        r"""
        ### Purpose:
        - Load the catheter table from a phantom object.
        
        ### Inputs:
        - image: Path | BrachyPhantom := the path to the phantom file or the phantom object.
        
        ### Outputs:
        - catheter_table_dict := the dictionary containing the catheter table.
        """
        # if "image" a path to a file, just pass along
        if isinstance(image, Path) or isinstance(image, str):
            image = Path(image)                
        # if "image" is a BrachyPhantom object, convert it to sitk and pass it along 
        elif isinstance(image, BrachyPhantom):
            image = imageToSITK(image.image_obj)
        else:
            raise ValueError("image should be either a Path or a BrachyPhantom object.")

        cat_table_dict = ct_to_catheter_table(image=image)
        return cat_table_dict
