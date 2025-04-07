import numpy as np
from typing import List, Union, Dict, Any
from pathlib import Path
from pydantic import BaseModel, model_validator, computed_field
import json
from opentps.core.data.images import ROIMask
from brachyutils.planning.simulation_utils import BrachySource
class DwellPosition(BaseModel):
    r"""
    ### Purpose:
        - This class holds the information regarding a dwell position.

    ### Attributes:
        - index
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
    time: float
    # weight: float = None

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

    @model_validator(mode="before")
    def finish_initialization(cls, all_inputs):
        r"""
        ### Purpose:
            - If the position and rotation are provided as dictionaries, convert
            them to lists.
        """
        if isinstance(all_inputs["position"], dict):
            all_inputs["position"] = list(all_inputs["position"].values())
        if isinstance(all_inputs["rotation"], dict):
            all_inputs["rotation"] = list(all_inputs["rotation"].values())
        return all_inputs

    def to_dict(self, total_time) -> dict:
        r"""
        Purpose:
            - To convert the dwell position to a dictionary.
        Inputs:
            - self := the DwellPosition object.
        Outputs:
            - dict := the dictionary containing the dwell position.
        """
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
        - This class holds the information regarding a catheter.
    
    ### Attributes:
        - index:int := the index of the catheter.
        - points:List[np.array] := the list of digitization points of the catheter.
        - dwells:List[DwellPosition] := the list of dwell positions of the catheter.
        - afterloader_channel_number:int := the afterloader channel number of the catheter.
        - channel_total_time:float := the total time of the catheter.

    ### Functions:
        - to_dict() -> dict := convert the catheter to a dictionary.
        - add_dwell(dwell:DwellPosition) -> None := add a dwell position to the catheter.
    """
    index: int
    dwells: List[DwellPosition] = None
    points: List[List[float]] = None
    afterloader_channel_number: int = None
    fit_function:Any = None
    tip_position: List[float] = None
    insert_position: List[float] = None

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

    @model_validator(mode="before")
    def finish_initialization(cls, all_inputs):
        r"""
        ### Purpose:
            - To conver the list of dwell dictionaries to a list of DwellPosition objects.
            - extract the channel_total_time from the dwells if it is not provided.
        """
        # load in the dwell positions directry
        if all_inputs.get("dwells", None) is not None:
            if isinstance(all_inputs["dwells"][0], dict):
                all_inputs["dwells"] = [DwellPosition(**dwell) for dwell in all_inputs["dwells"]]

        if all_inputs.get("points", None) is not None:        
        # TODO: generate fit from points.
        # TODO: generate dwell positions from fit.
            pass

        return all_inputs

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
            "points": self.points,
            "dwells": [dwell.to_dict(total_time) for dwell in self.dwells],
            "channel_total_time": self.channel_total_time,
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
    def get_fit_from_points(cls, points:List[List[float]]) -> List[List[float]]:
        r"""
        ### Purpose:
            - To generate a spline from a list of points.

        ### Inputs:
            - points:List[List[float]] := the list of points to generate the spline from.

        ### Outputs:
            - List[List[float]] := the list of points on the spline.
        """
        raise NotImplementedError("This function is not implemented yet.")

    @classmethod
    def get_dwells_from_fit(cls, spline:List[List[float]]) -> List[DwellPosition]:
        r"""
        ### Purpose:
            - To generate dwell positions from a spline.

        ### Inputs:
            - spline:List[List[float]] := the list of points on the spline.

        ### Outputs:
            - List[DwellPosition] := the list of dwell positions.
        """
        raise NotImplementedError("This function is not implemented yet.")

    @classmethod
    def get_contours_from_points(cls, points:List[List[float]]) -> ROIMask:
        r"""
        ### Purpose:
            - To generate contours from a list of points.

        ### Inputs:
            - points:List[List[float]] := the list of points to generate the contours from.

        ### Outputs:
            - ROIMask := the contours generated from the points.
        """
        raise NotImplementedError("This function is not implemented yet.")
    
class CatheterTable(BaseModel):
    r"""
    ### Purpose:
    - This class holds the information regarding the catheter table.

    ### Attributes:
    - catheter_list : List[Catheter] := the list of catheter objects in the catheter table.
    - step_size: float := the step size in mm between the dwell positions on the catheter table.
    - brachy_source: BrachySource = None
    - treatment_time: float = None := the total treatment time of the catheter table.
    this attributed is computed from the catheter list.
       
    ### Functions:
    - load_from_json(pth_json:Path) -> list
    - load_from_dicom(pth_dicom:Path) -> list
    """
    step_size: float = 5.0
    catheter_list: List[Catheter] | List[dict] | str | Path
    brachy_source: BrachySource = None
    channel_length: float = None

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
    
    @model_validator(mode="before")
    def finish_initialization(cls, all_inputs):
        r"""
        ### Purpose:
            - To handle the different types of inputs for the catheter list.
            if a file path or a string is provided, load the catheter table from the json or dicom file.
        """
        if (isinstance(all_inputs["catheter_list"], str) or
            isinstance(all_inputs["catheter_list"], Path)
            ):
            catheter_file = Path(all_inputs["catheter_list"])
            if str(catheter_file).endswith(".json"):
                all_inputs["catheter_list"] = cls.load_from_json(catheter_file)
            elif str(catheter_file).endswith(".dcm"):
                all_inputs["catheter_list"] = cls.load_from_dicom(pth_dicom=catheter_file)

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
        Purpose:
            - To convert the catheter table to a dictionary.
        Inputs:
            - self := the CatheterTable object.
        Outputs:
            - dict := the dictionary containing the catheter table.
        """
        return [catheter.to_dict(total_time=self.treatment_time) for catheter in self.catheter_list]

    def info(self) -> None:
        r"""
        Purpose:
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
            catheter_table_list = json.load(json_file)
            assert isinstance(
                catheter_table_list, list
            ), "The json file, should contain a list of catheters."
            for catheter_dict in catheter_table_list:
                raw_catheter_table.append(Catheter(**catheter_dict))
            return raw_catheter_table

    @classmethod
    def load_from_dicom(cls, pth_dicom: Path) -> List[dict]:
        r"""
        ### Purpose:
            - Load the catheter table from a dicom file.
        
        ### Inputs:
            - pth_dicom: Path := the path to the dicom file containing the catheter table.
        
        ### Outputs:
            - Void := will update the catheter table based on the dicom file.
        """
        try:
            from ai_assisted_brachy.catheter.catheter_api import dicom_to_catheter_table
        except:
            from ai_assisted_brachy.catheter.catheter_api import dicom_to_catheter_table            
        catheter_table_dict, _ = dicom_to_catheter_table(dir_dicom=pth_dicom.parent)
        return catheter_table_dict
