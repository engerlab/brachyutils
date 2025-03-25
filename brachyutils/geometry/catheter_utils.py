import numpy as np
from typing import List, Union, Dict
from pathlib import Path
from pydantic import BaseModel, model_validator, computed_field

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

    # @computed_field()
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
            "position": {
                "x": float(self.position[0]),
                "y": float(self.position[1]),
                "z": float(self.position[2]),
            },
            "relativePos": int(self.relativePos),
            "rotation": {
                "x": float(self.rotation[0]),
                "y": float(self.rotation[1]),
                "z": float(self.rotation[2]),
            },
            "time": float(self.time),
            "weight": float(self.weight(total_time)),
        }

class Catheter (BaseModel):
    r"""
    ### Purpose:
        - This class holds the information regarding a catheter.
    
    ### Attributes:
        - iD:int := the id of the catheter.
        - points:List[np.array] := the list of points of the catheter.
        - dwells:List[DwellPosition] := the list of dwell positions of the catheter.
        - channel_total_time:float := the total time of the catheter.
        - afterloader_channel_number:int := the afterloader channel number of the catheter.

    ### Functions:
        - to_dict() -> dict := convert the catheter to a dictionary.
    """
    iD: int
    dwells: List[DwellPosition]
    points: List[List[float]] = None
    afterloader_channel_number: int = None

    @computed_field
    def channel_total_time(self) -> float:
        r"""
        ### Purpose:
            - To calculate the total time of the catheter by summing over individual dwell times.
        
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
        if isinstance(all_inputs["dwells"][0], dict):
            all_inputs["dwells"] = [DwellPosition(**dwell) for dwell in all_inputs["dwells"]]
        # if "channel_total_time" not in all_inputs:
        #     all_inputs["channel_total_time"] = np.sum([dwell.time for dwell in all_inputs["dwells"]])
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
            "id": self.iD,
            "points": self.points,
            "dwells": [dwell.to_dict(total_time) for dwell in self.dwells],
            "channel_total_time": self.channel_total_time,
        }

class CatheterTable:
    r"""
    Purpose:
        - This class holds the information regarding the catheter table.
    Attributes:
        - catheter_list : List[Catheter] := the list of catheter objects in the catheter table.
    Functions:
        - load_from_json(pth_json:Path) -> list
        - load_from_dicom(pth_dicom:Path) -> list
    """

    def __init__(
        self,
        catheter_list: List[Union[Catheter, dict]] = None,
        pth_catheter_table: Path = None,
    ) -> None:
        r"""
        Purpose:
            - Initialize the CatheterTable object. from a list or a file. please provide only one of the inputs.
        Inputs:
            - catheter_list:List[Catheter] := the list of catheters in the catheter table.
            - pth_catheter_table:Path := the path to the catheter table file, which could be
            a dicom plan or a json file.
        """
        assert (catheter_list is not None) != (
            pth_catheter_table is not None
        ), "Either the catheter list or the path to the catheter table should be provided."

        if pth_catheter_table is not None:
            assert os.path.exists(
                pth_catheter_table
            ), f"The input json file does not exist: {pth_catheter_table}"
            extension = os.path.splitext(pth_catheter_table)[1]
            if extension == ".json":
                catheter_list = self.load_from_json(pth_catheter_table)
            elif extension == ".dcm":
                catheter_list = self.load_from_dicom(pth_catheter_table)
        if isinstance(catheter_list[0], dict):
            catheter_list = [
                Catheter(catheter_dict=catheter_dict) for catheter_dict in catheter_list
            ]

        assert isinstance(
            catheter_list[0], Catheter
        ), "The catheter list should contain Catheter objects."
        self.catheter_list: list = catheter_list

    def __iter__(self):
        for catheter in self.catheter_list:
            yield catheter

    def load_from_json(self, pth_json: Path) -> list:
        r"""
        Purpose:
            - Load the catheter table from a json file.
        Inputs:
            - pth_json: Path := the path to the json file containing the catheter table.
        Outputs:
            - Void := will update the catheter table based on the json file.
        """
        raw_catheter_table: list = []
        with open(pth_json, "r") as json_file:
            catheter_table_list = json.load(json_file)
            assert isinstance(
                catheter_table_list, list
            ), "The json file, should contain a list of catheters."
            for catheter_dict in catheter_table_list:
                raw_catheter_table.append(Catheter(catheter_dict=catheter_dict))
            return raw_catheter_table

    def load_from_dicom(self, pth_dicom: Path) -> list:
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
                    "id": int(catheter_dcm.ChannelNumber) - 1,
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
                        "position": {
                            "x":control_point["position"][0],
                            "y":control_point["position"][1],
                            "z":control_point["position"][2]
                            },
                        "relativePos": int(control_point["relativePos"]),
                        "rotation": {
                            "x":control_point["rotation"][0],
                            "y":control_point["rotation"][1],
                            "z":control_point["rotation"][2]
                            },
                        "time": dwell_time,
                        "weight": dwell_weight,
                    }
                )
            catheter["dwells"] = dwells
            if (
                np.all([np.all(list(catheter["dwells"][i]["rotation"].values()) == [0,0,0])
                        for i in range(len(catheter["dwells"]))])
                and len(catheter["dwells"]) > 1
            ):
                for i in range(len(dwells)):
                    dwells[i]["rotation"] = _get_rotation_from_position(i, dwells)
    
            final_catheter_table.append(Catheter(catheter_dict=catheter))
        return final_catheter_table

    def get_treatment_time(self) -> float:
        r"""
        Purpose:
            - To calculate the total treatment time.
        Inputs:
            - catheter_table:CatheterTable := the catheter table object.
        Outputs:
            - float := the total treatment time.
        """
        return np.sum([catheter.channel_total_time for catheter in self.catheter_list])

    def to_dict(self) -> dict:
        r"""
        Purpose:
            - To convert the catheter table to a dictionary.
        Inputs:
            - self := the CatheterTable object.
        Outputs:
            - dict := the dictionary containing the catheter table.
        """
        return [catheter.to_dict() for catheter in self.catheter_list]

    def info(self) -> None:
        r"""
        Purpose:
            - To print the information about the catheter table.
        """
        # print(self.to_dict())
        print("Catheter table info is as follows:")
        print(f"Number of catheters: {len(self.catheter_list)}")
        for catheter in self.catheter_list:
            print(f"Catheter ID: {catheter.id}")
            print(f"Number of dwell positions: {len(catheter.dwells)}")
            print(f"Total channel time: {catheter.channel_total_time}")

def _get_rotation_from_position(idx, control_points):
    r"""
    Purpose:
        - To get the rotation of the dwell point from the position of the dwell point.
    Inputs:
        - idx:int := the index of the dwell point.
        - control_point_dcm:pydicom.dataset.Dataset := the control point object.
    Outputs:
        - np.array := the rotation of the dwell point in each axis.
    """
    # TODO: Merge this dicom utils script with my catheter setup class.
    # We need all dwell positions, not only the non 0s ones to be able to 
    # compute correct angles when they are not provided by the DICOM.
    if len(control_points) == 2:
        return _angle_betwen_2_points(
            np.array(list(control_points[1]["position"].values()), dtype=np.float32),
            np.array(list(control_points[0]["position"].values()), dtype=np.float32),
        )

    if idx == 0:
        return _get_rotation_from_position(idx+1, control_points)
    elif idx == len(control_points) - 1:
        return _get_rotation_from_position(idx-1, control_points)
    else:
        return _angle_betwen_2_points(
            np.array(list(control_points[idx-1]["position"].values()), dtype=np.float32),
            np.array(list(control_points[idx+1]["position"].values()), dtype=np.float32),
        )


def _angle_betwen_2_points(a, b) -> dict:
    r"""
    Purpose:
        - To calculate the angle between two points.
    Inputs:
        - a:np.array := the first point.
        - b:np.array := the second point.      
    Outputs:
        - np.array := the angle between the two points in each axis.
    """
    vec = a - b
    normal = np.sqrt(np.sum(vec ** 2))
    angle_np = vec / normal
    return {"x":angle_np[0], "y":angle_np[1], "z":angle_np[2]}

