import numpy as np
from typing import List, Union
from pathlib import Path


class DwellPosition:
    r"""
    Purpose:
        - This class holds the information regarding a dwell position.

    Attributes:
        - angle := angle of the IMBT shield
        - position:dict: np.array := dwell position in the patient coordinate system [x, y, z]
        - relativePos: int := dwell coordinate along the catheter from the reference point. increments of 5 mm
        - rotation: np.array := rotation of the dwell position in the patient coordinate system [x, y, z]
        - time: float := dwell time for this dwell position
        - weight: float := ratio of this dwell time over the sum of all dwell times in all catheters.
    """

    def __init__(
        self,
        index: int = None,
        angle: float = 0,
        position: np.array = None,
        relativePos: int = None,
        rotation: np.array = None,
        time: float = None,
        weight: float = None,
        dwell_dict: dict = None,
    ) -> None:
        r"""
        Purpose:
            - Initialize the DwellPosition object.
        Inputs:
            - index:int := the index of the dwell position.
            - angle:float := angle of the IMBT shield
            - position:np.array := dwell position in the patient coordinate system [x, y, z]
            - relativePos:int := dwell coordinate along the catheter from the reference point. increments of 5 mm
            - rotation:np.array := rotation of the dwell position in the patient coordinate system [x, y, z]
            - time:float := dwell time for this dwell position
            - weight:float := ratio of this dwell time over the sum of all dwell times in all catheters.
            - dwell_dict:dict := the dictionary containing the dwell position.
            either provide the index, angle, position, relativePos, rotation, time and weight or provide the dwell_dict. Not both.
        """
        assert (
            (index is not None)
            and (angle is not None)
            and (position is not None)
            and (relativePos is not None)
            and (rotation is not None)
            and (time is not None)
            and (weight is not None)
        ) != (
            dwell_dict is not None
        ), "Either provide index, angle, position, relativePos, rotation, time and weight or provide catheter_dict. Not both."

        if dwell_dict is not None:
            index = dwell_dict.get("index", None)
            angle = float(dwell_dict.get("angle"))
            position = np.array(
                [
                    dwell_dict.get("position").get("x"),
                    dwell_dict.get("position").get("y"),
                    dwell_dict.get("position").get("z"),
                ]
            )
            relativePos = dwell_dict.get("relativePos")
            rotation = np.array(
                [
                    dwell_dict.get("rotation").get("x"),
                    dwell_dict.get("rotation").get("y"),
                    dwell_dict.get("rotation").get("z")
                ]
            )
            time = float(dwell_dict.get("time"))
            weight = float(dwell_dict.get("weight", None))

        assert isinstance(index, int), "index should be an integer"
        self.index = index
        assert isinstance(
            angle, float or int
        ), "index should be a floating point number"
        self.angle = angle
        assert isinstance(position, np.ndarray), "position should be a numpy array"
        self.position = position
        assert isinstance(relativePos, int), "relativePos should be an integer"
        self.relativePos = relativePos
        assert isinstance(rotation, np.ndarray), "rotation should be a numpy array"
        self.rotation = rotation
        assert isinstance(time, float), "time should be a float"
        self.time = time
        assert isinstance(weight, float), "weight should be a float"
        self.weight = weight

    def to_dict(self) -> dict:
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
            "weight": float(self.weight),
        }


class Catheter:
    r"""
    Purpose:
        - This class holds the information regarding a catheter.
    Attributes:
        - id:int := the id of the catheter.
        - points:List[np.array] := the list of points of the catheter.
        - dwells:List[DwellPosition] := the list of dwell positions of the catheter.
    """

    def __init__(
        self,
        iD: int = None,
        dwells: list = None,
        points: List[DwellPosition] = None,
        channel_total_time: float = None,
        catheter_dict: dict = None,
    ) -> None:
        r"""
        Purpose:
            - Initialize the Catheter object.
        Inputs:
            - iD:int := the id of the catheter.
            - dwells:List[DwellPosition] := the list of dwell positions of the catheter.
            - points:List[np.array] := the list of points of the catheter.
            - catheter_dict:dict := the dictionary containing the catheter.
        """
        assert (
            iD is not None
            and dwells is not None
            and points is not None
            and channel_total_time is not None
        ) != (
            catheter_dict is not None
        ), "Either provide iD, dwells and points or provide catheter_dict. Not both."
        if catheter_dict is not None:
            iD = catheter_dict.get("id")
            points = catheter_dict.get("points")
            dwells = []
            channel_total_time = catheter_dict.get("channel_total_time", 0.0)
            for i, dwell_dict in enumerate(catheter_dict.get("dwells")):
                if "index" not in dwell_dict:
                    dwell_dict["index"] = i
                dwells.append(DwellPosition(dwell_dict=dwell_dict))
                if "channel_total_time" not in catheter_dict:
                    channel_total_time += dwell_dict.get("time")

        assert isinstance(iD, int), "iD should be an integer"
        self.id = iD
        assert isinstance(points, list), "points should be a list"
        self.points = points
        assert isinstance(dwells, list), "dwells should be a list"
        self.dwells = dwells
        assert isinstance(
            channel_total_time, float
        ), "channel_total_time should be a float"
        self.channel_total_time = channel_total_time

    def to_dict(self) -> dict:
        r"""
        Purpose:
            - To convert the catheter to a dictionary.
        Inputs:
            - self := the Catheter object.
        Outputs:
            - dict := the dictionary containing the catheter.
        """
        return {
            "id": self.id,
            "points": self.points,
            "dwells": [dwell.to_dict() for dwell in self.dwells],
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

