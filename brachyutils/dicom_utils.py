import os

# from dicompylercore import dicomparser
from glob import glob

import numpy as np
import pydicom
from DicomRTTool.ReaderWriter import DicomReaderWriter

from brachyutils.dose_utils import BrachyDose


class BrachyDicom:
    r"""
    Puprose:
        - A wrapper around the DicomReaderWriter class to get the images, the structure masks
        and the index range of the structure masks.
    Attributes:
        - dicom_reader:DicomReaderWriter := an instance of the DicomReaderWriter class.
        - all_rois:list := a list of all the structure names in the dicom file.
        - image:np.array := the image of the patient. [z, y, x]
        - origin_coordinates:list := the origin of the image. [x, y, z]
        - structure_mask_dict:dict := a dictionary with the structure name as key and the mask as value.
        - structure_index_range_dict:dict := a dictionary with the structure name as key and the index range as value.
        - dose: BrachyDose := dose from dicom RD file saved as an instance of the BrachyDose class.
        - catheter_table := a dictionary returned by get_catheter_table_and_source_info_from_dicom()
        - source_info: dict := a dictionary with the source information.
    Dependencies:
        - DicomRTTool: https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485

    """

    def __init__(
        self,
        pth_dir_dicom: str,
        load_image: bool = True,
        load_structure: bool = True,
        load_dose: bool = False,
        load_plan: bool = False,
    ):
        r"""
        Purpose:
            - To gatheter all the information provided by the dicom files of a patient.
        """
        self.dicom_reader: DicomReaderWriter = None
        self.all_rois = None
        self.grid: np.array = None
        self.origin_coordinates: np.array = None
        self.voxel_size: np.array = None
        self.num_voxels: np.array = None
        self.structure_mask_dict: dict = {}
        self.structure_index_range_dict: dict = {}
        self.dose: BrachyDose = None
        self.catheter_table: dict = None
        self.source_info: dict = None

        os.path.abspath(pth_dir_dicom)
        assert os.path.exists(pth_dir_dicom), "given dicom path does not exist"
        assert glob(
            pth_dir_dicom + "/*.dcm"
        ), "there are no dicom files in this directory"
        # load the structure file into an rt_struct object
        self.dicom_reader = DicomReaderWriter(
            description="getting structure masks", arg_max=True
        )
        self.dicom_reader.walk_through_folders(pth_dir_dicom)
        self.dicom_reader.get_images()

        if load_image:
            self.grid = self.dicom_reader.ArrayDicom
            self.origin_coordinates = np.array(
                self.dicom_reader.dicom_handle.GetOrigin(), dtype=np.float32
            )
            self.voxel_size = np.array(
                self.dicom_reader.dicom_handle.GetSpacing(), dtype=np.float32
            )
            self.num_voxels = np.array(
                [
                    int(self.dicom_reader.return_key_info("0028|0010")),
                    int(self.dicom_reader.return_key_info("0028|0011")),
                    int(len(self.dicom_reader.series_instances_dictionary[0].files)),
                ]
            )

        if load_structure:
            self.all_rois = self.dicom_reader.return_rois()
            # self.get_strcuture_mask_from_dicom(self.all_rois)
            self.get_structure_index_range(self.all_rois)

        if load_dose:
            self.dose = BrachyDose(glob(pth_dir_dicom + "/RD*.dcm")[0])

        if load_plan:
            self.catheter_table, self.source_info = (
                get_catheter_table_and_source_info_from_dicom(
                    glob(pth_dir_dicom + "/RP*.dcm")[0]
                )
            )

    def get_structure_index_range(self, query_structure_list: list):
        r"""
        Purpose:
            To find the index extent of the structure voxels along each axis using dicom RT structure file.
            If the object already has this feature, it will return the stored value instead of over-writing it.
        Inputs:
            - query_structure_list := list of structure names to find the index range of.
        Outputs:
            - structure_index_range:np.array :=  a 3 x 2 array holding the min and max on x, y and axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]],
            - body_mask_shape:np.array := 1 x 3 array holding the dimension of the original mask
        Dependencies:
            - get_strcuture_mask_from_dicom()
        """
        if len(self.structure_index_range_dict) > 0:
            return self.structure_index_range_dict

        self.structure_index_range_dict = {}
        self.get_strcuture_mask_from_dicom(query_structure_list)
        for mask_name, mask_numpy in self.structure_mask_dict.items():
            # so we got the mask but the dimensions may not match the dimension of the dose
            # let's get the relative extent of the body mask compared to the whole grid and resample
            # the extents

            # skip the mask if it is empty
            if np.sum(mask_numpy) == 0:
                continue
            structure_index_range = np.zeros([3, 2], dtype=int)
            for i in range(3):
                structure_index_range[i, :] = np.floor(
                    np.array(
                        [
                            np.argwhere(mask_numpy == 1)[:, i].min(),
                            # off set of +1 is added to acount for python stopping before range end
                            np.argwhere(mask_numpy == 1)[:, i].max() + 1,
                        ]
                    )
                ).astype(int)
            structure_index_range = np.flip(structure_index_range, axis=0)
            self.structure_index_range_dict[mask_name] = {
                "structure_index_range": structure_index_range,
                "dicom_mask_shape": np.flip(np.array(mask_numpy.shape)),
            }
        return self.structure_index_range_dict

    def get_strcuture_mask_from_dicom(self, query_structure_list: list):
        r"""
        Purpose:
            To get the mask of the structures using dicom RT structure file.
            If the object already has this feature, it will return the stored value instead of over-writing it.
        Inputs:
            - query_structure_list := list of structure names to find the mask of.
        Outputs:
            - mask_dict:dict :=  a dictionary with the structure name as key and the mask as value.
        """
        if len(self.structure_mask_dict) > 0:
            return self.structure_mask_dict

        for query_structure_name in query_structure_list:
            # # find the name of the body structure inside the rt_structure object
            dicom_structure_name = [
                name for name in self.all_rois if query_structure_name in name.lower()
            ]
            # # get the numpy array of the body structure:
            assert len(dicom_structure_name) >= 1, "no contour was found!"
            self.dicom_reader.set_contour_names_and_associations(
                contour_names=dicom_structure_name
            )
            self.dicom_reader.get_mask()
            mask_numpy = self.dicom_reader.mask
            self.structure_mask_dict[query_structure_name] = mask_numpy

        return self.structure_mask_dict

    def reset(self):
        self.structure_mask_dict = {}
        self.structure_index_range_dict = {}

    def info(self):
        print(f"shape of the image: {self.grid.shape}")
        print(f"origin of the image: {self.origin_coordinates}")
        print(f"voxel size of the image: {self.voxel_size}")
        print(f"number of voxels: {self.num_voxels}")
        print(f"all the structures in the dicom: {self.all_rois}")
        if self.dose is not None:
            print(f"the shape of dose: {self.dose.num_voxels}")
            print(f"origin of the dose: {self.dose.origin_coordinates}")
            print(f"voxel size of the dose: {self.dose.voxel_size}")
        else:
            print("no dose file was loaded")
        if self.catheter_table is not None:
            num_dwell_positions = np.sum(
                [len(catheter["dwells"]) for catheter in self.catheter_table]
            )
            print(f"number of dwell positions: {num_dwell_positions}")
            treatment_time = np.sum(
                [catheter["channel_total_time"] for catheter in self.catheter_table]
            )
            print(f"treatment time: {treatment_time}")
            print(f"source info: {self.source_info}")
        else:
            print("no plan file was loaded")


def get_catheter_table_and_source_info_from_dicom(pth_dicom_plan: str):
    r"""
    Purpose:
        - To load the catheter table from the dicom plan file.
    Inputs:
        - pth_dicom_plan:str := the path to the dicom plan file.
    Outputs:
        - catheter_table:dict := a dictionary with the catheter id as key and the dwell points as value.
        the details of the keys in this dictionary are"
            - id:int := the id of the catheter.
            - points:list := a list of the dwell points.
                - index:int := the index of the dwell point.
                - angle:float := the angle of the dwell point.
                - position:np.array := the position of the dwell point.
                - relativePos:float := the relative position of the dwell point.
                - rotation:np.array := the rotation of the dwell point.
                - time:float := the time of the dwell point.
                - weight:float := the weight of the dwell point.

        - source_info:dict := a dictionary with the source information.
    Dependencies:
        - pydicom: https://pydicom.github.io/
    """
    # load the plan file into an rt_plan object
    plan = pydicom.dcmread(pth_dicom_plan)
    catheter_table = []
    # get the source info
    source_info = {
        "TotalReferenceAirKerma": (
            float(plan.ApplicationSetupSequence[0].TotalReferenceAirKerma)
            if hasattr(plan.ApplicationSetupSequence[0], "TotalReferenceAirKerma")
            else None
        ),
        # "BrachyTreatmentType": (
        #     plan.ApplicationSetupSequence[0].BrachyTreatmentType
        #     if hasattr(plan.ApplicationSetupSequence[0], "BrachyTreatmentType")
        #     else None
        # ),
        # "SourceType": None,
        # "SourceManufacturer"
        # ActiveSourceDiameter
        # ActiveSourceLength
        # SourceEncapsulationNominalThickness and many more is possible...
    }
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
                            control_point_dcm.ControlPoint3DPosition, dtype=np.float32
                        )
                        if hasattr(control_point_dcm, "ControlPoint3DPosition")
                        else None
                    ),
                    "relativePos": (
                        float(control_point_dcm.ControlPointRelativePosition)
                        if hasattr(control_point_dcm, "ControlPointRelativePosition")
                        else None
                    ),
                    "rotation": (
                        np.array(
                            control_point_dcm.ControlPointOrientation, dtype=np.float32
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
                    "index": control_point["index"] / 2,
                    "angle": control_point["angle"],
                    "position": control_point["position"],
                    "relativePos": control_point["relativePos"],
                    "rotation": control_point["rotation"],
                    "time": dwell_time,
                    "weight": dwell_weight,
                }
            )
        final_catheter_table.append(
            {
                "id": catheter["id"],
                "points": catheter["points"],
                "channel_total_time": catheter["channel_total_time"],
                "dwells": dwells,
            }
        )

    return final_catheter_table, source_info
