import os
from typing import Union, Literal, List, Optional
from pathlib import Path
from glob import glob
import warnings

import numpy as np

import pydicom
# from DicomRTTool.ReaderWriter import DicomReaderWriter

from brachyutils.dose_utils import BrachyDose

from opentps.core.data.images import (
    CTImage,
    MRImage,
    ROIMask,
    DoseImage
)

from opentps.core.data import RTStruct

from opentps.core.io.dicomIO import (
    readDicomCT,
    readDicomMRI,
    # readDicomPlan, dose not work on brachy
    readDicomStruct,
    writeDicomCT,
    writeRTDose,
    # writeRTPlan, dose not work on brachy
    # writeRTStruct
)

class BrachyDicom:
    r"""
    Puprose:
        - A wrapper around the DicomReaderWriter class to get the images, the structure masks
        and the index range of the structure masks.
    Attributes:
        - image: CTImage or MRImage := the image of the patient loaded by openTPS.
        - image_modality: Literal["CT", "MR"] := the modality of the image.
        - structures: RTStruct := the structure masks of the patient loaded by openTPS.
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
        load_image: Optional[bool] = True,
        load_structure: Optional[bool] = True,
        load_dose: Optional[bool] = False,
        load_plan: Optional[bool] = False,
    ):
        r"""
        Purpose:
            - To gatheter all the information provided by the dicom files of a patient. You can choose to load the image,
            the structure masks, the dose and the plan. by default, the image and the structure masks are loaded.
        """
        self.image: Union[CTImage, MRImage] = None
        self.image_modality: Literal["CT", "MR"] = None
        self.structures: RTStruct = None
        self.structure_mask_dict: dict = {}
        self.structure_index_range_dict: dict = {}
        self.dose: BrachyDose = None
        self.catheter_table: dict = None
        self.source_info: dict = None

        os.path.abspath(pth_dir_dicom)
        assert os.path.exists(pth_dir_dicom), "given dicom path does not exist"
        
        file_list: list = glob(pth_dir_dicom + "/*.dcm")
        assert len(file_list) > 0, "there are no dicom files in this directory"

        if load_image:
            if "CT" in file_list[0]:
                self.image_modality = "CT"
                ct_files = list(filter(lambda s: "CT" in s, file_list))
                # in python, images are represented as [z, y, x] but in dicom it's [x, y, z]
                image_xyz = readDicomCT(ct_files)
                self.image = CTImage(
                    imageArray=np.swapaxes(image_xyz.imageArray, 0, 2),
                    origin=np.flip(image_xyz.origin),
                    spacing=np.flip(image_xyz.spacing),
                    angles=np.flip(image_xyz.angles),
                    name=image_xyz.name,
                    seriesInstanceUID=image_xyz.seriesInstanceUID,
                    frameOfReferenceUID=image_xyz.frameOfReferenceUID,
                )
            elif "MR" in file_list[0]:
                self.image_modality = "MR"
                mr_files = list(filter(lambda s: "MR" in s, file_list))
                # in python, images are represented as [z, y, x] but in dicom it's [x, y, z]
                image_xyz = readDicomMRI(mr_files)
                self.image = MRImage(
                    imageArray=np.swapaxes(image_xyz.imageArray, 0, 2),
                    origin=np.flip(image_xyz.origin),
                    spacing=np.flip(image_xyz.spacing),
                    angles=np.flip(image_xyz.angles),
                    name=image_xyz.name,
                    seriesInstanceUID=image_xyz.seriesInstanceUID,
                    frameOfReferenceUID=image_xyz.frameOfReferenceUID,
                )
            else:
                raise ValueError("Image modality not recognized")

        if load_structure:
            structure_file = list(filter(lambda s: "RS" in s, file_list)).pop()
            self.load_structures(structure_file)
            
            # self.all_rois = self.dicom_reader.return_rois()
            # # self.get_strcuture_mask_from_dicom(self.all_rois)
            # self.get_structure_index_range(self.all_rois)

        if load_dose:
            dose_file = list(filter(lambda s: "RD" in s, file_list)).pop() 
            self.dose = BrachyDose(dose_file)
            
        if load_plan:
            plan_file = list(filter(lambda s: "RP" in s, file_list)).pop()
            self.catheter_table, self.source_info = (
                get_catheter_table_and_source_info_from_dicom(plan_file)
            )
    def load_structures(self, structure_file: str):
        r"""
        Purpose:
            To load the structures from the dicom RT structure file. 
            The structure masks are stored in the structure_mask_dict.
            each structure would have a binary mask with the same dimension as the image.
        Inputs:
            - structure_file:str := the path to the dicom RT structure file.
        Outputs:
            - void: self.structure_mask_dict will be updated.
        """
        assert os.path.exists(structure_file), "given structure file does not exist"
        assert self.image is not None, "image has not been loaded yet"
        
        self.structures_dcm = readDicomStruct(structure_file)
        for contour in self.structures_dcm.contours:
            self.structure_mask_dict[contour.name] = contour.getBinaryMask(
                origin=self.image.origin,
                gridSize=self.image.gridSize,
                spacing=self.image.spacing,
                ).imageArray

    def get_structure_index_range(self):
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
        
        assert self.structure_mask_dict is not None, "structure masks have not been loaded yet. please run load_structures() first"

        self.structure_index_range_dict = {}
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

    def get_strcuture_mask_from_dicom(
        self,
        query_structure_list: List[str]
        ):
        r"""
        Purpose:
            To return a dictionary with the requested structure masks from BrachyDicom object. The queried
            structure string should be a subset of the structure string in the dicom file. For example, 
            if the structure string in dicom file is CTV_BRACHY, then the query string can be CTV or ctv.
            
        Inputs:
            - query_structure_list := list of structure names to find the mask of.
        Outputs:
            - mask_dict:dict :=  a dictionary with the queried structure name as key and the mask as value.
        """
        assert self.structure_mask_dict is not None, "structure masks have not been loaded yet. please run load_structures() first"
        mask_dict:dict = {}
        for query_structure in query_structure_list:
            for mask_name, mask_numpy in self.structure_mask_dict.items():
                if query_structure.lower() in mask_name.lower():
                    if np.sum(mask_numpy) > 0:
                        mask_dict[query_structure] = mask_numpy
                    else:
                        mask_dict[query_structure] = []
                        warnings.warn(f"mask for {query_structure} is all zeros. returning empty")                        
        return mask_dict

    def reset(self):
        self.structure_mask_dict = {}
        self.structure_index_range_dict = {}

    def info(self):
        print(f"shape of the image: {self.image.gridSize}")
        print(f"origin of the image: {self.image.origin}")
        print(f"voxel size of the image: {self.image.spacing}")
        print(f"all the structures in the dicom: {self.structure_mask_dict.keys()}")
        if self.dose is not None:
            print(f"the shape of dose: {self.dose.dose_image.gridSize}")
            print(f"origin of the dose: {self.dose.dose_image.origin}")
            print(f"voxel size of the dose: {self.dose.dose_image.spacing}")
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
    
    def get_materials_dict(self):
        r"""
        Purpose:
            - To get the materials dictionary from the dicom file.
            The material table contains the following attributes for each structure
        """
        raise NotImplementedError("this function is not implemented yet")

    def write_to_dicom(self, dir_output: Path):
        r"""
        Purpose:
            - To write the image and the dose to a dicom file.
        """
        if self.image is not None:
            if self.image_modality == "CT":
                writeDicomCT(self.image, dir_output)
            elif self.image_modality == "MR":
                raise NotImplementedError("MR image writing is not implemented yet")
            else:
                raise ValueError("Image modality not recognized")

        if self.dose is not None:
            writeRTDose(self.dose, os.path.join(dir_output, "RD.dcm"))

        if self.structures is not None:
            raise NotImplementedError("writing structures to dicom is not implemented yet")

    def write_to_nrrd(self, dir_output: Path):
        r"""
        Purpose:
            - To write the image, the structure masks, the dose and the plan to a nrrd file.
        """
        raise NotImplementedError("writing to nrrd is not implemented yet")

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
