import os

# from dicompylercore import dicomparser
from glob import glob

import numpy as np
import pydicom
from DicomRTTool.ReaderWriter import DicomReaderWriter
from dose_utils import BrachyDose


class BrachyDicom:
    r"""
    Puprose:
        - A wrapper around the DicomReaderWriter class to get the images, the structure masks
        and the index range of the structure masks.
    Attributes:
        - dicom_reader:DicomReaderWriter := an instance of the DicomReaderWriter class.
        - all_rois:list := a list of all the structure names in the dicom file.
        - mask_dict:dict := a dictionary with the structure name as key and the mask as value.
        - structure_index_range_dict:dict := a dictionary with the structure name as key and the index range as value.
        - top_left:DicomReaderWriter := an instance of the DicomReaderWriter class.
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
        self.all_rois = self.dicom_reader.return_rois()

        if load_image:
            self.image = self.dicom_reader.ArrayDicom
            self.origin_coords = np.array(
                self.dicom_reader.dicom_handle.GetOrigin(), dtype=np.float32
            )
            self.voxel_size = np.array(
                self.dicom_reader.dicom_handle.GetSpacing(), dtype=np.float32
            )

        self.mask_dict = {}
        self.structure_index_range_dict = {}
        if load_structure:
            self.get_strcuture_mask_from_dicom(self.all_rois)
            self.get_structure_index_range(self.all_rois)

        if load_dose:
            self.dose = BrachyDose(glob(pth_dir_dicom + "/RD*.dcm")[0])
        if load_plan:
            self.catheter_table = load_catheter_table(
                glob(pth_dir_dicom + "/RP*.dcm")[0]
            )

    def get_structure_index_range(self, query_structure_list: list):
        r"""
        Purpose:
            to find the index extent of the structure voxels along each axis using dicom RT structure file.
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
        for mask_name, mask_numpy in self.mask_dict.items():
            # so we got the mask but the dimensions may not match the dimension of the dose
            # let's get the relative extent of the body mask compared to the whole grid and resample
            # the extents
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
            to get the mask of the structures using dicom RT structure file.
        Inputs:
            - query_structure_list := list of structure names to find the mask of.
        Outputs:
            - mask_dict:dict :=  a dictionary with the structure name as key and the mask as value.
        """
        if len(self.mask_dict) > 0:
            return self.mask_dict

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
            self.mask_dict[query_structure_name] = mask_numpy

        return self.mask_dict

    def reset(self):
        self.mask_dict = {}
        self.structure_index_range_dict = {}

    def info(self):
        print(f"shape of the image: {self.image.shape}")
        print(f"origin of the image: {self.origin_coords}")
        print(f"voxel size of the image: {self.voxel_size}")
        print(f"all the structures in the dicom: {self.all_rois}")
        print(f"the shape of dose: {self.dose.num_voxels}")
        print(f"origin of the dose: {self.dose.topleft}")
        print(f"voxel size of the dose: {self.dose.voxel_size}")


def load_catheter_table(pth_catheter_table: str):
    # load the plan file into an rt_plan object
    plan = pydicom.dcmread(pth_catheter_table)
    catheter_table = []
    total_reference_air_kerma = float(plan.ApplicationSetupSequence[0].TotalReferenceAirKerma)
    for catheter_dcm in plan.ApplicationSetupSequence[0].ChannelSequence:
        dwells = []
        catheter_time = float(catheter_dcm.ChannelTotalTime)
        for dwell_dcm in catheter_dcm.BrachyControlPointSequence:
            dwell_time_weight = float(dwell_dcm.CumulativeTimeWeight)
            dwells.append(
                {
                    "index": int(dwell_dcm.ControlPointIndex),
                    "angle": None,
                    "position": np.array(dwell_dcm.ControlPoint3DPosition, dtype=np.float32),
                    "relativePos": catheter_time*dwell_time_weight,
                    "rotation": None,
                    "time": float(dwell_dcm.CumulativeTimeWeight),
                    "weight": dwell_time_weight,
                    "total rerence air kerma": total_reference_air_kerma
                }
            )
        catheter_table.append(
            {
                "id": catheter_dcm.ChannelNumber,
                "points": [],
                "channel total time": catheter_time,
                "dwells": dwells,
            }
        )
    return catheter_table
