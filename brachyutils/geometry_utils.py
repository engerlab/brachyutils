import os
import warnings
from glob import glob
from pathlib import Path
from typing import Literal, Union, Optional, List, Tuple, Dict

import numpy as np
import pydicom
from opentps.core.data import ROIContour, RTStruct
from opentps.core.data.images import CTImage, MRImage, ROIMask
from opentps.core.io.dicomIO import (  # readDicomPlan, dose not work on brachy; writeRTPlan, dose not work on brachy; writeRTStruct
    readDicomCT,
    readDicomMRI,
    readDicomStruct,
    writeDicomCT,
    writeRTDose,
)


class BrachyGeometry:
    r"""
    Puprose:
        - A class to load any voxelized geometry related to an HDR brachytherapy patient or phantom
        and perform some operations.
    Attributes:
        - id: str := the path of the geometry source file or files.
        - image_obj: CTImage or MRImage := the image of the patient loaded by openTPS. [x, y, z]
        - image_modality: Literal["CT", "MR", "US"] := the modality of the image.
        - structure_set: RTStruct := the structure set of the patient loaded by openTPS. [x, y, z]
        - unit_length: Literal["mm"] := the unit of length in the dicom file. default is mm.
    Dependencies:
        - openTPS.core
    """
    def __init__(
        self,
        pth_image: Path,
        input_file_type: Literal["DICOM", "NRRD"],
        pth_structure: Optional[Path] = None,
    ):
        r"""
        Purpose:
            - Initialize the BrachyGeometry class based on the input path.
        Inputs:
            - pth_image: Path := the path of the geometry source files (if DICOM) or file (if NRRD).
            - input_file_type: Literal["DICOM", "NRRD"] := the type of the input file.
            - pth_structure: Optional[Path] := the path of the structure source file
             (could be a single DICOM or NRRD file).    
        Outputs:
            - None
        Dependencies:
            - openTPS.core
            - BrachyEgsphant
        """
        self.id: Path = pth_image
        self.image_obj: Union[CTImage, MRImage] = None
        self.image_modality: Literal["CT", "MR", "US"] = None
        self.structure_set: RTStruct = None
        self.structure_names_dcm: List[str] = []
        self.unit_length: Literal["mm"] = "mm"
        self.xyz_format:bool = True
        
        assert os.path.exists(pth_image), "The input path does not exist."
        
        if input_file_type == "DICOM":
            self._load_dicom_image_files(pth_image)
        elif input_file_type == "NRRD":
            self._load_nrrd_image_file(pth_image)
        else:
            raise ValueError("The input file type is not supported.")
        
        if pth_structure is not None:
            assert os.path.exists(pth_structure), "The input path does not exist."
            self._load_structure_file(pth_structure)
            
    def _load_dicom_image_files(self, pth_image: Path):
        r"""
        Purpose:
            - Load the DICOM image files.
        Inputs:
            - pth_image: Path := the path of the geometry source files.
        Outputs:
            - None
        Dependencies:
            - openTPS.core
        """
        # Load the image and structure set
        image_files = glob(str(pth_image / "*.dcm"))
        if len(image_files) == 0:
            raise ValueError("No DICOM files found in the input directory.")
        if "CT" in image_files[0]:
            self.image_obj = readDicomCT(image_files)
            self.image_modality = "CT"
        elif "MR" in image_files[0]:
            self.image_obj = readDicomMRI(image_files)
            self.image_modality = "MR"
        elif "US" in image_files[0]:
            self.image_obj = readDicomUS(image_files)
            self.image_modality = "US"

    def _load_nrrd_image_file(self, pth_image: Path):
        r"""
        Purpose:
            - Load the NRRD image file.
        Inputs:
            - pth_image: Path := the path of the geometry source file.
        Outputs:
            - None
        Dependencies:
            - openTPS.core
        """
        raise NotImplementedError("NRRD files are not supported yet.")
    
    def _load_structure_file(self, pth_structure: Path):
        r"""
        Purpose:
            - Load the structure file.
        Inputs:
            - pth_structure: Path := the path of the structure source file.
        Outputs:
            - None
        Dependencies:
            - openTPS.core
        """
        structure_file_type = os.path.splitext(pth_structure)[-1]
        if structure_file_type == ".dcm":
            self.structure_set = readDicomStruct(pth_structure)
        elif structure_file_type == ".nrrd":
            raise NotImplementedError("NRRD files are not supported for structures yet.")
        else:
            raise ValueError("The structure file type is currently not supported.")
        for structure in self.structure_set.structures:
            self.structure_names_dcm = []
            self.structure_names_dcm.append(structure.name)

    def get_strcuture_mask_from_dicom(
        self,
        query_structure_list: List[str],
        mask_type: Union[np.ndarray, ROIContour, ROIMask] = ROIMask,
    ) -> dict:
        r"""
        Purpose:
            To return a dictionary with the requested structure masks from BrachyDicom object. The queried
            structure string should be a subset of the structure string in the dicom file. For example,
            if the structure string in dicom file is CTV_BRACHY, then the query string can be CTV or ctv.
            The keys in the dictionary match the query_structure_list and the values are the masks.
        Inputs:
            - query_structure_list := list of structure names to find the mask of.
            - mask_type: Union[np.ndarray, ROIContour, ROIMask] := the type of the mask to return.
             if np.ndarray, the mask will be returned as a numpy array in [z, y, x] format.
             if ROIContour, the mask will be returned as a ROIContour object in [x, y, z] format.
             if ROIMask, the mask will be returned as a ROIMask object in [x, y, z] format.
        Outputs:
            - mask_dict:dict :=  a dictionary with the queried structure name as key and the mask as value.
        """
        assert (
            self.structure_mask_dict is not None
        ), "structure masks have not been loaded yet. please run load_structures() first"
        mask_dict: dict = {}
        for query_structure in query_structure_list:
            for mask_name, mask in self.structure_mask_dict.items():
                if query_structure.lower() in mask_name.lower():
                    if np.any(mask.imageArray):
                        if mask_type == np.ndarray:
                            mask_dict[query_structure] = np.swapaxes(mask.imageArray, 0, 2)
                        elif mask_type == ROIContour:
                            mask_dict[query_structure] = (
                                self.structures_dcm.getContourByName(mask_name)
                            )
                        elif mask_type == ROIMask:
                            mask_dict[query_structure] = mask
                        else:
                            raise ValueError("mask_type not recognized")
                    else:
                        mask_dict[query_structure] = None
                        warnings.warn(
                            f"mask for {query_structure} is all zeros. returning empty",
                            stacklevel=2,
                        )
        return mask_dict

    def info(self):
        r"""
        Purpose:
            - Print the information of the BrachyGeometry object.
        Inputs:
            - None
        Outputs:
            - None
        """
        print(f"Geometry ID: {self.id}")
        print(f"Image Modality: {self.image_modality}")
        print(f"Unit Length: {self.unit_length}")
        print(f"Image Shape [x, y, z]: {self.image_obj.gridSize}")
        print(f"Image size in world unit [x, y, z]: {self.image_obj.gridSizeInWorldUnit}")
        print(f"Image Origin [x, y, z]: {self.image_obj.origin}")
        print(f"Image Spacing [x, y, z]: {self.image_obj.spacing}")
        print(f"Structure Names: {self.structure_names_dcm}")
        print(f"Structure Count: {len(self.structure_names_dcm)}")

    def reset(self):
        r"""
        Purpose:
            - Reset the BrachyGeometry object.
        Inputs:
            - None
        Outputs:
            - None
        """
        self.id = None
        self.image_obj = None
        self.image_modality = None
        self.structure_set = None
        self.unit_length = None
        self.structure_names_dcm = []

    def is_equal(self, other: "BrachyGeometry") -> bool:
        r"""
        Purpose:
            - Check if two BrachyGeometry objects have equal image_obj.
        Inputs:
            - other: BrachyGeometry := the other BrachyGeometry object.
        Outputs:
            - bool := True if the two objects are equal, False otherwise.
        """
        if not isinstance(other, BrachyGeometry):
            warnings.warn("The input object is not a BrachyGeometry object.")
            return False
        if not self.image_modality == other.image_modality:
            warnings.warn("The image modalities are not the same.")
            return False
        if not self.unit_length == other.unit_length:
            warnings.warn("The unit lengths are not the same.")
            return False
        if not np.array_equal(self.image_obj.imageArray, other.image_obj.imageArray):
            warnings.warn("The image arrays are not the same.")
            return False
        for structure_name in self.structure_names_dcm:
            if self.structure_set.getContourByName(structure_name) != other.structure_set.getContourByName(structure_name):
                warnings.warn(f"The structure masks for {structure_name} are not the same.")
                return False
        

# helper functions
def readDicomUS(image_files):
    r"""
    Purpose:
        - Read the US DICOM files.
    Inputs:
        - image_files: List[Path] := the list of the US DICOM image files.
    Outputs:
        - USImage := the US image object.
    Dependencies:
        - openTPS.core
    """
    raise NotImplementedError("US DICOM files are not supported yet.")