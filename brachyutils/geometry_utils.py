import os
import warnings
from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
# import pydicom
from opentps.core.data import ROIContour, RTStruct
from opentps.core.data.images import CTImage, MRImage, ROIMask
from opentps.core.io.dicomIO import (
    readDicomCT,
    readDicomMRI,
    readDicomStruct,
    writeDicomCT,
    writeRTStruct
    # writeRTDose,
)
import SimpleITK as sitk
# import slicerio

class BrachyPhantom:
    r"""
    Puprose:
        - A class to load any voxelized geometry related to an HDR brachytherapy patient or phantom
        and perform some operations.
    Attributes:
        - id: str := the path of the geometry source file or files.
        - image_obj: CTImage or MRImage := the image of the patient loaded by openTPS. [x, y, z]
        - image_modality: Literal["CT", "MR", "US"] := the modality of the image.
        - structure_set: RTStruct := the structure set of the patient loaded by openTPS. [x, y, z].
        Other names for structure are contours, masks, segmentations.
        - unit_length: Literal["mm"] := the unit of length in the dicom file. default is mm.
    Dependencies:
        - openTPS.core
    """

    def __init__(
        self,
        dir_dicom: Optional[Path] = None,
        pth_phantom_file: Optional[Path] = None,
        pth_structures_file: Optional[Path] = None,
        pth_egsphant_file: Optional[Path] = None,
    ) -> None:
        r"""
        Purpose:
            - Initialize the BrachyPhantom class based on the input path. The input path can be either
            the directory of the DICOM files or the path of the phantom file (in .nrrd). The structures file
            is optional. It is also possible to load the structures only without a phantom file. in that case,
            an empty image_obj is created with the dimensions matching the structures file.
        Inputs:
            - dir_dicom: Path := the directory of the DICOM files.
            - pth_phantom_file: Path := the path of the phantom file.
            - pth_structures_file: Path := the path of the structure file.
        Outputs:
            - None
        Dependencies:
            - openTPS.core
            - BrachyEgsphant
        """
        if dir_dicom is not None and pth_phantom_file is not None:
            raise ValueError(
                "Please provide either the directory of the DICOM files or the path of the phantom file."
            )
        # Attributes for patient images
        self.pth_image: Path = dir_dicom if dir_dicom is not None else pth_phantom_file
        self.image_obj: Union[CTImage, MRImage] = None
        self.image_modality: Literal["CT", "MR", "US"] = None
        self.structure_set: RTStruct = None
        self.structure_names_dcm: List[str] = []
        self.unit_length: Literal["mm"] = "mm"
        self.xyz_format: bool = True

        # Attributes for Egsphant files
        self.egsphant_obj:BrachyEgsphant = None

        if dir_dicom is not None:
            self._load_dicom_image_files(self.pth_image)
        elif pth_phantom_file is not None:
            self._load_nrrd_image_file(self.pth_image)
        else:
            raise ValueError("No geometry source file provided. Please provide either the directory of the DICOM files or the path of the phantom file.")
            warnings.warn("No geometry source file provided.", stacklevel=2)

        if pth_structures_file is not None:
            assert os.path.exists(pth_structures_file), "The input path does not exist."
            self._load_structure_file(pth_structures_file)

        if pth_egsphant_file is not None:
            self.egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_egsphant_file)

    def _load_dicom_image_files(self, pth_image: Path) -> None:
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
        assert os.path.exists(pth_image), "The input path does not exist."
        # Load the image and structure set
        image_files = glob((pth_image+"/*.dcm"))
        if len(image_files) == 0:
            raise ValueError("No DICOM files found in the input directory.")
        if "CT" in image_files[0].upper():
            ct_files = list(filter(lambda s: "CT" in s.upper(), image_files))
            self.image_obj = readDicomCT(ct_files)
            self.image_modality = "CT"
        elif "MR" in image_files[0].upper():
            mr_files = list(filter(lambda s: "MR" in s.upper(), image_files))
            self.image_obj = readDicomMRI(mr_files)
            self.image_modality = "MR"
        elif "US" in image_files[0].upper():
            us_files = list(filter(lambda s: "US" in s.upper(), image_files))
            self.image_obj = readDicomUS(us_files)
            self.image_modality = "US"

    def _load_nrrd_image_file(self, pth_image: Path) -> None:
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
        assert os.path.exists(pth_image), "The input path does not exist."
        image_nrrd = sitk.ReadImage(pth_image, imageIO="NrrdImageIO")
        self.pth_image = pth_image
        self.image_obj = CTImage(
            imageArray=np.swapaxes(sitk.GetArrayFromImage(image_nrrd), 0, 2),
            origin=np.array(image_nrrd.GetOrigin()),
            spacing=np.array(image_nrrd.GetSpacing()),
        )
        self.image_modality = image_nrrd.GetMetaData("Modality")


    def _load_structure_file(self, pth_structure: Path) -> None:
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
            self.structure_set = readNrrdStruct(pth_structure)
            self.structure_set.setPatient(self.image_obj.patient if self.image_obj is not None else None)
            # self.structure_set.seriesInstanceUID = self.image_obj.seriesInstanceUID if self.structure_set is not None else ""
            # self.structure_set.sopInstanceUID = self.image_obj.sopInstanceUID if self.structure_set is None else ""
        else:
            raise ValueError("The structure file type is currently not supported.")
        self.structure_names_dcm = []
        for structure in self.structure_set.contours:
            self.structure_names_dcm.append(structure.name)

    def get_structure_mask(
        self,
        query_structure_list: List[str],
        mask_type: Union[np.ndarray, ROIContour, ROIMask] = ROIMask,
    ) -> Dict[str, Union[np.ndarray, ROIContour, ROIMask]]:
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
            self.structure_set is not None
        ), "structure masks have not been loaded yet. please run load_structures() first"
        mask_dict: dict = {}
        for query_structure in query_structure_list:
            for mask_name in self.structure_names_dcm:
                if query_structure.lower() in mask_name.lower():
                    mask = self.structure_set.getContourByName(mask_name).getBinaryMask(
                        origin=self.image_obj.origin,
                        gridSize=self.image_obj.gridSize,
                        spacing=self.image_obj.spacing,
                    )
                    if np.any(mask.imageArray):
                        if mask_type == np.ndarray:
                            mask_dict[query_structure] = np.swapaxes(
                                mask.imageArray, 0, 2
                            )
                        elif mask_type == ROIContour:
                            mask_dict[query_structure] = (
                                self.structure_set.getContourByName(mask_name)
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

    def info(self) -> None:
        r"""
        Purpose:
            - Print the information of the BrachyPhantom object.
        Inputs:
            - None
        Outputs:
            - None
        """
        print(f"Geometry File source: {self.pth_image}" if self.pth_image is not None else "No geometry file source.")
        print(f"Image Modality: {self.image_modality}" if self.image_modality is not None else "No image modality.")
        print(f"Unit Length: {self.unit_length}" if self.unit_length is not None else "No unit length.")
        print(f"Image Shape [x, y, z]: {self.image_obj.gridSize}" if self.image_obj is not None else "No image object.")
        print(
            f"Image size in world unit [x, y, z]: {self.image_obj.gridSizeInWorldUnit}" if self.image_obj is not None else "No image object."
        )
        print(f"Image Origin [x, y, z]: {self.image_obj.origin}" if self.image_obj is not None else "No image object.")
        print(f"Image Spacing [x, y, z]: {self.image_obj.spacing}" if self.image_obj is not None else "No image object.")
        print(f"Structure Names: {self.structure_names_dcm}" if self.structure_names_dcm is not None else "No structure names.")
        print(f"Structure Count: {len(self.structure_names_dcm)}" if self.structure_names_dcm is not None else "No structure names.")

    def reset(self):
        r"""
        Purpose:
            - Reset the BrachyPhantom object.
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

    def is_equal(self, other: "BrachyPhantom") -> bool:
        r"""
        Purpose:
            - Check if two BrachyPhantom objects have equal image_obj.
        Inputs:
            - other: BrachyPhantom := the other BrachyPhantom object.
        Outputs:
            - bool := True if the two objects are equal, False otherwise.
        """
        if not isinstance(other, BrachyPhantom):
            warnings.warn(
                "The input object is not a BrachyPhantom object.", stacklevel=2
            )
            return False
        elif not self.image_modality == other.image_modality:
            warnings.warn("The image modalities are not the same.", stacklevel=2)
            return False
        elif not self.unit_length == other.unit_length:
            warnings.warn("The unit lengths are not the same.", stacklevel=2)
            return False
        elif not np.array_equal(self.image_obj.imageArray, other.image_obj.imageArray):
            warnings.warn("The image arrays are not the same.", stacklevel=2)
            return False
        elif self.structure_set is not None and other.structure_set is not None:
            for structure_name in self.structure_names_dcm:
                if self.structure_set.getContourByName(
                    structure_name
                ) != other.structure_set.getContourByName(structure_name):
                    warnings.warn(
                        f"The structure masks for {structure_name} are not the same.",
                        stacklevel=2,
                    )
                    return False
        else:
            return True
    def get_image_array(self) -> np.ndarray:
        r"""
        Purpose:
            - To return the image as a numpy array in z y x format.
        """
        return np.swapaxes(self.image_obj.imageArray, 0, 2)

    def write_image_to_dicom(self, dir_output: Path) -> None:
        r"""
        Purpose:
            - To write the image and the dose to a dicom file.
        """
        if self.image_obj is not None:
            os.makedirs(dir_output, exist_ok=True)
            if self.image_modality == "CT":
                writeDicomCT(self.image_obj, dir_output)
            elif self.image_modality == "MR":
                raise NotImplementedError("MR image writing is not implemented yet")
            elif self.image_modality == "US":
                raise NotImplementedError("US image writing is not implemented yet")
            else:
                raise ValueError("Image modality not recognized")

    def write_structures_to_dicom(self, dir_output: Path) -> None:
        r"""
        Purpose:
            - To write the structures to a dicom file.
        """
        if self.structure_set is not None:
            os.makedirs(dir_output, exist_ok=True)
            writeRTStruct(self.structure_set, dir_output)

    def write_image_to_nrrd(self, pth_output: Path) -> None:
        r"""
        Purpose:
            - To write the image to a nrrd file.
        """
        assert os.path.splitext(pth_output)[-1] == ".nrrd", "the file should have '.nrrd' extension"
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        image_array_zyx = self.get_image_array()
        image_nrrd = sitk.GetImageFromArray(image_array_zyx.astype(float))
        image_nrrd.SetSpacing(self.image_obj.spacing.astype(float))
        image_nrrd.SetOrigin(self.image_obj.origin.astype(float))
        image_nrrd.SetMetaData("Modality", self.image_modality)
        sitk.WriteImage(image_nrrd, str(pth_output))
    
    def write_structures_to_nrrd(
        self,
        pth_output: Path,
        no_overlap:Optional[bool] = True,
        ) -> None:
        r"""
        Purpose:
            - To write the structures to a nrrd file. By defualt, we remove the overlap between the structures. the smaller structures
            overwrite the larger structures if there is an overlap.
        Inputs:
            - pth_output: Path := the path to write the structures to.
        """
        assert os.path.splitext(pth_output)[-1] == ".nrrd", "the file should have '.nrrd' extension"
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        structure_mask_dict: dict = self.get_structure_mask(self.structure_names_dcm, mask_type=np.ndarray)
        
        if no_overlap:
            # create the sitk segmentation image
            sorted_by_size = _sort_segementation_dict_by_size(structure_mask_dict)
            all_masks = _convert_many_binary_masks_to_1_int_mask(sorted_by_size) # this removes overlap
            sitk_image = sitk.GetImageFromArray(all_masks.astype(int))
            sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
            sitk_image.SetSpacing(self.image_obj.spacing)
            sitk_image.SetOrigin(self.image_obj.origin)

            # Add necessary metadata for Slicer to recognize it as a segmentation
            # sitk_image.SetMetaData("Segmentation_MasterRepresentation", "Fractional labelmap")
            # sitk_image.SetMetaData("Segmentation_ReferenceImageExtentOffset", "0 0 0")
            # sitk_image.SetMetaData("Segmentation_SourceRepresentation", "Fractional")
            for i, name in enumerate(structure_mask_dict):
                label_dict = {
                    # f"Segment{i+1}_Tags": "Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|",
                    f"Segment{i+1}_Name": f"{name}",
                    f"Segment{i+1}_NameAutoGenerated": "0",
                    f"Segment{i+1}_LabelValue": f"{i+1}",
                    f"Segment{i+1}_ID": f"Segment{i+1}",
                    f"Segment{i+1}_Layer": "0",
                }
                for key, value in label_dict.items():
                    sitk_image.SetMetaData(key, value)
        else:
            raise NotImplementedError("Overlapping structures are not supported yet.")
        
        # Write the image
        writer = sitk.ImageFileWriter()
        writer.SetFileName(pth_output)
        writer.SetUseCompression(True)
        writer.Execute(sitk_image)

    def write_to_egsphant(
        self,
        pth_output: Path,
        material_dict: dict | Path,
        assign_material_from_ct: bool) -> None:
        r"""
        Purpose:
            - Write the BrachyPhantom object to an Egsphant file.
        Inputs:
            - pth_output: Path := the path to write the Egsphant file to.
        """
        assert os.path.splitext(pth_output)[-1] == ".egsphant", "the file should have '.egsphant' extension"
        from brachyutils.egsphant_utils import BrachyEgsphant
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        if self.egsphant_obj is not None:
            self.egsphant_obj.write_to_ctegsphant(pth_output)
        elif self.image_obj is not None:
            self.egsphant_obj = BrachyEgsphant(
                phantom=self,
                material_dict=material_dict,
                assign_material_from_ct=assign_material_from_ct,
            )
            self.egsphant_obj.write_to_ctegsphant(pth_output)
        else:
            raise ValueError("No image object or egsphant object to write to Egsphant file. Please load the image object first.")
# helper functions
def _sort_segementation_dict_by_size(seg_dict) -> dict:
    r"""
    Purpose:
        - will sort the items in a mask dictionary by the size of the segmentation.
    Inputs:
        - seg_dict: dict := the dictionary of the masks. the values are numpy arrays in
        [z, y, x] format.
    Outputs:
        - sorted_dict: dict := the sorted dictionary.
    """
    sorted_dict_list = sorted(
        seg_dict.items(), key=lambda x: np.sum(x[1]), reverse=True
    )
    return dict(sorted_dict_list)

def _convert_many_binary_masks_to_1_int_mask(seg_dict: dict) -> np.ndarray:
    r"""
    Purpose:
        - Convert many binary masks to one integer mask. The masks should be ordered 
        from largest to smallest as the smallest mask will overwrite the larger mask.
        use _sort_segementation_dict_by_size() to sort the masks.
    Inputs:
        - seg_dict: dict := the dictionary of the masks. the values are numpy arrays in
        [z, y, x] format.
    Outputs:
        - int_mask: np.ndarray := the integer mask.
    """
    int_mask = np.zeros_like(list(seg_dict.values())[0], dtype=int)
    for i, (key, mask) in enumerate(seg_dict.items()):
        int_mask[mask] = i + 1
    return int_mask

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

def readNrrdStruct(pth_structure: Path) -> RTStruct:
    r"""
    Purpose:
        - Load the NRRD structure file.
    Inputs:
        - pth_structure: Path := the path of the structure source file.
    Outputs:
        - RTStruct := the structure set object.
    Dependencies:
        - openTPS.core
    """
    assert os.path.exists(pth_structure), "The input path does not exist."
    assert ".seg.nrrd" in pth_structure, "The input file is not a NRRD structure file."
    sitk_image = sitk.ReadImage(pth_structure, imageIO="NrrdImageIO")
    segment_all_masks = sitk.GetArrayFromImage(sitk_image)
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()

    meta_data_keys = sitk_image.GetMetaDataKeys()
    structure_set = RTStruct()
    for key in meta_data_keys:
        if "_ID" in key:
            segment_id = sitk_image.GetMetaData(key)
            segment_name = sitk_image.GetMetaData(segment_id + "_Name")
            segment_label = sitk_image.GetMetaData(segment_id + "_LabelValue")
            segment_mask = segment_all_masks == int(segment_label)
            segment_mask = np.pad(segment_mask, 1, mode="constant", constant_values=0)
            roi_mask = ROIMask(
                imageArray=np.swapaxes(segment_mask, 0, 2),
                origin=origin,
                spacing=spacing,
                name=segment_name,
            )
            structure_set.appendContour(roi_mask.getROIContour())
    return structure_set