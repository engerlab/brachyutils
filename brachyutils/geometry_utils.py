import json
import os
import warnings
from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import SimpleITK as sitk

# import pydicom
from opentps.core.data import ROIContour, RTStruct
from opentps.core.data.images import CTImage, MRImage, ROIMask, Image3D
from opentps.core.io.dicomIO import (  # writeRTDose,
    readDicomCT,
    readDicomMRI,
    readDicomStruct,
    writeDicomCT,
    writeRTStruct,
)

# Imports for brachy applicator
from vtk import (
    vtkCellArray,
    vtkFillHolesFilter,
    vtkPoints,
    vtkPolyData,
    vtkTransform,
    vtkTransformPolyDataFilter,
)
from vtk.util import numpy_support
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter

import nrrd
import pydicom

class BrachyPhantom:
    r"""
    Puprose:
        - A class to load any voxelized geometry related to an HDR brachytherapy patient or phantom
        and perform some operations.
    Attributes:
        - pth_image: Path := the path of the geometry source file or files.
        - image_obj: CTImage or MRImage := the image of the patient loaded by openTPS. [x, y, z]
        - image_modality: Literal["CT", "MR", "US"] := the modality of the image.
        - structure_set: RTStruct := the structure set of the patient loaded by openTPS. [x, y, z].
        Other names for structure are contours, masks, segmentations.
        - structure_names_dcm: List[str] := the names of the structures in the dicom file.
        - unit_length: Literal["mm"] := the unit of length in the dicom file. default is mm.
        - xyz_format: bool := the format of the image. if True, the image is in [z, y, x] format.
        - orientation: Literal["LAS", "RAS", "LPS"] := the orientation of the image. default is LPS, same as 
        DICOM and slicer.
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
            - pth_phantom_file: Path := the path of the phantom .nrrd file.
            - pth_structures_file: Path := the path of the structure file.
            - pth_egsphant_file: Path := the path of the Egsphant file to be loaded.
            note that it is possible to generate an Egsphant from BrachyPhantom object.
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
        if dir_dicom is not None:
            self.pth_image: Path = Path(dir_dicom)
        elif pth_phantom_file is not None:
            self.pth_image: Path = Path(pth_phantom_file)
        else:
            self.pth_image = None
        self.image_obj: Union[CTImage, MRImage] = None
        self.image_modality: Literal["CT", "MR", "US"] = None
        self.structure_set: RTStruct = None
        self.structure_names_dcm: List[str] = []
        self.unit_length: Literal["mm"] = "mm"
        self.xyz_format: bool = True
        self.anatomical_coordinate_system: Literal["LAS", "RAS", "LPS"] = "LPS"
        # Attributes for Egsphant files
        from brachyutils.egsphant_utils import BrachyEgsphant

        self.egsphant_obj: "BrachyEgsphant" = None

        if dir_dicom is not None:
            self._load_dicom_image_files(self.pth_image)
        elif pth_phantom_file is not None:
            if str(pth_phantom_file).endswith(".nrrd"):
                self._load_nrrd_image_file(self.pth_image)
            elif str(pth_phantom_file).endswith(".nii.gz"):
                self._load_nifti_image_file(self.pth_image)
        elif pth_egsphant_file is not None:
            self.egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_egsphant_file)
        else:
            # raise ValueError(
            #     "No geometry source file provided. Please provide either the directory of the DICOM files or the path of the phantom file."
            # )
            warnings.warn("No geometry source file provided. Creating an empty Phantom", stacklevel=2)
        if self.image_obj is not None:
            self._convert_orientation_to_LPS()

        if pth_structures_file is not None:
            pth_structures_file = Path(pth_structures_file)
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
        # Load the images only, RD, RS, RP files are not needed here.
        image_files = [file for file in glob((str(pth_image) + "/*.dcm"))
                       if not os.path.basename(file).startswith("R")]
       
        if len(image_files) == 0:
            raise ValueError("No DICOM files found in the input directory.")
        if "CT" in image_files[0].upper():
            ct_files = list(filter(lambda s: "CT" in s.upper(), image_files))
            self.image_obj = readDicomCT(ct_files)
            self.image_modality = "CT"
            # get the orientation of the image
            header = pydicom.read_file(ct_files[0])
            orientation = header.get((0x0010, 0x2210), "LPS")
            if orientation == "BIPED":
                orientation = "LPS"
            self.anatomical_coordinate_system = orientation if orientation is not None else "LPS"
        
        elif "MR" in image_files[0].upper():
            mr_files = list(filter(lambda s: "MR" in s.upper(), image_files))
            self.image_obj = readDicomMRI(mr_files)
            self.image_modality = "MR"
            header = pydicom.read_file(ct_files[0])
            orientation = header.get((0x0010, 0x2210), "LPS")
            if orientation == "BIPED":
                orientation = "LPS"
            self.anatomical_coordinate_system = orientation if orientation is not None else "LPS"
        
        elif "US" in image_files[0].upper():
            us_files = list(filter(lambda s: "US" in s.upper(), image_files))
            self.image_obj = readDicomUS(us_files)
            self.image_modality = "US"
            header = pydicom.read_file(ct_files[0])
            orientation = header.get((0x0010, 0x2210), "LPS")
            if orientation == "BIPED":
                orientation = "LPS"
            self.anatomical_coordinate_system = orientation if orientation is not None else "LPS"
        else:
            raise ValueError("The image modality is not recognized. the dicom file names should contain CT, MR or US.")

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
        image_nrrd, header = nrrd.read(str(pth_image), index_order="C")
        origin = header["space origin"]
        affine = header["space directions"]
        spacing = affine.diagonal()

        # get the orientation of the image:
        orientation = header.get("space", "LPS")
        # orientation could be in spelled out, let's convert it to the 3 letter format
        if "superior" in orientation.lower() or "inferior" in orientation.lower():
            char_list = []
            if "left" in orientation.lower():
                char_list.append("L")
            elif "right" in orientation.lower():
                char_list.append("R")
            if "anterior" in orientation.lower():
                char_list.append("A")
            elif "posterior" in orientation.lower():
                char_list.append("P")
            if "superior" in orientation.lower():
                char_list.append("S")
            elif "inferior" in orientation.lower():
                char_list.append("I")
            orientation = "".join(char_list)

        self.anatomical_coordinate_system = orientation

        modality = header.get("modality", "unknown")
        if modality == "unknown":
            if "ct" in pth_image.name.lower():
                modality = "CT"
            elif "mr" in pth_image.name.lower():
                modality = "MR"
            elif "us" in pth_image.name.lower():
                modality = "US"
            else:
                warnings.warn("The modality of the image is not recognized.")

        self.image_obj = Image3D(
            origin=origin,
            spacing=spacing,
        )
        self.set_image_array(image_nrrd)
        self.image_modality = modality

    def _load_nifti_image_file(self, pth_image: Path) -> None:
        r"""
        Purpose:
            - Load the NIFTI image file.
        Inputs:
            - pth_image: Path := the path of the geometry source file.
        Outputs:
            - None
        Dependencies:
            - nibabel
        """
        import nibabel as nib

        assert pth_image.exists(), "The input path does not exist."
        image_nifti = nib.load(self.pth_image)
        orientation = "".join(nib.aff2axcodes(image_nifti.affine))
        image_data = np.ascontiguousarray(image_nifti.get_fdata())
        if image_data.ndim == 4:
            image_data = image_data[:, :, :, 0]
        if image_nifti.header.data_layout == "F":
            image_data = np.swapaxes(image_data, 0, 2)
        if image_nifti.header.default_x_flip:
            image_data = np.flip(image_data, axis=0)

        # if image_nifti.header
        # # flip the image if the orientation is not LPS:
        # # this worked for the messed up protate mri images from the micro-registration
        # # challenge. however, be careful with it on a new Nifti images. please
        # # do not modify the file writers.
        if orientation == "RAS":
            image_data = np.swapaxes(image_data, 0, 2)
            # image_data = np.swapaxes(image_data, 1, 2)
            orientation = "LPS"
        elif orientation == "LAS":
            image_data = np.flip(image_data, axis=1)
            orientation = "LPS"
        elif orientation == "LPS":
            pass
        else:
            raise ValueError("The orientation of the image is not recognized.")

        origin = image_nifti.affine[:3, 3]
        spacing = image_nifti.header.get("pixdim")[1:4]
        self.image_modality = image_nifti.header.get("modality", "unknown")
        if self.image_modality == "unknown":
            if "ct" in self.pth_image.name.lower():
                self.image_modality = "CT"
            elif "mr" in self.pth_image.name.lower():
                self.image_modality = "MR"
            elif "us" in self.pth_image.name.lower():
                self.image_modality = "US"
            else:
                warnings.warn("The modality of the image is not recognized.")

        self.anatomical_coordinate_system = orientation
        self.image_obj = Image3D(
            origin=origin,
            spacing=spacing,
        )
        self.set_image_array(image_data)

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
        # structure_file_type = "".join(pth_structure.suffixes)
        if str(pth_structure).endswith(".dcm"):
            self.structure_set = readDicomStruct(pth_structure)
            header = pydicom.read_file(pth_structure)
            structure_orientation = header.get((0x0010, 0x2210), "LPS")
            if structure_orientation == "BIPED":
                structure_orientation = "LPS"
            # self.anatomical_coordinate_system = orientation
        elif str(pth_structure).endswith(".nrrd"):
            self.structure_set, structure_orientation = readNrrdStruct(pth_structure)
            self.structure_set.setPatient(
                self.image_obj.patient if self.image_obj is not None else None
            )
            # self.structure_set.seriesInstanceUID = self.image_obj.seriesInstanceUID if self.structure_set is not None else ""
            # self.structure_set.sopInstanceUID = self.image_obj.sopInstanceUID if self.structure_set is None else ""
        elif str(pth_structure).endswith(".nii.gz"):
            self.structure_set, structure_orientation = readNiftiStruct(pth_structure)
        else:
            raise ValueError("The structure file type is not recognized.")

        if self.anatomical_coordinate_system is None:
            self.anatomical_coordinate_system = structure_orientation
        else:
            assert self.anatomical_coordinate_system == structure_orientation, "The orientation of the structure file is not the same as the image file."

        self.structure_names_dcm = []
        for structure in self.structure_set.contours:
            self.structure_names_dcm.append(structure.name)

    def get_structure_mask(
        self,
        query_structure_list: List[str],
        mask_type: Union[np.ndarray, ROIContour, ROIMask],
    ) -> Dict[str, Union[np.ndarray, ROIContour, ROIMask]]:
        r"""
        Purpose:
            To return a dictionary with the requested structure masks from BrachyPhantom object. The queried
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
        ), "structure masks have not been loaded yet. please run load_structure_file() first"
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
        print(
            f"Geometry File source: {self.pth_image}"
            if self.pth_image is not None
            else "No geometry file source."
        )
        print(
            f"Image Modality: {self.image_modality}"
            if self.image_modality is not None
            else "No image modality."
        )
        print(
            f"Unit Length: {self.unit_length}"
            if self.unit_length is not None
            else "No unit length."
        )
        print(
            f"Image Shape [x, y, z]: {self.image_obj.gridSize}"
            if self.image_obj is not None
            else "No image object."
        )
        print(
            f"Image size in world unit [x, y, z]: {self.image_obj.gridSizeInWorldUnit}"
            if self.image_obj is not None
            else "No image object."
        )
        print(
            f"Image Origin [x, y, z]: {self.image_obj.origin}"
            if self.image_obj is not None
            else "No image object."
        )
        print(
            f"Image Spacing [x, y, z]: {self.image_obj.spacing}"
            if self.image_obj is not None
            else "No image object."
        )
        print(
            f"Structure Names: {self.structure_names_dcm}"
            if self.structure_names_dcm is not None
            else "No structure names."
        )
        print(
            f"Structure Count: {len(self.structure_names_dcm)}"
            if self.structure_names_dcm is not None
            else "No structure names."
        )

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

    def set_image_array(self, image_array: np.ndarray) -> None:
        r"""
        Purpose:
            - To set the image array.
        """
        self.image_obj.imageArray = np.swapaxes(image_array, 0, 2)
    
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

    def write_image_to_nrrd(
        self,
        pth_output: Path,
        metadata: Optional[Dict[str, str]] = None,
        ) -> None:
        r"""
        Purpose:
            - To write the image to a nrrd file. By default, all images are written as Left Posterior Superior.
        Inputs:
            - pth_output: Path := the path to write the image to.
            - metadata := a dictionary containing the following meta data key values (should be changed later):
                "cancer site":
                "care center":
                "number of dwell positions":
                "number of segmented structures":
                "patient number":
                "Image content": "[3D dose, 3D uncertainty]"
        Outputs
            - None
        Dependencies:
            - pynrrd
        """
        assert (
            os.path.splitext(pth_output)[-1] == ".nrrd"
        ), "the file should have '.nrrd' extension"
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        from collections import defaultdict
        
        image_array_zyx = self.get_image_array()
        header = defaultdict(str)
        header["type"] = "double"
        # header["space dimension"] = "3"
        header["space"] = self.anatomical_coordinate_system
        header["sizes"] = (
            " ".join(map(str, self.image_obj.gridSize.tolist()))
        )
        header["space directions"] = [
            [self.image_obj.spacing[0], 0.0, 0.0],
            [0.0, self.image_obj.spacing[1], 0.0],
            [0.0, 0.0, self.image_obj.spacing[2]],
        ]
        header["kinds"] = ["space", "space", "space"]
        header["labels"] = ["x", "y", "z"]
        header["endian"] = "little"
        header["encoding"] = "gzip"
        header["space origin"] = self.image_obj.origin.tolist()
        header["voxel spacing"] = self.image_obj.spacing.tolist()
        header["space units"] = ["mm", "mm", "mm"]
        header["modality"] = self.image_modality
        header = header | metadata if metadata is not None else header
        nrrd.write(str(pth_output), image_array_zyx, header, index_order="C")

    def write_structures_to_nrrd(
        self,
        pth_output: Path,
        overlap: Optional[bool] = True,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        r"""
        Purpose:
            - To write the structures to a nrrd file. By defualt, we remove the overlap between the structures. the smaller structures
            overwrite the larger structures if there is an overlap.
        Inputs:
            - pth_output: Path := the path to write the structures to.
            - overlap: Optional[bool] := if True, the structures will be written with overlap, meaning each structure will be represented
            by a binary matrix with 1s and 0s. if False, the structures will be written without overlap, an integer is asigned to 
            each structure and all the structures are represented by a single volume.
            - metadata: Optional[Dict[str, str]] := a dictionary containing any meta data additional to the minimal required meta data.
        Outputs:
            - None
        Dependencies:
            - pynrrd
        """
        assert (
            os.path.splitext(pth_output)[-1] == ".nrrd"
        ), "the file should have '.nrrd' extension"
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        structure_mask_dict: dict = self.get_structure_mask(
            self.structure_names_dcm, mask_type=np.ndarray
        )

        if not overlap:

            # this removes overlap
            sorted_by_size = _sort_segementation_dict_by_size(structure_mask_dict)
            all_masks = _convert_many_binary_masks_to_1_int_mask(
                sorted_by_size
            )
            from collections import defaultdict
            # # Generic phantom meta data
            header = defaultdict(str)
            header["type"] = "double"
            # header["space dimension"] = "3"
            header["space"] = self.anatomical_coordinate_system
            header["sizes"] = (
                " ".join(map(str, self.image_obj.gridSize.tolist()))
            )
            header["space directions"] = [
                [self.image_obj.spacing[0], 0.0, 0.0],
                [0.0, self.image_obj.spacing[1], 0.0],
                [0.0, 0.0, self.image_obj.spacing[2]],
            ]
            header["kinds"] = ["space", "space", "space"]
            header["labels"] = ["x", "y", "z"]
            header["endian"] = "little"
            header["encoding"] = "gzip"
            header["space origin"] = self.image_obj.origin.tolist()
            header["voxel spacing"] = self.image_obj.spacing.tolist()
            header["space units"] = ["mm", "mm", "mm"]
  
        else:
            # stack up all the masks
            sorted_by_size = _sort_segementation_dict_by_size(structure_mask_dict)
            all_masks = np.stack(list(sorted_by_size.values()), axis=3).astype(np.uint8)
            from collections import defaultdict
            # # Generic phantom meta data
            header = defaultdict(str)
            header["type"] = "unsigned char"
            header["space dimension"] = "4"
            header["space"] = self.anatomical_coordinate_system
            header["sizes"] = (
                " ".join(map(str, [all_masks.shape[-1]]+self.image_obj.gridSize.tolist()))
            )
            header["space directions"] = [
                [np.nan, np.nan, np.nan],
                [self.image_obj.spacing[0], 0.0, 0.0],
                [0.0, self.image_obj.spacing[1], 0.0],
                [0.0, 0.0, self.image_obj.spacing[2]],
            ]
            header["kinds"] = ["list", "domain", "domain", "domain"]
            # header["labels"] = ["x", "y", "z"]
            header["endian"] = "little"
            header["encoding"] = "gzip"
            header["space origin"] = self.image_obj.origin.tolist()
            # header["voxel spacing"] = self.image_obj.spacing.tolist()
            # header["space units"] = ["mm", "mm", "mm"]

        # # Generic Segmentation meta data
        header["Segmentation_ContainedRepresentationNames"] = "Binary labelmap|Closed surface|"
        header["Segmentation_MasterRepresentation"] = "Binary labelmap"
        header["Segmentation_ReferenceImageExtentOffset"] = "0 0 0"
        # header["Segmentation_ConversionParameters"] = "None"  this one is crazy long
        # # Specific segmentation meta data
        for i, name in enumerate(sorted_by_size):
            # header[f"Segment{i}_Color"] = 
            # header[f"Segment{i}_ColorAutoGenerated"] =
            header[f"Segment{i}_ID"] = f"Segment_{i+1}"
            header[f"Segment{i}_LabelValue"] = f"{i+1}"
            header[f"Segment{i}_Layer"] = f"{i}" if overlap else "0"
            header[f"Segment{i}_Name"] = f"{name}"
            header[f"Segment{i}_NameAutoGenerated"] = "0"
            header[f"Segment{i}_Extent"] = " ".join(map(str, _getExtentOfMask(sorted_by_size[name])))
            header[f"Segment{i}_Tags"] = "Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|"

        # # any other meta data
        header = header | metadata if metadata is not None else header

        # # Write the image
        nrrd.write(str(pth_output), all_masks, header, index_order="C")

    def write_to_egsphant(
        self,
        pth_output: Path,
        material_dict: dict | Path = None,
        assign_material_from_ct: bool = None,
    ) -> None:
        r"""
        Purpose:
            - Write the BrachyPhantom object to an Egsphant file.
        Inputs:
            - pth_output: Path := the path to write the Egsphant file to.
            - material_dict: dict | Path := the dictionary of the materials. if Path, the path to the material file.
            The dictionary contains the name of the elements for each voxel,
            and the following keys: [
                "density" := the density of the material in g/cm^3,
                "HU_limit" := the lower HU limit threshold of the material,
                "structure_name := {optional} the name of the structure in the dicom file that represents the material,"
            ]
            - assign_material_from_ct: bool := if True, the material will be assigned from the CT image.
        """
        assert (
            os.path.splitext(pth_output)[-1] == ".egsphant"
        ), "the file should have '.egsphant' extension"
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        if self.egsphant_obj is not None:
            self.egsphant_obj.write_to_ctegsphant(pth_output)
        elif self.image_obj is not None:
            from brachyutils.egsphant_utils import BrachyEgsphant

            self.egsphant_obj = BrachyEgsphant(
                phantom=self,
                material_dict=material_dict,
                assign_material_from_ct=assign_material_from_ct,
            )
            self.egsphant_obj.write_to_ctegsphant(pth_output)
        else:
            raise ValueError(
                "No image object or egsphant object to write to Egsphant file. Please load the image object first."
            )

    def crop_by_coordinates(
        self, croodinate_range: List[float] | np.array, inplace: "BrachyPhantom" = True
    ) -> None:
        r"""
        Purpose:
            - Crop the phantom by the input coordinates.
        Inputs:
            - coordinate_range := a 3 x 2 array holding the min and max on x, y and z axis
                [[ix_min, ix_max], [iy_min, iy_max], [iz_min, iz_max]]
            - inplace := if True, the cropping will be done in place. if False, a new BrachyPhantom object will be returned.
        Outputs:
            - None
        """
        import copy

        from opentps.core.processing.imageProcessing.resampler3D import (
            crop3DDataAroundBox,
        )

        croodinate_range = np.array(croodinate_range)
        assert croodinate_range.shape == (
            3,
            2,
        ), "coordinate_range should be a 3x2 array in x, y, z order"
        if inplace:
            crop3DDataAroundBox(self.image_obj, croodinate_range, marginInMM=[1, 1, 1])
        else:
            new_phantom: BrachyPhantom = copy.deepcopy(self)
            new_phantom.crop_by_coordinates(croodinate_range, inplace=True)
            return new_phantom

    def crop_by_index(
        self, index_range: List[int] | np.array, inplace: Optional[bool] = True
    ) -> Union[None, "BrachyPhantom"]:
        r"""
        Purpose:
            - given a range of indicies (mix and max on each axis), this function will crop
            image_obj and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyEgsphant object
            - index_range := a 3 x 2 array holding the min and max index on x, y and z axis
                [[ix_min, ix_max], [iy_min, iy_max], [iz_min, iz_max]]
            - inplace := if True, the cropping will be done in place. if False, a new BrachyPhantom object will be returned.
        Outputs:
            - None
        """
        index_range = np.array(index_range)
        assert index_range.shape == (
            3,
            2,
        ), "index_range should be a 3x2 array in x, y, z order"
        new_origin_coords = self.density_image.getPositionFromVoxelIndex(
            index_range[:, 0]
        )
        new_ending_coords = self.density_image.getPositionFromVoxelIndex(
            index_range[:, 1]
        )
        new_coords_range = np.column_stack([new_origin_coords, new_ending_coords])
        return self.crop_by_coordinates(new_coords_range, inplace)

    def crop_by_contour(
        self, contour_name: str, inplace: Optional[bool] = True
    ) -> Union[None, "BrachyPhantom"]:
        r"""
        Purpose:
            - Crop the phantom by the input contours.
        Inputs:
            - contour_name: str := the name of the contour to crop by.
            - inplace: Optional[None | "BrachyPhantom"] := if True, the cropping will be done in place. if False, a new BrachyPhantom object will be returned.
        Outputs:
            - None
        """
        from opentps.core.processing.imageProcessing.resampler3D import (
            resampleImage3DOnImage3D,
        )
        from opentps.core.processing.segmentation.segmentation3D import getBoxAroundROI

        mask_dict = self.get_structure_mask([contour_name], mask_type=ROIMask)
        resampled_mask = resampleImage3DOnImage3D(
            mask_dict[contour_name], self.image_obj
        )
        box_around_mask = np.array(getBoxAroundROI(resampled_mask))
        return self.crop_by_coordinates(box_around_mask, inplace)

    def set_structure_set(self, mask_dict: dict) -> None:
        r"""
        Purpose:
            - Set the structure set with the input mask dictionary.
        Inputs:
            - mask_dict: dict := the dictionary of the masks. the values are numpy arrays in
            [z, y, x] format.
        Outputs:
            - None
        """
        for structure_name in mask_dict:
            self.structure_set.removeContour(
                self.structure_set.getContourByName(structure_name)
            )
            self.structure_set.appendContour(
                mask_dict.get(structure_name).getROIContour()
            )

    def _convert_orientation_to_LPS(self) -> None:
        r"""
        Purpose:
            - Convert the orientation of the image from what ever it is to LPS.
        Inputs:
            - None
        Outputs:
            - None
        """
        # raise DeprecationWarning("This function is deprecated. converting to LPS is done when loading from each file type.")
        assert self.image_obj is not None, "No image object to convert orientation."
        assert self.anatomical_coordinate_system is not None, "Orientation is not set."
        if self.anatomical_coordinate_system == "LAS":
            raise NotImplementedError("Conversion from LAS to LPS is not implemented yet.")
        elif self.anatomical_coordinate_system == "RAS":
            # raise NotImplementedError("Conversion from RAS to LPS is not implemented yet.")
            image_array = self.get_image_array()
            # image_array = np.flip(image_array, axis=0)
            # image_array = np.flip(image_array, axis=1)
            self.anatomical_coordinate_system = "LPS"
            
        elif self.anatomical_coordinate_system == "LPS":
            pass
        else:
            raise ValueError("The orientation is not recognized. please leave an issue on github.")
        self.anatomical_coordinate_system = "LPS"

# helper functions
def phantom_with_empty_image_like(phantom: BrachyPhantom) -> BrachyPhantom:
    r"""
    Purpose:
        - Create a new BrachyPhantom object with the same structure set as the input phantom but with an empty image.
    Inputs:
        - phantom: BrachyPhantom := the input phantom object.
    Outputs:
        - new_phantom: BrachyPhantom := the new phantom object.
    """
    new_phantom = BrachyPhantom()
    new_phantom.pth_image = None
    new_phantom.image_obj = None
    new_phantom.image_modality = phantom.image_modality
    new_phantom.structure_set = phantom.structure_set
    new_phantom.structure_names_dcm = phantom.structure_names_dcm
    new_phantom.unit_length = phantom.unit_length
    new_phantom.xyz_format = phantom.xyz_format

    return new_phantom

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
    for i, (_, mask) in enumerate(seg_dict.items()):
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


def readNrrdStruct(pth_structure: Path) -> Union[RTStruct, str]:
    r"""
    Purpose:
        - Load the NRRD structure file.
    Inputs:
        - pth_structure: Path := the path of the structure source file.
    Outputs:
        - RTStruct := the structure set object.
        - str := the orientation of the structure mask, which is recommended to be LPS.
    Dependencies:
        - openTPS.core
    """
    assert os.path.exists(pth_structure), "The input path does not exist."
    assert ".seg.nrrd" in str(pth_structure), "The input file is not a NRRD structure file."
    structures_data, header = nrrd.read(str(pth_structure), index_order="C")
    origin = header["space origin"]
    overlap = True if structures_data.ndim == 4 else False
    affine = header["space directions"]
    if overlap:
        affine = affine[1:]
    spacing = affine.diagonal()
    # get orientation of the image:
    orientation = header.get("space", "LPS")
        # orientation could be in spelled out, let's convert it to the 3 letter format
    if "superior" in orientation.lower() or "inferior" in orientation.lower():
        char_list = []
        if "left" in orientation.lower():
            char_list.append("L")
        elif "right" in orientation.lower():
            char_list.append("R")
        if "anterior" in orientation.lower():
            char_list.append("A")
        elif "posterior" in orientation.lower():
            char_list.append("P")
        if "superior" in orientation.lower():
            char_list.append("S")
        elif "inferior" in orientation.lower():
            char_list.append("I")
        orientation = "".join(char_list)

    structure_set = RTStruct()
    i = 0
    for key in header:
        if f"Segment{i}_Name" == key:
            label_value = header[f"Segment{i}_LabelValue"]
            name = header[f"Segment{i}_Name"]
            if overlap:
                segment_mask = structures_data[:, :, :, i]
            else:
                segment_mask = structures_data == int(label_value)
            segment_mask = np.pad(segment_mask, 1, mode="constant", constant_values=0)
            roi_mask = ROIMask(
                imageArray=np.swapaxes(segment_mask, 0, 2),
                origin=origin,
                spacing=spacing,
                name=name,
            )
            structure_set.appendContour(roi_mask.getROIContour())
            i += 1
    return structure_set, orientation

def readNiftiStruct(pth_structure: Path) -> Union[RTStruct, str]:
    r"""
    Purpose:
        - Load the NIFTI structure file.
    Inputs:
        - pth_structure: Path := the path of the structure source file.
    Outputs:
        - RTStruct := the structure set object.
    Dependencies:
        - nibabel
    """
    assert os.path.exists(pth_structure), "The input path does not exist."
    import nibabel as nib
    structure_nifti = nib.load(pth_structure)
    orientation = "LPS"#"".join(nib.aff2axcodes(structure_nifti.affine))
    structure_data = np.ascontiguousarray(structure_nifti.get_fdata())
    origin = structure_nifti.affine[:3, 3]
    spacing = structure_nifti.header.get("pixdim")[1:4]

    # God knows what is the name of the structures in the nifti files
    # I will just number them and hope for the best
    num_structures = structure_nifti.header.get("dim")[0]
    structure_set = RTStruct()
    for i in range(num_structures):
        # generate segment labels
        segment_id = f"Segment{i+1}"
        segment_name = segment_id + "_Name"
        segment_label =  segment_id + "_LabelValue"
        # get the segment mask
        segment_mask = structure_data
        if structure_data.ndim == 4:
            segment_mask = structure_data[:, :, :, i]
        else:
            segment_mask = structure_data == i
        segment_mask = np.pad(segment_mask, 1, mode="constant", constant_values=0)
        # flip the image if the orientation is not LPS:
        # this worked for the messed up protate mri images from the micro-registration
        # challenge. however, be careful with it on a new Nifti images. please
        # do not modify the file writers.
        # if orientation == "RAS":
        #     # segment_mask = np.flip(segment_mask, axis=0)
        #     # segment_mask = np.flip(segment_mask, axis=1)
        #     segment_mask = np.swapaxes(segment_mask, 1, 2)
        #     orientation = "LPS"
        # elif orientation == "LAS":
        #     segment_mask = np.flip(segment_mask, axis=1)
        #     orientation = "LPS"
        # elif orientation == "LPS":
        #     pass
        # else:
        #     raise ValueError("The orientation of the image is not recognized.")

        roi_mask = ROIMask(
            imageArray=np.swapaxes(segment_mask, 0, 2),
            origin=origin,
            spacing=spacing,
            name=segment_name,
        )
        structure_set.appendContour(roi_mask.getROIContour())
    return structure_set, orientation    

def _get_image_orientation(pth_image: Path) -> str:
    """
    Purpose:
        - Get the image orientation from the DICOM, NRRD or NIFTI files.
        The orientation could be LAS, RAS, or LPS. BrachyUtils by default
        uses LPS orientation, which is the default in DICOM standard and likely
        the origin of all medical images.
    Inputs:
        - pth_image: Path := the path of the image file. Hopefully
        it has some sort of header information.
    Outputs:
        - orientation: str := the orientation of the image.
    Depenedencies:
        - pydicom
        - nibabel
        - pynrrd
    """
    raise DeprecationWarning("This function will soon be deleted. orientation should be handled in each file type loader.")
    # extension = "".join(pth_image.suffixes)
    if str(pth_image).endswith(".dcm"):
        import pydicom
        header = pydicom.read_file(pth_image)
        orientation = header.get((0x0010, 0x2210))
        if orientation is not None:
            return orientation
        else:
            # default orientation in dicom is LPS
            return "LPS"
    elif str(pth_image).endswith(".nrrd"):
        warnings.warn("NRRD orientation is not tested yet")
        import nrrd
        nrrd_header = nrrd.read(pth_image, index_order="C")[1]
        orientation = nrrd_header.get("space directions")
        if orientation is not None:
            if "left" in orientation[0] and "posterior" in orientation[1]:
                return "LAS"
            elif "right" in orientation[0] and "posterior" in orientation[1]:
                return "RAS"
            elif "left" in orientation[0] and "anterior" in orientation[1]:
                return "LPS"
            elif "right" in orientation[0] and "anterior" in orientation[1]:
                return "RPS"
            else:
                return "LAS"
        else:
            return "LPS"
    elif str(pth_image).endswith(".nii.gz"):
        import nibabel as nib
        nifti_image = nib.load(pth_image)
        # Get the affine matrix
        affine = nifti_image.affine
        # Check the signs of the first two columns
        if affine[0, 0] > 0 and affine[1, 1] > 0:
            return "LPS"
        elif affine[0, 0] < 0 and affine[1, 1] < 0:
            return "RAS"
        elif affine[0, 0] > 0 and affine[1, 1] < 0:
            return "LAS"
        else:
            print("The orientation is neither RAS nor LPS")
    else:
        return "LPS"

def _getExtentOfMask(mask: np.array) -> List[int]:
    r"""
    Purpose:
        - Get the extent of the mask in voxel indecies.
    Inputs:
        - mask: np.array := the mask, whcih is a binary numpy array (z, y, x).
    Outputs:
        - extent: List[int] := the extent of the mask in [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    ones = np.where(mask == True)
    boxInVoxel = [np.min(ones[2]), np.max(ones[2]),
                np.min(ones[1]), np.max(ones[1]),
                np.min(ones[0]), np.max(ones[0])]
    # for bound in boxInVoxel:
    #     if bound == 0:
    #         bound = 1
    #     if bound == mask.shape[0]:
    #         bound = mask.shape[0] - 1
    #     if bound == mask.shape[1]:
    #         bound = mask.shape[1] - 1
    #     if bound == mask.shape[2]:
    #         bound = mask.shape[2] - 1
    return boxInVoxel

class BrachyApplicator:
    r"""
    Purpose:
        - This class holds the information regarding the brachytherapy applicator.
        as well as all the functions to support the necessary applicator operations.

    Attributes:
        - path:str := path to the applicator geometry file.
        - name:str := name of the applicator, which is taken as the basename of the path.
        - applicator_mesh := the vtk mesh of the applicator.
        - verticies:np.array := the verticies of the applicator mesh.
        - faces:np.array := the faces of the applicator mesh.
        - origin:np.array := the origin of the applicator.
        - rotation:np.array := the rotation of the applicator.
        - material:str := the material of the applicator.
        - density:float := the density of the applicator.
        - normal:np.array := the normal of the applicator in the patient coordinate system. this is used for RapidBrachy only.

    Functions:
        - load_stl(pth_input:str)
        - load_json(pth_input:str)
        - to_dict()
        - to_json(pth_output:str)
    """

    def __init__(
        self,
        pth_input_file: str,
        material: str = None,
        density: float = None,
        origin: np.array = None,
        rotation: np.array = None,
        rotation_origin: np.array = None,
        coordinates: np.array = None,
        normal: np.array = None,
        catheter_trajectory: list = None,
    ) -> None:
        """
        Purpose:
            - Initialize the Applicator object.
        Inputs:
            - pth_input_file (str): The path to the input file.
            - material (str, optional): The material of the applicator. Defaults to None.
            - density (float, optional): The density of the applicator. Defaults to None.
            - origin (np.array, optional): The origin of the applicator in [x,y,z] . Defaults to None.
            - rotation (np.array, optional): The rotation vector of the applicator in [w,x,y,z]. Defaults to None.
            - rotation_origin (np.array, optional): The origin point with respect to which the rotaion vector is created.
            - coordinates (np.array, optional): The coordinates of the applicator in patient frame. Defaults to None.
            - normal (np.array, optional): The normal of the applicator in the patient frame. Defaults to None.
            - catheter_trajectory: (list, optional): The list of start dwell poisition and end dwell position of the catheter inside
            the applicator [[x,y,z,x,y,z]]. Defaults to None.
        Outputs:
            - Void: an applicator object is created dependeing on the inputs.
        """
        assert os.path.exists(
            pth_input_file
        ), f"input file {pth_input_file} does not exist"
        self.path = pth_input_file
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        self.applicator_mesh: vtkPolyData = None
        self.verticies: np.array = None
        self.faces: np.array = None
        self.origin: np.array = np.array([0, 0, 0])  # [x, y, z]
        self.rotation: np.array = np.array([0, 0, 0, 0])  # [w, x, y, z]
        self.coordinates: np.array = np.array([0, 0, 0])  # [x, y, z]
        self.material: str = None
        self.density: float = None
        self.normal: np.array = None
        self.catheter_trajectory: np.array = None

        input_extension = os.path.splitext(self.path)[1]
        if input_extension == ".stl":
            self.load_stl(self.path)
        elif input_extension == ".json":
            self.load_json(self.path)
        else:
            raise ValueError("invalid input file extension")

        if material is not None:
            self.material = material
        if density is not None:
            self.density = density
        if origin is not None:
            self.set_origin(origin)
        if rotation is not None and rotation_origin is not None:
            self.set_rotation(rotation, rotation_origin)
        if coordinates is not None:
            self.set_coordinates(coordinates)
        if normal is not None:
            self.normal = normal
        if catheter_trajectory is not None:
            self.catheter_trajectory = catheter_trajectory

    def load_stl(self, pth_input: str) -> None:
        r"""
        Purpose:
            - To load the applicator geometry from an stl file.
        Inputs:
            - pth_input:str := path to the stl file containing the applicator geometry.
        Outputs:
            - Void := will update the BrachyApplicator object based on the stl file.
        """
        reader = vtkSTLReader()
        reader.SetFileName(pth_input)
        reader.Update()
        self.applicator_mesh = reader.GetOutput()
        self._update_brachy_applicator_from_applicator_mesh()

    def load_json(self, pth_input: str) -> None:
        r"""
        Purpose:
            - To load the applicator geometry from a json file.
        Inputs:
            - pth_input:str := path to the stl file containing the applicator geometry.
        Outputs:
            - Void := will update the BrachyApplicator object based on the json file.
        """
        with open(pth_input, "r") as json_file:
            applicator_dict = json.load(json_file)

        self.verticies = np.array(applicator_dict["verticies"], dtype=np.float32)
        self.faces = np.array(applicator_dict["faces"], dtype=np.int32)
        self.set_origin(np.array(applicator_dict["origin"]))
        self.set_rotation(np.array(applicator_dict["rotation"]))
        self.set_coordinates(np.array(applicator_dict["coordinates"]))
        self.material = applicator_dict["material"]
        self.density = applicator_dict["density"]

    def load_mac(self, pth_input: str) -> None:
        r"""
        Purpose:
            - To load the applicator geometry from a mac file.
        Inputs:
            - pth_input:str := path to the mac file containing the applicator geometry.
        Outputs:
            - Void := will update the BrachyApplicator object based on the mac file.
        """
        raise NotImplementedError("to be implemented soon")

    def info(self) -> None:
        r"""
        Purpose:
            - To print the information about the applicator.
        Inputs:
            - self := the BrachyApplicator object.
        Outputs:
            - Void := will print the information about the applicator.
        """
        print("Applicator info is as follows:")
        print(self.to_dict())

    def is_equal(self, other) -> bool:
        r"""
        Purpose:
            - To compare the current applicator with another applicator.
        Inputs:
            - other:BrachyApplicator := the other applicator to compare with.
        Outputs:
            - bool := True if the two applicators are equal, False otherwise.
        """
        if type(self) is not type(other):
            return False
        if self.name != other.name:
            return False
        if not np.isclose(self.verticies, other.verticies, atol=1e-6).all():
            return False
        if not np.isclose(self.faces, other.faces, atol=1e-6).all():
            return False
        if not np.isclose(self.origin, other.origin, atol=1e-6).all():
            return False
        if not np.isclose(self.rotation, other.rotation, atol=1e-6).all():
            return False
        if self.material != other.material:
            return False
        if self.density != other.density:
            return False
        return True

    def _update_applicator_mesh_from_brachy_applicator(self) -> None:
        r"""
        Purpose:
            - To update the applicator mesh from the verticies and faces.
        Inputs:
            - self := the BrachyApplicator object.
        Outputs:
            - Void := will update the applicator mesh from the verticies and faces.
        """
        points = vtkPoints()
        for vertex in self.verticies:
            points.InsertNextPoint(vertex)
        self.applicator_mesh.SetPoints(points)

        cell_array = vtkCellArray()
        for face in self.faces:
            cell_array.InsertNextCell(3, face)
        self.applicator_mesh.SetPolys(cell_array)
        fill_holes_filter = vtkFillHolesFilter()
        fill_holes_filter.SetInputData(self.applicator_mesh)
        fill_holes_filter.Update()
        self.applicator_mesh = fill_holes_filter.GetOutput()

    def _update_brachy_applicator_from_applicator_mesh(self) -> None:
        r"""
        Purpose:
            - To update the brachy applicator from the applicator mesh.
        Inputs:
            - self := the BrachyApplicator object.
        Outputs:
            - Void := will update the brachy applicator from the applicator mesh.
        """
        self.verticies = numpy_support.vtk_to_numpy(
            self.applicator_mesh.GetPoints().GetData()
        )
        self.faces = numpy_support.vtk_to_numpy(
            self.applicator_mesh.GetPolys().GetData()
        )
        self.faces = self.faces.reshape(-1, 4)[:, 1:]

    def set_origin(self, origin: np.array) -> None:
        r"""
        Purpose:
            - To set the origin of the applicator.
        Inputs:
            - origin:np.array := the origin of the applicator.
        Outputs:
            - Void := will update the applicator verticies based on the new origin.
        """
        old_origin = self.origin
        change_in_origin = np.ones_like(self.verticies) * (origin - old_origin)
        self.origin = origin
        self.verticies += change_in_origin
        self._update_applicator_mesh_from_brachy_applicator()

    def set_rotation(
        self, rotation: np.array, rotation_origin: np.array = None
    ) -> None:
        r"""
        Purpose:
            - To set the rotation of the applicator.
            the rotation origin is assumed to be the origin of applicator. To rotate the
            applicator around its center, coordinates of the center of applicator should
            be provided. The rotation angle is the first element of the rotation vector. the rotation
            axis is the last three elements of the rotation vector [w,x,y,z].
        Inputs:
            - rotation:np.array := the rotation of the applicator.
            The rotation vector is in quaternion ([w, x, y, z]).
            - rotation_origin:np.array := the origin of the rotation. if not provided, the
            origin of the applicator will be used.
        Outputs:
            - Void := will update the applicator verticies based on the new rotation.
        """
        # set the rotation attribute
        self.rotation = rotation
        # by default, the rotation origin is the origin of the applicator
        # if rotation is provided, the applicator is translated to the rotation origin
        # then it is rotated and translated back to the original position.
        if rotation_origin is not None:
            transform_translate = vtkTransform()
            transform_translate.Translate(
                -rotation_origin[0], -rotation_origin[1], -rotation_origin[2]
            )
            transform_translate_filter = vtkTransformPolyDataFilter()
            transform_translate_filter.SetTransform(transform_translate)
            transform_translate_filter.SetInputData(self.applicator_mesh)
            transform_translate_filter.Update()
            self.applicator_mesh = transform_translate_filter.GetOutput()

        # # now apply the rotation
        # create the transformation matrix
        transform = vtkTransform()
        transform.RotateWXYZ(rotation[0], rotation[1], rotation[2], rotation[3])

        # apply the transformation
        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(self.applicator_mesh)
        transform_filter.Update()
        self.applicator_mesh = transform_filter.GetOutput()

        # if rotation origin is provided, translate the applicator back to the original position
        if rotation_origin is not None:
            transform_translate = vtkTransform()
            transform_translate.Translate(
                rotation_origin[0], rotation_origin[1], rotation_origin[2]
            )
            transform_translate_filter = vtkTransformPolyDataFilter()
            transform_translate_filter.SetTransform(transform_translate)
            transform_translate_filter.SetInputData(self.applicator_mesh)
            transform_translate_filter.Update()
            self.applicator_mesh = transform_translate_filter.GetOutput()

        # update the BrachyApplicator based on the transformation
        self._update_brachy_applicator_from_applicator_mesh()

    def set_coordinates(self, coordinates: np.array) -> None:
        r"""
        Purpose:
            - to located the applicator at a given coordinate with respect to
            self.origin.
        Inputs:
            - coordinates:np.array := the coordinates of the applicator.
        Outputs:
            - Void := will update the applicator verticies based on the new coordinates.
        """
        # set the coordinate attributes
        self.coordinates = coordinates

        # create transformation matrix
        transform = vtkTransform()
        transform.Translate(coordinates[0], coordinates[1], coordinates[2])

        # apply the transformation
        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(self.applicator_mesh)
        transform_filter.Update()
        self.applicator_mesh = transform_filter.GetOutput()

        # update the BrachyApplicator based on the transformation
        self._update_brachy_applicator_from_applicator_mesh()

    def _update_catheter_trajectory(
        self,
    ) -> None:
        r"""
        Purpose:
            - to update the trajectory of the dwell positions inside the applicator after the applicator has
            been rotated or translated.
        Inputs:
            - self := the BrachyApplicator object.
        Outputs:
            - Void := will update the catheter trajectory.
        """

        raise NotImplementedError("to be implemented soon")

    def to_dict(self) -> dict:
        r"""
        Purpose:
            - To convert the applicator geometry to a dictionary.
        Inputs:
            - self := the BrachyApplicator object.
        Outputs:
            - dict := the dictionary containing the applicator geometry.
        """
        return {
            "name": self.name,
            "path": self.path,
            # "verticies": self.verticies.tolist(),
            # "faces": self.faces.tolist(),
            "origin": self.origin,
            "rotation": self.rotation,
            "material": self.material,
            "density": self.density,
            "normal": self.normal,
            "catheter_trajectory": self.catheter_trajectory,
        }

    def to_json(self, pth_output: str) -> None:
        r"""
        Purpose:
            - To save the applicator geometry to a json file.
        Inputs:
            - pth_output:str := path to the output json file.
        Outputs:
            - Void := will save the applicator geometry to a json file.
        """
        applicator_dict = self.to_dict()

        with open(pth_output, "w") as json_file:
            json.dump(applicator_dict, json_file, indent=4)

    def to_mac(self, pth_output: str) -> None:
        r"""
        Purpose:
            - To save the applicator geometry to a mac file.
        Inputs:
            - pth_output:str := path to the output mac file.
        Outputs:
            - Void := will save the applicator geometry to a mac file.
        """
        macfile_string = ""

        # add in the vertex info
        float_formatter = "{:.3f}".format
        for vertex in self.verticies:
            macfile_string += f"/applicator/vertex {float_formatter(vertex[0])} {float_formatter(vertex[1])} {float_formatter(vertex[2])} mm\n"

        # add in the face info
        for face in self.faces:
            macfile_string += f"/applicator/face {face[0]} {face[1]} {face[2]}\n"
        # add in the material info
        macfile_string += f"/applicator/material {self.material}\n"
        # add in the density info
        macfile_string += f"/applicator/density {self.density}\n"
        # add in the origin info
        macfile_string += "/applicator/xPosition 0 mm\n"
        macfile_string += "/applicator/yPosition 0 mm\n"
        macfile_string += "/applicator/zPosition 0 mm\n"
        # add in rotation nfo
        macfile_string += "/applicator/xRotation 0 deg\n"
        macfile_string += "/applicator/yRotation 0 deg\n"
        macfile_string += "/applicator/zRotation 0 deg\n"
        # add in the done flag
        macfile_string += "/applicator/done\n"

        with open(pth_output, "w") as mac_file:
            mac_file.write(macfile_string)

    def to_stl(self, pth_output: str) -> None:
        r"""
        Purpose:
            - To save the applicator geometry to an stl file.
        Inputs:
            - pth_output:str := path to the output stl file.
        Outputs:
            - Void := will save the applicator geometry to an stl file.
        """
        self._update_applicator_mesh_from_brachy_applicator()
        # write the polydata to an stl file
        stl_writer = vtkSTLWriter()
        stl_writer.SetFileName(pth_output)
        stl_writer.SetInputData(self.applicator_mesh)
        stl_writer.Write()


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
                    dwell_dict.get("position")[0],
                    dwell_dict.get("position")[1],
                    dwell_dict.get("position")[2]
                ]
            )
            relativePos = dwell_dict.get("relativePos")
            rotation = np.array(
                [
                    dwell_dict.get("rotation")[0],
                    dwell_dict.get("rotation")[1],
                    dwell_dict.get("rotation")[2]
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
            ), "The input json file does not exist."
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
                        "position": control_point["position"],
                        "relativePos": int(control_point["relativePos"]),
                        "rotation": control_point["rotation"],
                        "time": dwell_time,
                        "weight": dwell_weight,
                    }
                )
                catheter["dwells"] = dwells
            final_catheter_table.append(Catheter(catheter_dict=catheter))
        return final_catheter_table

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
