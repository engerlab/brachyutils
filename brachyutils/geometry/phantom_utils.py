import warnings
import os
import numpy as np
from glob import glob
from typing import Dict, List, Literal, Optional, Union, Tuple, Sequence
from collections import defaultdict
import numpy as np
import SimpleITK as sitk
from SimpleITK import Image, GetArrayFromImage 
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

from vtk import vtkPolyData, vtkDelaunay2D, vtkPoints, vtkDecimatePro, vtkPolygon
from vtkmodules.vtkIOGeometry import vtkSTLWriter

import nrrd
import pydicom

from opentps.core.data.images import CTImage, MRImage, ROIMask, Image3D
from opentps.core.data import ROIContour, RTStruct
from opentps.core.processing.imageProcessing.resampler3D import resampleImage3D, resampleImage3DOnImage3D
from opentps.core.processing.imageProcessing.sitkImageProcessing import imageToSITK
from opentps.core.io.dicomIO import (  # writeRTDose,
    readDicomCT,
    readDicomMRI,
    readDicomStruct,
    readDicomPET,
    writeDicomCT,
    writeRTStruct,
)

import json
from pathlib import Path

# Module-level cache for slicer colors
_SLICER_COLORS_CACHE = None

def _get_slicer_colors():
    """Get slicer colors with lazy loading and caching."""
    global _SLICER_COLORS_CACHE
    if _SLICER_COLORS_CACHE is None:
        with open(
            Path(__file__).parent.parent.parent / "admin/constants/slicer_colors.json", "r"
        ) as json_file:
            _SLICER_COLORS_CACHE = json.load(json_file)
    return _SLICER_COLORS_CACHE

class BrachyPhantom:
    r"""
    ### Puprose:
    - A class to load any voxelized geometry related to an HDR brachytherapy patient or phantom
        and perform some operations.
    ### Attributes:
    - pth_image: Path := the path of the geometry source file or files.
    - image_obj: CTImage or MRImage := the image of the patient loaded by openTPS. [x, y, z]
    - image_modality: Literal["CT", "MR", "US", "PET"] := the modality of the image.
    - structure_set: RTStruct := the structure set of the patient loaded by openTPS. [x, y, z].
    Other names for structure are contours, masks, segmentations.
    - cached_structure_masks: Dict[str, np.ndarray] := a cache for the structure masks.
    - structure_names: List[str] := the names of the structures in the dicom file.
    - unit_length: Literal["mm"] := the unit of length in the dicom file. default is mm.
    - xyz_format: bool := the format of the image. if True, the image is in [z, y, x] format.
    - orientation: Literal["LAS", "RAS", "LPS"] := the orientation of the image. default is LPS, same as 
    DICOM and slicer.
    ### Dependencies:
    - openTPS.core
    """

    def __init__(
        self,
        dir_dicom: Optional[Path] = None,
        pth_phantom_file: Optional[Path] = None,
        pth_structures_file: Optional[Path] = None,
        pth_egsphant_file: Optional[Path] = None,
        image_obj: Optional[Image3D] = None,
        structure_set: Optional[
            RTStruct | Dict[str, Union[ROIMask, np.ndarray, ROIContour]]
            ] = None,
    ) -> None:
        r"""
        ### Purpose:
        - Initialize the BrachyPhantom class based on the input path. The input path can be either
        the directory of the DICOM files or the path of the phantom file (in .nrrd). The structures file
        is optional. It is also possible to load the structures only without a phantom file. in that case,
        an empty image_obj is created with the dimensions matching the structures file.
        ### Inputs:
        - dir_dicom: Path := the directory of the DICOM files.
        - pth_phantom_file: Path := the path of the phantom .nrrd file.
        - pth_structures_file: Path := the path of the structure file.
        - pth_egsphant_file: Path := the path of the Egsphant file to be loaded.
        note that it is possible to generate an Egsphant from BrachyPhantom object.
        ### Outputs:
        - None
        ### Dependencies:
        - openTPS.core
        - BrachyEgsphant
        """
        if sum(x is not None for x in [dir_dicom, pth_phantom_file, image_obj]) > 1:
            raise ValueError(
                "Please provide only one geometry source file. \
Please provide either the directory of the DICOM files, \
the path of the phantom file, or the path of the Egsphant file."
            )
        if sum(x is not None for x in [structure_set, pth_structures_file]) > 1:
            raise ValueError(
                "Please provide only one structure source file. \
Please provide either the structure_set or the path of the structure file."
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
        self.structure_set = RTStruct()
        self.structure_names: List[str] = []
        # Used to avoid creating masks from contour multiple times, optional.
        # User needs to manually create the cache if they want to reuse it as a class attribute.
        self.cached_structure_masks: Dict[str, ROIMask] = defaultdict(ROIMask)
        self.unit_length: Literal["mm"] = "mm"
        self.xyz_format: bool = True
        self.anatomical_coordinate_system: Literal["LAS", "RAS", "LPS"] = "LPS"
        # Attributes for Egsphant files
        from brachyutils.geometry.egsphant_utils import BrachyEgsphant

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
        elif image_obj is not None:
            self.image_obj = image_obj

        if pth_structures_file is not None:
            pth_structures_file = Path(pth_structures_file)
            assert os.path.exists(pth_structures_file), "The input path does not exist."
            self._load_structure_file(pth_structures_file)
        elif structure_set is not None:
            if isinstance(structure_set, RTStruct):
                self.structure_set = structure_set
            else:
                self.set_structure_set(structure_set)

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
        image_files = [file for file in glob((str(pth_image) + "/*.[Dd][Cc][Mm]"))
                       if not os.path.basename(file).startswith(("R", "r"))]

        if len(image_files) == 0:
            raise ValueError("No DICOM files found in the input directory.")
        if "CT" in str(Path(image_files[0]).stem).upper():
            ct_files = list(filter(lambda s: "CT" in s.upper(), image_files))
            self.image_obj = readDicomCT(ct_files)
            self.image_modality = "CT"
            # get the orientation of the image
            header = pydicom.dcmread(ct_files[0])
            orientation = header.get((0x0010, 0x2210), "LPS")
            if orientation == "BIPED":
                orientation = "LPS"
            self.anatomical_coordinate_system = orientation if orientation is not None else "LPS"
        
        elif "MR" in  str(Path(image_files[0]).stem).upper():
            mr_files = list(filter(lambda s: "MR" in s.upper(), image_files))
            self.image_obj = readDicomMRI(mr_files)
            self.image_modality = "MR"
            header = pydicom.dcmread(mr_files[0])
            orientation = header.get((0x0010, 0x2210), "LPS")
            if orientation == "BIPED":
                orientation = "LPS"
            self.anatomical_coordinate_system = orientation if orientation is not None else "LPS"

        elif "US" in str(Path(image_files[0]).stem).upper():
            us_files = list(filter(lambda s: "US" in s.upper(), image_files))
            self.image_obj = readDicomUS(us_files)
            self.image_modality = "US"
            header = pydicom.dcmread(us_files[0])
            orientation = header.get((0x0010, 0x2210), "LPS")
            if orientation == "BIPED":
                orientation = "LPS"
            self.anatomical_coordinate_system = orientation if orientation is not None else "LPS"
        elif "PT" in str(Path(image_files[0]).stem).upper():
            pet_files = list(filter(lambda s: "PT" in s.upper(), image_files))
            self.image_obj = readDicomPET(pet_files)
            self.image_modality = "PET"
            header = pydicom.dcmread(pet_files[0])
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
        self.image_obj.to_lps(current_orientation=orientation)
        self.anatomical_coordinate_system = "LPS"
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
        origin = image_nifti.affine[:3, 3]
        spacing = image_nifti.header.get("pixdim")[1:4]
        origin = origin * [
            np.sign(image_nifti.affine[0][0]),
            np.sign(image_nifti.affine[1][1]),
            np.sign(image_nifti.affine[2][2]),
        ]
        if image_data.ndim == 4:
            image_data = image_data[:, :, :, 0]
        if image_nifti.header.data_layout == "F":
            image_data = np.swapaxes(image_data, 0, 2)
        # if image_nifti.header.default_x_flip:
            # image_data = np.flip(image_data, axis=2)


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

        self.image_obj = Image3D(
            origin=origin,
            spacing=spacing,
        )
        self.set_image_array(image_data)
        self.image_obj.to_lps(current_orientation=orientation)
        self.anatomical_coordinate_system = "LPS"

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
            structure_set = readDicomStruct(pth_structure)
            structure_mask_dict = {contour.name:contour for contour in structure_set}
            header = pydicom.dcmread(pth_structure)
            structure_orientation = header.get((0x0010, 0x2210), "LPS")
            if structure_orientation == "BIPED":
                structure_orientation = "LPS"
        elif str(pth_structure).endswith(".nrrd"):
            structure_mask_dict, structure_orientation = readNrrdStruct(pth_structure)
        elif str(pth_structure).endswith(".nii.gz"):
            structure_mask_dict, structure_orientation = readNiftiStruct(pth_structure)
        else:
            raise ValueError("The structure file type is not recognized.")

        if self.anatomical_coordinate_system is None:
            self.anatomical_coordinate_system = structure_orientation
        else:
            assert (self.anatomical_coordinate_system == structure_orientation), \
                "The orientation of the structure file is not the same as the image file."

        self.set_structure_set(structure_mask_dict)
    
    def get_structure_mask(
        self,
        query_structure_list: List[str],
        mask_type: Union[
            np.ndarray, ROIContour, ROIMask,
            Literal["array", "contour", "mask"]] = ROIMask,
        strict_name_match: bool = True,
    ) -> Dict[str, Union[np.ndarray, ROIContour, ROIMask]]:
        r"""
        ### Purpose:
        - To return a dictionary with the requested structure masks from BrachyPhantom object.
        When looking for a structure, self.cached_structure_mask is prioretized over self.structure_set
        to avoid unnecessary contour to mask conversion.

        ### Inputs:
        - query_structure_list := list of structure names to find the mask of.
        - mask_type: Union[np.ndarray, ROIContour, ROIMask] := the type of the mask to return.
            if np.ndarray (or str "array"), the mask will be returned as a numpy array in [z, y, x] format.
            if ROIContour (or str "contour"), the mask will be returned as a ROIContour object in [x, y, z] format.
            if ROIMask (or str "mask"), the mask will be returned as a ROIMask object in [x, y, z] format.
        - strict_name_match: if True, the queried structure names must match exactly the structure_names.
        if False, The queried structure string should be a subset of the structure string in the dicom file. For example,
        if the structure string in dicom file is CTV_BRACHY, then the query string can be CTV or ctv.

        ### Outputs:
        - mask_dict:dict :=  a dictionary with the queried structure name as key and the mask as value.
        """
        assert (
            self.structure_set is not None or self.cached_structure_masks is not None
        ), "structure masks have not been loaded yet. please run load_structure_file() first"
        mask_dict: dict = {}
        flattened_query_structure_list = []

        for query_structure in query_structure_list:
            if isinstance(query_structure, list):
                flattened_query_structure_list.extend(query_structure)
            else:
                flattened_query_structure_list.append(query_structure)

        for query_structure in flattened_query_structure_list:
            for mask_name in self.structure_names:
                if strict_name_match:
                    pick_structure = query_structure.lower() == mask_name.lower()
                else:
                    pick_structure = query_structure.lower() in mask_name.lower()

                if pick_structure:
                    if self.cached_structure_masks is not None:
                        mask = self.cached_structure_masks.get(mask_name, None)
                    else:
                        mask = self.structure_set.getContourByName(mask_name).getBinaryMask(
                            origin=self.image_obj.origin,
                            gridSize=self.image_obj.gridSize,
                            spacing=self.image_obj.spacing,
                        )

                    if not np.any(mask.imageArray):
                        warnings.warn(
                            f"mask for {query_structure} is all zeros",
                            stacklevel=2
                        )
                        mask.imageArray = np.zeros(self.image_obj.gridSize)
                        mask.origin = self.image_obj.origin
                        mask.spacing = self.image_obj.spacing
                        mask.gridSize = self.image_obj.gridSize
                    if mask_type == np.ndarray or mask_type == "array":
                        mask_dict[query_structure] = np.swapaxes(
                            mask.imageArray, 0, 2
                        )
                    elif mask_type == ROIContour or mask_type == "contour":
                        mask_dict[query_structure] = (
                            self.structure_set.getContourByName(mask_name)
                        )
                    elif mask_type == ROIMask or mask_type == "mask":
                        mask_dict[query_structure] = mask
                    else:
                        raise ValueError(f"mask_type {mask_type} not recognized")
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
            f"Structure Names: {self.structure_names}"
            if self.structure_names is not None
            else "No structure names."
        )
        print(
            f"Structure Count: {len(self.structure_names)}"
            if self.structure_names is not None
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
        self.structure_names = []

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
            for structure_name in self.structure_names:
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
                writeDicomCT(self.image_obj, str(dir_output))
            elif self.image_modality == "MR":
                warnings.warn("MR image writing is not implemented yet. writing volume as CT")
                writeDicomCT(self.image_obj, str(dir_output))
            elif self.image_modality == "US":
                warnings.warn("MR image writing is not implemented yet. writing volume as CT")
                writeDicomCT(self.image_obj, str(dir_output))
            else:
                raise ValueError("Image modality not recognized")

    def write_structures_to_dicom(self, dir_output: Path) -> None:
        r"""
        Purpose:
            - To write the structures to a dicom file.
        """
        if self.structure_set is not None and len(self.structure_set.contours) > 0:
            os.makedirs(dir_output, exist_ok=True)
            writeRTStruct(self.structure_set, str(dir_output))

    def write_image_to_nrrd(
        self,
        pth_output: Path | str,
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
        Outputs:
            - None
        Dependencies:
            - pynrrd
        """
        if isinstance(pth_output, str):
            pth_output = Path(pth_output)
        assert (
            os.path.splitext(pth_output)[-1] == ".nrrd"
        ), "the file should have '.nrrd' extension"
        imageToNrrd(
            image_obj=self.image_obj,
            pth_output=Path(pth_output),
            # anatomical_coordinate_system=self.anatomical_coordinate_system,
            modality=self.image_modality,
            metadata=metadata,
        )

    def write_structures_to_nrrd(
        self,
        pth_output: Path | str,
        overlap: Optional[bool] = True,
        representation: Literal["contour", "mask"] = "mask",
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
        if isinstance(pth_output, str):
            pth_output = Path(pth_output)
        if representation == "mask":
            structure_mask_dict: Dict[str, ROIMask] = self.get_structure_mask(
                self.structure_names, mask_type=ROIMask
            )
            #copy over the colors from the structure set
            for structure_name in structure_mask_dict.keys():
                contour = self.structure_set.getContourByName(structure_name)
                if contour is not None:
                    structure_mask_dict[structure_name]._displayColor = contour._displayColor

            masksToNrrd(
                structure_mask_dict=structure_mask_dict,
                pth_output=pth_output,
                overlap=overlap,
                metadata=metadata,
            )
        elif representation == "contour":
            raise NotImplementedError(
                "Writing structures to nrrd in contour representation is not implemented yet."
            )
        else:
            raise ValueError(
                f"Format {representation} not recognized. Please use 'mask' or 'contour'."
            )

    def write_to_egsphant(
        self,
        pth_output: Path,
        material_dict: dict | Path = None,
        assign_material_from_ct: bool = None,
        crop_by_contour: str | List[str] = None,
        marginInMM: float | List[float] = 0.0,
        resampled_spacing: List[float] = None,
        resampled_origin: List[float] = None,
        resample_phantom_base: Optional[bool] = True,
        background_material: Optional[str] = "Air",
        strict_name_match: bool = True,
    ) -> None:
        r"""
        ### Purpose:
        - Write the BrachyPhantom object to an Egsphant file.
        ### Inputs:
        - pth_output: Path := the path to write the Egsphant file to.
        - material_dict: dict | Path := the dictionary of the materials. if Path, the path to the material file.
        The dictionary contains the name of the elements for each voxel,
        and the following keys: [
            "density" := the density of the material in g/cm^3,
            "HU_limit" := the lower HU limit threshold of the material,
            "structure_name := {optional} the name of the structure in the dicom file that represents the material,"
        ]
        - assign_material_from_ct: bool := if True, the material will be assigned from the CT image.
        - crop_by_contour: str | List[str] := the name of the structure in the dicom file to crop the phantom by.
        If a list of names is provided, the union of the structures will be used to crop the phantom.
        - marginInMM: float := the margin in mm to add to the cropped phantom. default is 0.
        - resampled_spacing: List[float] := the spacing to resample the egsphant to.
        - background_material: Optional[str] := the name of the background material. default is "Air".
        """
        pth_output = Path(pth_output)
        if not str(pth_output).endswith(".egsphant") and not str(pth_output).endswith(".seq.nrrd"):
            raise ValueError("The output file should have '.egsphant' or '.seq.nrrd' extension.")
        #if the egsphant is already made, write it
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        if self.egsphant_obj is not None:
            if str(pth_output).endswith(".egsphant"):
                self.egsphant_obj.write_to_ctegsphant(pth_output)
            elif str(pth_output).endswith(".seq.nrrd"):
                self.egsphant_obj.write_to_nrrd(pth_output)
#prepare the phantom for egsphant conversion
        elif self.image_obj is not None:
            phantom_used_for_egsphant = deepcopy(self)
            from brachyutils.geometry.egsphant_utils import BrachyEgsphant
            
            if resampled_spacing is not None or resampled_origin is not None: #if we want to resample
                if resample_phantom_base: #resample the phantom and structures that the egsphant is based on
                    phantom_used_for_egsphant.resample_to(
                        origin=resampled_origin,
                        spacing=resampled_spacing,
                        inplace=True
                    )

            self.egsphant_obj = BrachyEgsphant(
                phantom=phantom_used_for_egsphant,
                material_dict=material_dict,
                assign_material_from_ct=assign_material_from_ct,
                background_material=background_material
            )
  
            if crop_by_contour is not None:
                self.egsphant_obj.crop_by_contour(
                    phantom_used_for_egsphant,
                    crop_by_contour,
                    strict_name_match=strict_name_match,
                    marginInMM=marginInMM)

            if resampled_spacing is not None and not resample_phantom_base:
                self.egsphant_obj.material_image = resampleImage3D(
                    image=self.egsphant_obj.material_image,
                    origin=resampled_origin,
                    spacing=resampled_spacing, 
                    outputType=np.int16)
                self.egsphant_obj.density_image = resampleImage3D(
                    image=self.egsphant_obj.density_image,
                    origin=resampled_origin,
                    spacing=resampled_spacing,)
                    # sitk_interpolator=sitk.sitkNearestNeighbor)
                self.egsphant_obj.get_voxel_edges()
            if str(pth_output).endswith(".egsphant"):
                self.egsphant_obj.write_to_ctegsphant(pth_output)
            elif str(pth_output).endswith(".seq.nrrd"):
                self.egsphant_obj.write_to_nrrd(pth_output)
        else:
            raise ValueError(
                "No image object or egsphant object to write to Egsphant file. Please load the image object first."
            )

    def export_to(
        self,
        pth_image_out: Path | str = None,
        pth_structures_out: Path | str = None,
        dir_dicom_out: Path | str = None,
        dir_nrrd_out: Path | str = None
        ):
        r"""
        Purpose:
            - To export the image and/or the structures to file. This function will call the appropriate
            export function depending on the extension of the given path. If you would like to export
            to egsphant, please use write_to_egsphant() function.
        Inputs:
            pth_image_out:= path to the output image file. If the extension could be .nrrd. To export images
            to dicom, use dir_dicom_out.
            pth_structures_out:= path to the output structure file, the extension could be .nrrd or .dcm
            dir_dicom_out:= path to export all the dicom informatin to. it has to be a directory
        """
        if pth_image_out is not None:
            pth_image_out = Path(pth_image_out)
            assert self.image_obj is not None, "no image is loaded into this BrachyPhantom"
            if str(pth_image_out).endswith(".nrrd"):
                self.write_image_to_nrrd(pth_output=pth_image_out)

        if pth_structures_out is not None:
            pth_structures_out = Path(pth_structures_out)
            assert self.structure_set is not None, "no structures is loaded into this BrachyPhantom"
            self.write_structures_to_nrrd(pth_output=pth_structures_out)

        if dir_dicom_out is not None:
            dir_dicom_out = Path(dir_dicom_out)
            os.makedirs(dir_dicom_out, exist_ok=True)
            assert dir_dicom_out.is_dir(), f"the provided path {dir_dicom_out} is not a directory"
            if self.image_obj is not None:
                self.write_image_to_dicom(dir_output=dir_dicom_out)
            if self.structure_set is not None:
                self.write_structures_to_dicom(dir_output=dir_dicom_out)

        if dir_nrrd_out is not None:
            dir_nrrd_out = Path(dir_nrrd_out)
            os.makedirs(dir_nrrd_out, exist_ok=True)
            assert dir_nrrd_out.is_dir(), f"the provided path {dir_nrrd_out} is not a directory"
            if self.image_obj is not None:
                self.write_image_to_nrrd(
                    pth_output=Path.joinpath(dir_nrrd_out, str(self.pth_image.name).split(".")[0]+".nrrd")
                    )
            if self.structure_set is not None and len(self.structure_set.contours) > 0:
                self.write_structures_to_nrrd(
                    pth_output=Path.joinpath(dir_nrrd_out, str(self.pth_image.name).split(".")[0]+".seg.nrrd")
                )

    def crop_by_coordinates(
        self,
        coordinate_range: List[float] | np.array,
        inplace: "BrachyPhantom" = True,
        marginInMM: float | List[float] = 0.0,
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

        coordinate_range = np.array(coordinate_range)
        assert coordinate_range.shape == (
            3,
            2,
        ), "coordinate_range should be a 3x2 array in x, y, z order"
        if isinstance(marginInMM, float):
            marginInMM = [marginInMM]*3
        if inplace:
            crop3DDataAroundBox(self.image_obj, coordinate_range, marginInMM=marginInMM)
        else:
            new_phantom: BrachyPhantom = copy.deepcopy(self)
            new_phantom.crop_by_coordinates(coordinate_range, inplace=True, marginInMM=marginInMM)
            return new_phantom

    def crop_by_index(
        self,
        index_range: List[int] | np.array,
        inplace: Optional[bool] = True,
        marginInMM: float | List[float] = 0.0,
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
        assert index_range.shape == (
            3,
            2,
        ), "index_range should be a 3x2 array in x, y, z order"
        new_origin_coords = self.image_obj.getPositionFromVoxelIndex(
            index_range[:, 0]
        )
        new_ending_coords = self.image_obj.getPositionFromVoxelIndex(
            index_range[:, 1]
        )
        new_coords_range = np.column_stack([new_origin_coords, new_ending_coords])
        return self.crop_by_coordinates(new_coords_range, inplace, marginInMM=marginInMM)

    def crop_by_contour(
        self,
        contour_name: str | List[str],
        inplace: bool = True,
        strict_name_match: bool = True,
        marginInMM: float | List[float] = 0.0,
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
        if isinstance(contour_name, str):
            contour_name = [contour_name]
        if isinstance(marginInMM, float):
            marginInMM = [marginInMM, marginInMM, marginInMM]
        mask_dict = self.get_structure_mask(
            contour_name,
            mask_type=ROIMask,
            strict_name_match=strict_name_match)
        combined_mask_array = np.zeros_like(
            mask_dict[contour_name[0]].imageArray, dtype=bool
        )
        for name in contour_name:
            combined_mask_array = np.logical_or(combined_mask_array, mask_dict[name].imageArray)
        combined_mask = ROIMask(
            imageArray=combined_mask_array,
            origin=self.image_obj.origin,
            spacing=self.image_obj.spacing,
        )
        resampled_mask = resampleImage3DOnImage3D(
            combined_mask, self.image_obj
        )
        box_around_mask = np.array(getBoxAroundROI(resampled_mask))
        return self.crop_by_coordinates(box_around_mask, inplace, marginInMM)

    def cache_structure_set_as_masks(
        self,
        interpolator_contours=sitk.sitkNearestNeighbor,
        pth_structures_file: str | Path = None,
        mask_colors: Dict[str, Sequence[int]] | Sequence[int] = None,
    ) -> None:
        r"""
        Purpose:
            - Cache the structure set as masks. This will resample the masks to the image object.
            Function used in case you query multiple time the masks from the BrachyPhantom object
            and you want to avoid resampling the masks each time.
        Inputs:
            - interpolator_contours: sitk.InterpolatorEnum := the interpolator to use for resampling the contours.
        Outputs:
            - None
        """
        from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D

        if pth_structures_file is not None:
            assert pth_structures_file.endswith(".nrrd"), "the structure file should be a nrrd file"
            structure_dict, _ = readNrrdStruct(pth_structures_file)
        else:
            structure_dict = self.get_structure_mask(
                self.structure_names,
                mask_type=ROIMask,
                )
        slicer_colors = _get_slicer_colors()
        # set the default color dictionary if not provided
        if mask_colors is None:
            mask_colors = {
                k: slicer_colors[i+1]["color"] for i, k in enumerate(structure_dict.keys())
            }
        elif isinstance(mask_colors, Sequence):
            mask_colors = {
                k: mask_colors for k in structure_dict.keys()
            }

        new_structure_dict = {}
        for struc in structure_dict.keys():
            mask = structure_dict[struc]
            old_color = mask._displayColor
            structure_color = mask_colors.get(struc)
            if not(old_color is None):
                new_color = old_color
            else:
                new_color = structure_color
            if not np.array_equal(mask.spacing, self.image_obj.spacing) or \
                not np.array_equal(mask.origin, self.image_obj.origin) or \
                not np.array_equal(mask.gridSize, self.image_obj.gridSize):
                new_structure_dict[struc] = resampleImage3DOnImage3D(
                    mask,
                    self.image_obj,
                    sitk_interpolator=interpolator_contours
                    )
            else:
                new_structure_dict[struc] = mask
            new_structure_dict[struc]._displayColor = new_color
        # Store the resampled masks in the cached_structure_masks attribute
        self.cached_structure_masks = new_structure_dict

    def set_structure_set(
        self,
        mask_dict: Dict[str, Union[ROIMask, ROIContour, np.ndarray, sitk.Image]],
        mask_colors: Dict[str, Tuple[int, int, int]] | Tuple[int, int, int] = None,
        ) -> None:
        r"""
        ### Purpose:
        - Set the structure set with the input mask dictionary mapping structure names to ROIMask.
        If the name of a structure is in the structure set, the mask will be replaced.
        If the name of a structure is not in the structure set, a new structure will be added.
        The mask will be resampled to the image object if it exists.

        ### Inputs:
        - mask_dict: dict := the dictionary of the masks.
        - mask_colors: dict | tuple := the dictionary of the colors for each structure. If a tuple is provided,
        the same color will be used for all structures. If None is provided, the default colors will be used based
        on the slicer color table https://www.slicer.org/wiki/Slicer3:2010_GenericAnatomyColors.
        The values could be numpy arrays, ROIContour or ROIMask objects.

        ### Outputs:
        - None
        """
        from opentps.core.processing.imageProcessing.resampler3D import (
            resampleImage3DOnImage3D,
        )
        from opentps.core.processing.segmentation.segmentation3D import getBoxAroundROI
        slicer_colors = _get_slicer_colors()
        # set the default color dictionary if not provided
        if mask_colors is None:
            mask_colors = {
                k: slicer_colors[i+1]["color"] for i, k in enumerate(mask_dict.keys())
            }
        elif isinstance(mask_colors, Sequence):
            mask_colors = {
                k: mask_colors for k in mask_dict.keys()
            }

        new_cached_mask = defaultdict(ROIMask)
        for structure_name in mask_dict:
            structure_color = mask_colors.get(structure_name)
            # check if the structure already exists in structure set, remove it if yes.
            # but inherit the color!
            old_structure = list(filter(lambda x: x == structure_name, self.structure_names))
            if len(old_structure) == 0:
                pass
            else:
                self.structure_set.removeContour(old_structure)
                structure_color = old_structure.color
            # check if the old structure was also cached, remove it if yes.
            old_cached_structure = self.cached_structure_masks.get(structure_name, None)
            if old_cached_structure is not None:
                del self.cached_structure_masks[structure_name]

            if mask_dict.get(structure_name) is None:
                # skip planning/optimization structures, i.e. hot spot volumes.
                continue
            if isinstance(mask_dict.get(structure_name), np.ndarray):
                mask = ROIMask(
                    name=structure_name,
                    imageArray=np.swapaxes(mask_dict[structure_name], 0, 2),
                    origin=self.image_obj.origin,
                    spacing=self.image_obj.spacing,
                )
            elif isinstance(mask_dict.get(structure_name), ROIContour):
                mask = mask_dict.get(structure_name).getBinaryMask(
                    origin=self.image_obj.origin,
                    spacing=self.image_obj.spacing,
                    gridSize=self.image_obj.gridSize,
                )
                mask.name = structure_name

            elif isinstance(mask_dict.get(structure_name), ROIMask):
                mask = mask_dict.get(structure_name)
                mask.name = structure_name
                if self.image_obj is not None:
                    # Check if the spacings, the shape and origin already match or not
                    if not np.array_equal(mask.spacing, self.image_obj.spacing) or \
                        not np.array_equal(mask.origin, self.image_obj.origin) or \
                        not np.array_equal(mask.gridSize, self.image_obj.gridSize):
                        # Resample the mask to the image object
                        mask = resampleImage3DOnImage3D(mask, self.image_obj)   
            elif isinstance(mask_dict.get(structure_name), sitk.Image):
                mask = ROIMask(
                    name=structure_name,
                    imageArray=sitk.GetArrayFromImage(mask_dict[structure_name]),
                    origin=mask_dict[structure_name].GetOrigin(),
                    spacing=mask_dict[structure_name].GetSpacing(),
                )
            else:
                raise ValueError("The mask type is not recognized.")
 
            # if mask hits the boundary of the image, set the boundary to 0.
            tight_box_coordinates = np.round(getBoxAroundROI(mask), decimals=2)
            mask_edges = np.array(
                [
                    mask.getPositionFromVoxelIndex([0, 0, 0]),
                    mask.getPositionFromVoxelIndex(mask.gridSize-1)
                    ]
                )
            mask_edges = np.round(mask_edges, decimals=2)
            mask_edges = np.reshape(mask_edges.T, (3, 2))
            touching_edge = (tight_box_coordinates == mask_edges).flatten()
            for i, edge in enumerate(touching_edge):
                if edge:
                    if i == 0:
                        mask.imageArray[0, :, :] = 0
                    elif i == 1:
                        mask.imageArray[-1, :, :] = 0
                    elif i == 2:
                        mask.imageArray[:, 0, :] = 0
                    elif i == 3:
                        mask.imageArray[:, -1, :] = 0
                    # hitting the ends of the z axis is not problematic
                    # elif i == 4:
                    #     mask.imageArray[:, :, 0] = 0
                    # elif i == 5:
                        # mask.imageArray[:, :, -1] = 0

            mask._displayColor = structure_color
            new_cached_mask[mask.name] = mask
            self.structure_set.appendContour(mask.getROIContour())

        self.structure_set.setPatient(
                self.image_obj.patient if self.image_obj is not None else None
            )

        self._update_structure_names()
        self.cached_structure_masks = new_cached_mask

    def _update_structure_names(self) -> None:
        r"""
        Purpose:
            - Update the structure names based ont he structure set.
        Inputs:
            - None
        Outputs:
            - None
        """
        self.structure_names = [structure.name for structure in self.structure_set.contours]

    def rename_structures(self, structure_name_dict: dict) -> None:
        r"""
        Purpose:
            - First purpose is to rename the structures in the structure set.
            - Additional purpose is to remove structures if their new name is 'REMOVE'.
        
        Inputs:
            - structure_name_dict: dict := the dictionary of the structure names to rename.
            The keys are the old names and the values are the new names.
        
        Outputs:
            - None
        """
        assert len(self.structure_set) > 0 or len(self.cached_structure_masks) > 0, (
            "No structures to rename. Please load the structures first."
        )
        if len(self.cached_structure_masks) > 0:
            new_cached_structure_masks = {}
            for old_name, new_name in structure_name_dict.items():
                if old_name in self.cached_structure_masks:
                    if new_name == "REMOVE":
                        continue
                    new_cached_structure_masks[new_name] = self.cached_structure_masks[old_name]
                else:
                    warnings.warn(f"The structure {old_name} does not exist in the cached masks.")
            self.cached_structure_masks = new_cached_structure_masks
            
        if len(self.structure_set) > 0:
            for old_name, new_name in structure_name_dict.items():
                if new_name == "REMOVE":
                    self.remove_structure(old_name)
                    continue
                structure = self.structure_set.getContourByName(old_name)
                if structure is not None:
                    structure.name = new_name
                else:
                    warnings.warn(f"The structure {old_name} does not exist.")
            self._update_structure_names()
    
    def remove_structure(self, structure_name: str) -> None:
        r"""
        Purpose:
            - Remove the structure from the structure set.
        Inputs:
            - structure_name: str := the name of the structure to remove.
        Outputs:
            - None
        """
        structure = self.structure_set.getContourByName(structure_name)
        if structure is not None:
            self.structure_set.removeContour(structure)
            self._update_structure_names()
        else:
            warnings.warn(f"The structure {structure_name} does not exist.")
        cached_structure = self.cached_structure_masks.get(structure_name, None)
        if cached_structure is not None:
            del self.cached_structure_masks[structure_name]

    def resample_to(
        self,
        origin:np.array=None,
        spacing:np.array=None,
        inplace:bool=True,
        gridSize:np.array=None,
        interpolator_img=sitk.sitkLinear, 
        interpolator_contours=sitk.sitkLinear) -> "BrachyPhantom":
        r"""
        ### Purpose:
            - resample the phantom to a new origin and spacing.
        
        ### Inputs:
            - origin:np.array := the new origin of the image.
            - spacing:np.array := the new spacing of the image.
            - inplace:bool := if True, the resampling will be done in place.
        
        ### Outputs:
            - BrachyPhantom := the resampled phantom object if the inplace is False
        """
        from opentps.core.processing.imageProcessing.resampler3D import resampleImage3D
        new_phantom = phantom_with_empty_image_like(self, new_pth_image=self.pth_image)
        new_img_obj = resampleImage3D(
            self.image_obj,
            origin=origin,
            spacing=spacing,
            gridSize=gridSize,
            sitk_interpolator=interpolator_img
            )

        if self.cached_structure_masks is not None and len(self.cached_structure_masks) > 0:
            new_cached_structure_masks = {}
            for structure_name, mask in self.cached_structure_masks.items():
                old_color = mask._displayColor
                new_cached_structure_masks[structure_name] = resampleImage3DOnImage3D(
                    mask,
                    new_img_obj,
                    sitk_interpolator=interpolator_contours
                    )
                new_cached_structure_masks[structure_name]._displayColor = old_color
            self.cached_structure_masks = new_cached_structure_masks

        if inplace:
            self.image_obj = new_img_obj
        else:
            new_phantom.image_obj = new_img_obj
            return new_phantom

    def sort_structures_by_name(self, sorted_names):
        r"""
        Purpose:
            - Sort the structures in the structure set by the input list of names.
        Inputs:
            - sorted_names: list := the list of names to sort the structures by.
        Outputs:
            - None
        """
        self.structure_set._contours = sorted(
            self.structure_set._contours,
            key=lambda x: sorted_names.index(x.name)
            if x.name in sorted_names
            else len(sorted_names)
        )
        self._update_structure_names()

    def get_structures_volume(self, structure_names:List[str]) -> Dict[str, float]:
        r"""
        Purpose:
            - Get the volume of each structure that is requested.
        Inputs:
            - structure_names: list := the list of names of the structures to get the volumes for.
        Outputs:
            - volume_dict: dict := the dictionary of the volumes of each structure in cm^3.
        """
        assert self.image_obj is not None, "No image object to get the volume from."
        assert self.structure_set is not None, "No structure set to get the volume from."
        mask_dict = self.get_structure_mask(structure_names, mask_type=ROIMask)
        volume_dict = {}
        for name, mask in mask_dict.items():
            assert mask is not None, f"No mask found for structure {name}."
            volume_dict[name] = mask.getVolume()/1000 # convert to cm^3
        return volume_dict

# helper functions
def phantom_with_empty_image_like(
    phantom: BrachyPhantom,
    new_pth_image: Path | str=None
    ) -> BrachyPhantom:
    r"""
    ### Purpose:
        - Create a new BrachyPhantom object with the same structure set as the input phantom but with an empty image.
    
    ### Inputs:
        - phantom: BrachyPhantom := the input phantom object.
        - new_pth_image := the new name for the empty phantom.
    
    ### Outputs:
        - new_phantom: BrachyPhantom := the new phantom object.
    """
    from copy import deepcopy
    old_phantom = deepcopy(phantom)
    new_phantom = BrachyPhantom()
    new_phantom.pth_image = Path(new_pth_image)
    new_phantom.image_obj = None
    new_phantom.image_modality = old_phantom.image_modality
    new_phantom.structure_set = old_phantom.structure_set
    new_phantom.structure_names = [structure.name for structure in new_phantom.structure_set.contours]
    new_phantom.unit_length = old_phantom.unit_length
    new_phantom.xyz_format = old_phantom.xyz_format

    return new_phantom

def get_uniform_phantom(
    voxel_value: float = 0.0,
    gridSize: List[int] = [100, 100, 100],
    spacing: List[float] = [1.0, 1.0, 1.0],
    origin: List[float] = [0.0, 0.0, 0.0],
    )-> BrachyPhantom:
    r"""
    ### Purpose:
    - Create a uniform cubic phantom object where all the voxels have the same value.

    ### Inputs:
    - voxel_value: float := the value of the voxels in the phantom.
    - gridSizeInMilimeters: List[int] := the size of the phantom in millimeters.
    - spacing: List[float] := the spacing of the phantom in millimeters.
    - origin: List[float] := the origin of the phantom in millimeters.

    ### Outputs:
    - phantom: BrachyPhantom := the new phantom object.
    """
    phantom = BrachyPhantom()
    phantom.image_obj = Image3D(
        imageArray=np.ones(gridSize) * voxel_value,
        spacing=spacing,
        origin=origin,
    )
    phantom.image_modality = None
    phantom.structure_set = None
    phantom.structure_names = []
    phantom.unit_length = "cm"
    phantom.xyz_format = "LPS"

    return phantom
    

def _sort_segmentation_dict_by_size(
    mask_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
    r"""
    Purpose:
        - will sort the items in a mask dictionary by the size of the segmentation.
    Inputs:
        - mask_dict: dict := the dictionary of the masks. the values are numpy arrays in
        [z, y, x] format.
    Outputs:
        - sorted_dict: dict := the sorted dictionary.
    """

    sorted_dict_list = sorted(
        mask_dict.items(),
        key=lambda x: np.sum(x[1]),
        reverse=True
    )
    return dict(sorted_dict_list)


def _convert_many_binary_masks_to_1_int_mask(
    mask_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
    r"""
    Purpose:
        - Convert many binary masks to one integer mask. The masks should be ordered
        from largest to smallest as the smallest mask will overwrite the larger mask.
        use _sort_segmentation_dict_by_size() to sort the masks.
    Inputs:
        - mask_dict: dict := the dictionary of the masks. the values are numpy arrays in
        [z, y, x] format.
    Outputs:
        - int_mask: np.ndarray := the integer mask.
    """
    int_mask = np.zeros_like(list(mask_dict.values())[0], dtype=int)
    for i, (_, mask) in enumerate(mask_dict.items()):
        int_mask[mask] = i + 1
    return int_mask

def readDicomUS(dcmFiles):
    r""""
    Generate a US image object from a list of dicom US slices.

    Parameters
    ----------
    dcmFiles: list
        List of paths for Dicom US slices to be imported.

    Returns
    -------
    image: mrImage object
        The function returns the imported US image
    """
    import logging
    # read dicom slices
    images = []
    sopInstanceUIDs = []
    sliceLocation = np.zeros(len(dcmFiles), dtype='float')
    firstdcm = dcmFiles[0]
    
    for i in range(len(dcmFiles)):
        dcm = pydicom.dcmread(dcmFiles[i])
        sliceLocation[i] = float(dcm.ImagePositionPatient[2])
        images.append(dcm.pixel_array)
        sopInstanceUIDs.append(dcm.SOPInstanceUID)

    # sort slices according to their location in order to reconstruct the 3d image
    sortIndex = np.argsort(sliceLocation)
    sliceLocation = sliceLocation[sortIndex]
    sopInstanceUIDs = [sopInstanceUIDs[n] for n in sortIndex]
    images = [images[n] for n in sortIndex]
    imageData = np.dstack(images).astype("float32").transpose(1, 0, 2)

    # verify reconstructed volume
    if imageData.shape[0:2] != (dcm.Columns, dcm.Rows):
        logging.warning("WARNING: GridSize " + str(imageData.shape[0:2]) + " different from Dicom Columns (" + str(
            dcm.Columns) + ") and Rows (" + str(dcm.Rows) + ")")

    # collect image information
    meanSliceDistance = (sliceLocation[-1] - sliceLocation[0]) / (len(images) - 1)
    if (hasattr(dcm, 'SliceThickness') and (
            type(dcm.SliceThickness) == int or type(dcm.SliceThickness) == float) and abs(
            meanSliceDistance - dcm.SliceThickness) > 0.001):
        logging.warning(
            "WARNING: Mean Slice Distance (" + str(meanSliceDistance) + ") is different from Slice Thickness (" + str(
                dcm.SliceThickness) + ")")

    if (hasattr(dcm, 'SeriesDescription') and dcm.SeriesDescription != ""):
        imgName = dcm.SeriesDescription
    else:
        imgName = dcm.SeriesInstanceUID

    pixelSpacing = (float(dcm.PixelSpacing[1]), float(dcm.PixelSpacing[0]), meanSliceDistance)
    imagePositionPatient = (float(dcm.ImagePositionPatient[0]), float(dcm.ImagePositionPatient[1]), sliceLocation[0])

    # collect patient information
    if hasattr(dcm, 'PatientID'):
        from opentps.core.io.dicomIO import Patient
        birth = dcm.PatientBirthDate if hasattr(dcm, 'PatientBirthDate') else ""
        sex = dcm.PatientSex if hasattr(dcm, 'PatientSex') else None

        patient = Patient(id=dcm.PatientID, name=str(dcm.PatientName), birthDate=birth, sex=sex)
    else:
        patient = Patient()

    # generate MR image object
    FrameOfReferenceUID = dcm.FrameOfReferenceUID if hasattr(dcm, 'FrameOfReferenceUID') else pydicom.uid.generate_uid()
        
    image = MRImage(imageArray=imageData, name=imgName, origin=imagePositionPatient,
                    spacing=pixelSpacing, seriesInstanceUID=dcm.SeriesInstanceUID,
                    frameOfReferenceUID=FrameOfReferenceUID, sliceLocation=sliceLocation,
                    sopInstanceUIDs=sopInstanceUIDs)
       
    image.patient = patient
    if hasattr(dcm, 'StudyDate'):
        image.studyDate = float(dcm.StudyDate)
    if hasattr(dcm, 'PatientPosition'):
        image.patientPosition = dcm.PatientPosition
    if hasattr(dcm, 'SeriesNumber'):
        image.seriesNumber = dcm.SeriesNumber
    image.studyInstanceUID = dcm.StudyInstanceUID if hasattr(dcm, 'StudyInstanceUID') else pydicom.uid.generate_uid()
    image.bitsAllocated = dcm.BitsAllocated if hasattr(dcm, 'BitsAllocated') else "16"
    image.bitsStored = dcm.BitsStored if hasattr(dcm, 'BitsStored') else ""
    image.samplesPerPixel = dcm.SamplesPerPixel if hasattr(dcm, 'SamplesPerPixel') else "1"
    image.hotometricInterpretation = dcm.PhotometricInterpretation if hasattr(dcm ,'PhotometricInterpretation') else 'MONOCHROME2'
    # image.softwareVersions = 'syngo MR E11'
    
    return image

def readNrrdStruct(pth_structure: Path) -> Tuple[Dict[str, ROIMask], str]:
    r"""
    Purpose:
        - Load the NRRD structure file.
    Inputs:
        - pth_structure: Path := the path of the structure source file.
    Outputs:
        - structure_mask_dict: Dict[str, ROIMask] := the dictionary of the structure masks.
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

    structure_mask_dict = defaultdict(ROIMask)
    i = 0
    for key in header:
        if f"Segment{i}_Name" == key:
            name = header[f"Segment{i}_Name"]
            if overlap:
                segment_mask = structures_data[:, :, :, i]
            else:
                label_value = header[f"Segment{i}_LabelValue"]
                segment_mask = structures_data == int(label_value)
            # segment_mask = np.pad(segment_mask, 1, mode="constant", constant_values=0)
            if segment_mask.sum() == 0:
                continue
            roi_mask = ROIMask(
                imageArray=np.swapaxes(segment_mask, 0, 2),
                origin=origin,
                spacing=spacing,
                name=name,
            )
            roi_mask.to_lps(current_orientation=orientation)
            structure_mask_dict[name] = roi_mask
            i += 1
    return structure_mask_dict, "LPS"

def readNiftiStruct(pth_structure: Path) -> Tuple[Dict[str, ROIMask], str]:
    r"""
    Purpose:
        - Load the NIFTI structure file into a dictionary of ROIMask objects.
    Inputs:
        - pth_structure: Path := the path of the structure source file.
    Outputs:
        - structure_mask_dict: Dict[str, ROIMask] := the dictionary of the structure masks.
        - orientation: str := the orientation of the structure mask, which is recommended to be LPS.
    Dependencies:
        - nibabel
    """
    assert os.path.exists(pth_structure), "The input path does not exist."
    import nibabel as nib
    structure_nifti = nib.load(pth_structure)
    orientation = "".join(nib.aff2axcodes(structure_nifti.affine))
    structure_data = np.ascontiguousarray(structure_nifti.get_fdata())
    if structure_nifti.header.data_layout == "F":
        structure_data = np.swapaxes(structure_data, 0, 2)
    origin = structure_nifti.affine[:3, 3]
    spacing = structure_nifti.header.get("pixdim")[1:4]    
    origin = origin * [
    np.sign(structure_nifti.affine[0][0]),
    np.sign(structure_nifti.affine[1][1]),
    np.sign(structure_nifti.affine[2][2]),
    ]
    # God knows what is the name of the structures in the nifti files
    # I will just number them and hope for the best
    n_dim = structure_data.ndim
    if n_dim == 4:
        # the segments are over lapping, stored in a 4 dimensinal array 
        num_structures = structure_data.shape[-1]
    else:
        # the segments are non-overlapping, stored in a 3 dimensional array and
        # encoded by value. zero is ignored.
        num_structures = len(np.unique(structure_data))-1

    structure_mask_dict: Dict[str, ROIMask] = {}
    for i in range(num_structures):
        # generate segment labels
        segment_id = f"Segment{i+1}"
        segment_name = segment_id + "_Name"
        segment_label =  segment_id + "_LabelValue"
        # get the segment mask
        if n_dim == 4:
            segment_mask = structure_data[:, :, :, i]
            mask_encoding = np.unique(segment_mask)

            if len(mask_encoding) == 2:
                mask_encoding = mask_encoding[1]
            elif len(mask_encoding) == 1:
                mask_encoding = mask_encoding[0]
            else:
                raise ValueError("The segment mask has more than one unique value, which is not supported.")
            segment_mask = segment_mask == mask_encoding
        else:
            segment_mask = structure_data == i+1
        # segment_mask = np.pad(segment_mask, 1, mode="constant", constant_values=0)
        if segment_mask is None or segment_mask.sum() == 0:
            continue
        roi_mask = ROIMask(
            imageArray=np.swapaxes(segment_mask, 0, 2),
            origin=origin,
            spacing=spacing,
            name=segment_name,
        )
        roi_mask.to_lps(current_orientation=orientation)
        structure_mask_dict[segment_name] = roi_mask
        # del segment_mask
    return structure_mask_dict, "LPS"

def sitk_to_Image3D(sitk_image:Image)-> Image3D | ROIMask:
    r"""
    ### Purpose:
        - to convert a sitk image to an openTPS Image3D object.
    
    ### Inputs:
        - sitk_image: SimpleITK.Image := the image to be converted.
    
    ### Outputs:
        - Image3D := the converted image.
    
    ### Dependencies:
        - SimpleITK
    """    
    image_array = GetArrayFromImage(sitk_image)
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()

    if image_array.dtype == "uint8":
        image_array = image_array.astype("bool")
        return ROIMask(
            imageArray=np.swapaxes(image_array, 0, 2),
            origin=origin,
            spacing=spacing,
        )
    else:
        return Image3D(
            imageArray=np.swapaxes(image_array, 0, 2),
            origin=origin,
            spacing=spacing,
        )

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
    return boxInVoxel

def generate_sphere_mask(
    center: np.ndarray | List[float],
    radius: float,
    gridSize: List[int],
    spacing: List[float] = [1.0, 1.0, 1.0],
    origin: List[float] = [0.0, 0.0, 0.0],
    name: str = "Sphere",
) -> ROIMask:
    r"""
    Purpose:
        - Generate a sphere mask with the given center and radius inside a 3D grid. 
    Inputs:
        - center: np.ndarray | List[float] := the center of the sphere.
        - radius: float := the radius of the sphere.
        - gridSize: List[int] := the size of the grid in [x, y, z].
        - spacing: List[float] := the spacing of the grid in [x, y, z].
        - origin: List[float] := the origin of the grid in [x, y, z].
        - name: str := the name of the sphere mask.
    Outputs:
        - mask_opentps: ROIMask := the generated sphere mask.
    """ 
    center = np.array(center, dtype=float)
    spacing = np.array(spacing, dtype=float)
    origin = np.array(origin, dtype=float)
    
    # create coordiante grid using meshgrid
    x = np.arange(gridSize[0]) * spacing[0] + origin[0]
    y = np.arange(gridSize[1]) * spacing[1] + origin[1]
    z = np.arange(gridSize[2]) * spacing[2] + origin[2]
    
    # meshgrid to ij indexing
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Vectorized distance calculation
    distance_squared = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    mask3D = (distance_squared <= radius**2).astype(np.uint8)
    
    
    mask_opentps = ROIMask(
        imageArray=mask3D,
        name=name,
        origin=origin.tolist(),
        spacing=spacing.tolist(),
    )
    return mask_opentps

def imageToNrrd(
    image_obj: Image3D,
    pth_output: Path,
    anatomical_coordinate_system: str = "left-posterior-superior",
    modality: str = "N/A",
    metadata: Optional[Dict[str, str]] = None,
    ) -> None:
    r"""
    Purpose:
        - To write the image to a nrrd file. By default, all images are written as Left Posterior Superior.
    Inputs:
        - image_obj: Image3D := the image object to write to a nrrd file.
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
    if pth_output.suffix != ".nrrd":
        raise ValueError("The output path should have a '.nrrd' extension.")
    Path.mkdir(pth_output.parent, exist_ok=True, parents=True)
    from collections import defaultdict
    
    image_array_zyx = image_obj.imageArray.swapaxes(0, 2).astype(float)
    header = defaultdict(str)
    header["type"] = "double"
    # header["space dimension"] = "3"
    header["space"] = anatomical_coordinate_system
    header["sizes"] = (
        " ".join(map(str, image_obj.gridSize.tolist()))
    )
    header["space directions"] = [
        [image_obj.spacing[0], 0.0, 0.0],
        [0.0, image_obj.spacing[1], 0.0],
        [0.0, 0.0, image_obj.spacing[2]],
    ]
    header["kinds"] = ["space", "space", "space"]
    header["labels"] = ["x", "y", "z"]
    header["endian"] = "little"
    header["encoding"] = "gzip"
    header["space origin"] = image_obj.origin.tolist()
    header["voxel spacing"] = image_obj.spacing.tolist()
    header["space units"] = ["mm", "mm", "mm"]
    header["modality"] = modality
    header = header | metadata if metadata is not None else header
    nrrd.write(str(pth_output), image_array_zyx, header, index_order="C", compression_level=1)

def masksToNrrd(
        structure_mask_dict: Dict[str, ROIMask],
        pth_output: Path,
        overlap: Optional[bool] = True,
        anatomical_coordinate_system: str = "LPS",
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
        if isinstance(pth_output, str):
            pth_output = Path(pth_output)
        if str(pth_output).endswith("seg.nrrd") is False:
            raise ValueError("The output path should have a 'seg.nrrd' extension.")
        Path.mkdir(pth_output.parent, exist_ok=True)

        spacing  = structure_mask_dict[list(structure_mask_dict.keys())[0]].spacing
        origin = structure_mask_dict[list(structure_mask_dict.keys())[0]].origin
        gridSize = structure_mask_dict[list(structure_mask_dict.keys())[0]].gridSize

        mask_dict = {k: v.imageArray.swapaxes(0,2) for k, v in structure_mask_dict.items()}

        if not overlap:
            sorted_by_size = _sort_segmentation_dict_by_size(mask_dict)
            # this removes overlap
            all_masks = _convert_many_binary_masks_to_1_int_mask(
                sorted_by_size
            )
            from collections import defaultdict
            # # Generic phantom meta data
            header = defaultdict(str)
            header["type"] = "double"
            # header["space dimension"] = "3"
            header["space"] = anatomical_coordinate_system
            header["sizes"] = (
                " ".join(map(str, gridSize.tolist()))
            )
            header["space directions"] = [
                [spacing[0], 0.0, 0.0],
                [0.0, spacing[1], 0.0],
                [0.0, 0.0, spacing[2]],
            ]
            header["kinds"] = ["space", "space", "space"]
            header["labels"] = ["x", "y", "z"]
            header["endian"] = "little"
            header["encoding"] = "gzip"
            header["space origin"] = origin.tolist()
            header["voxel spacing"] = spacing.tolist()
            header["space units"] = ["mm", "mm", "mm"]
  
        else:
            # do not sort the masks, keep the original order
            sorted_by_size = mask_dict
            # stack up all the masks
            all_masks = np.stack(list(sorted_by_size.values()), axis=3).astype(np.uint8)
            from collections import defaultdict
            # # Generic phantom meta data
            header = defaultdict(str)
            header["type"] = "unsigned char"
            header["space dimension"] = "4"
            header["space"] = anatomical_coordinate_system
            header["sizes"] = (
                " ".join(map(str, [all_masks.shape[-1]]+gridSize.tolist()))
            )
            header["space directions"] = [
                [np.nan, np.nan, np.nan],
                [spacing[0], 0.0, 0.0],
                [0.0, spacing[1], 0.0],
                [0.0, 0.0, spacing[2]],
            ]
            header["kinds"] = ["list", "domain", "domain", "domain"]
            # header["labels"] = ["x", "y", "z"]
            header["endian"] = "little"
            header["encoding"] = "gzip"
            header["space origin"] = origin.tolist()
            # header["voxel spacing"] = self.image_obj.spacing.tolist()
            # header["space units"] = ["mm", "mm", "mm"]

        # # Generic Segmentation meta data
        header["Segmentation_ContainedRepresentationNames"] = "Binary labelmap|Closed surface|"
        header["Segmentation_MasterRepresentation"] = "Binary labelmap"
        header["Segmentation_ReferenceImageExtentOffset"] = "0 0 0"
        # header["Segmentation_ConversionParameters"] = "None"  this one is crazy long
        # # Specific segmentation meta data
        for i, name in enumerate(sorted_by_size):
            header[f"Segment{i}_Color"] = " ".join(
                np.round(
                    np.array(
                        structure_mask_dict[name]._displayColor)/ 255, decimals=3
                    ).astype(str)
                )
            # header[f"Segment{i}_ColorAutoGenerated"] =
            header[f"Segment{i}_ID"] = f"Segment_{i+1}"
            if not overlap:
                header[f"Segment{i}_LabelValue"] = f"{i+1}"
            header[f"Segment{i}_Layer"] = f"{i}" if overlap else "0"
            header[f"Segment{i}_Name"] = f"{name}"
            header[f"Segment{i}_NameAutoGenerated"] = "0"
            if sorted_by_size[name].any():
                header[f"Segment{i}_Extent"] = " ".join(map(str, _getExtentOfMask(sorted_by_size[name])))
            # header[f"Segment{i}_Tags"] = "Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|"

        # # any other meta data
        header = header | metadata if metadata is not None else header

        # # Write the image
        nrrd.write(str(pth_output), all_masks, header, index_order="C", compression_level=1)

def contour_to_stl(roi_contour: ROIContour, pth_output: Path) -> None:
    r"""
    Purpose:
        - Export the contour to an STL file via vtkPolyData
    Inputs:
        - roi_contour: ROIContour := the contour to export.
        - pth_output: Path := the path to save the STL file.
    Outputs:
        - None
    :
    """
    raise NotImplementedError("The implementation of this conversion from " \
        "slicewise polygons of the contours to a 3D structured mesh is highly non-trivial." \
        "Please use mask_to_stl instead.")

def mask_to_stl(roi_mask: ROIMask, pth_output: Path) -> None:
    r"""
    Purpose:
        - Convert an ROI mask to an STL file.
    
    Inputs:
        - roi_mask: ROIMask := The ROI mask object containing the 3D binary mask data to be converted.
        - pth_output: Path := The output file path where the STL file will be saved.
    
    Outputs:
        - None
    """
    import vtk
    # Note: Implementation is cannablized from PolySeg (https://github.com/PerkLab/PolySeg/)
    if not isinstance(roi_mask, ROIMask):
        raise ValueError("The input roi_mask should be an instance of ROIMask.")

    elif not pth_output.suffix.lower() == ".stl":
        raise ValueError("The output file must have a .stl extension.")
    
    # Get mask data in [z, y, x] format for VTK
    mask_array = roi_mask.imageArray.astype(np.uint8)
    
    # Create VTK image data
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(mask_array.shape)
    vtk_image.SetSpacing(roi_mask.spacing)
    vtk_image.SetOrigin(roi_mask.origin)
    vtk_image.GetPointData().SetScalars(vtk.util.numpy_support.numpy_to_vtk(
        num_array=mask_array.ravel(order='F'),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    ))

    
    # Pad the image if border voxels are non-zero to ensure closed surface
    extent = vtk_image.GetExtent()
    padder = vtk.vtkImageConstantPad()
    padder.SetInputData(vtk_image)
    padder.SetOutputWholeExtent(
        extent[0] - 1, extent[1] + 1,
        extent[2] - 1, extent[3] + 1,
        extent[4] - 1, extent[5] + 1
    )
    padder.SetConstant(0)
    padder.Update()
    vtk_image = padder.GetOutput()
    
    # Use Flying Edges (faster than marching cubes) or Marching Cubes for surface extraction
    marching_cubes = vtk.vtkDiscreteFlyingEdges3D()

    marching_cubes.SetInputData(vtk_image)
    marching_cubes.SetValue(0, 1)  # Extract surface at label value 1
    marching_cubes.ComputeGradientsOff()
    marching_cubes.ComputeNormalsOff()
    marching_cubes.Update()
    
    poly_data = marching_cubes.GetOutput()
    
    if poly_data.GetNumberOfPolys() == 0:
        raise ValueError("No surface could be generated from the mask. The mask may be empty.")
    

    print(f"Pre-filtration mesh quality: {poly_data.GetNumberOfPolys()} polygons")
    i = 0
    while poly_data.GetNumberOfPolys() > 10000 and i < 10:
        print(f"Current mesh quality: {poly_data.GetNumberOfPolys()} polygons")
        # Apply decimation (0.0 = no decimation, using minimal decimation)
        decimation_factor = 0.5
        if decimation_factor > 0.0:
            decimator = vtk.vtkDecimatePro()
            decimator.SetInputData(poly_data)
            decimator.SetTargetReduction(decimation_factor)
            decimator.SetFeatureAngle(60)
            decimator.SplittingOff()
            decimator.PreserveTopologyOn()
            decimator.SetMaximumError(1.0)
            decimator.Update()
            poly_data = decimator.GetOutput()
                
        # Apply smoothing (0.5 = moderate smoothing)
        smoothing_factor = 1.0
        if smoothing_factor > 0:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(poly_data)
            smoother.SetNumberOfIterations(50)
            # Map smoothing factor to pass band: 0.0->1.0, 0.5->0.01, 1.0->0.001
            pass_band = pow(10.0, -4.0 * smoothing_factor)
            smoother.SetPassBand(pass_band)
            smoother.BoundarySmoothingOn()
            smoother.FeatureEdgeSmoothingOn()
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            poly_data = smoother.GetOutput()
        print(f"Iter {i+1} post-smoothing mesh quality: {poly_data.GetNumberOfPolys()} polygons")
        i += 1

    #clean the mesh
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly_data)
    cleaner.SetTolerance(1e-3)
    cleaner.SetPointMerging(True)
    cleaner.SetConvertLinesToPoints(True)
    cleaner.SetConvertPolysToLines(True)
    cleaner.SetConvertStripsToPolys(True)
    cleaner.Update()
    poly_data = cleaner.GetOutput()
        
    print(f"Final mesh quality: {poly_data.GetNumberOfPolys()} polygons")
        
    # Write to STL file
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(pth_output))
    writer.SetInputData(poly_data)
    writer.Write()

    print(f"STL file saved to {pth_output}")



def get_slicer_color_by_name(name: str) -> List[int]:
    r"""
    Purpose:
        - Get a color by the name of the structure. The color is generated by hashing the name.
    Inputs:
        - name: str := the name of the structure.
    Outputs:
        - color: List[int] := the color of the structure in [R, G, B].
    """
    all_colors = _get_slicer_colors()
    for color in all_colors:
        if color["text_label"]== name.lower():
            return np.array(color["color"])/255
    return np.array([0, 0, 0])/255  # Default color (black) if not found


# Conversion utilities for phantom files
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

def _prepare_phantom_loading_item(pth_input: Path) -> dict:
    """Prepare loading item for phantom files."""
    base_name = pth_input.stem
    full_suffix = "".join(pth_input.suffixes)
    
    if full_suffix in [".nrrd", ".nii", ".nii.gz"]:
        # Look for matching segmentation file
        pth_seg = pth_input.parent / f"{base_name}.seg{full_suffix}"
        args_dict = {"pth_phantom_file": pth_input}
        if pth_seg.exists():
            args_dict["pth_structures_file"] = pth_seg
    elif full_suffix in [".seg.nrrd", ".seg.nii", ".seg.nii.gz"]:
        # Look for matching image file
        pth_input_image = pth_input.parent / f"{base_name}{full_suffix[4:]}"
        args_dict = {"pth_structures_file": pth_input}
        if pth_input_image.exists():
            args_dict["pth_phantom_file"] = pth_input_image
        else:
            args_dict["pth_phantom_file"] = pth_input
    else:
        raise ValueError(
            f"Unsupported file type {full_suffix} for phantom conversion. "
            "Please provide a .nrrd, .nii, .nii.gz, or a dicom directory."
        )    
    # return {"loader_class": BrachyPhantom, "args_dict": args_dict}
    return {"args_dict": args_dict}

def _handle_dicom_directory_phantom(pth_input: Path) -> List[dict]:
    """Process a directory containing DICOM files, return only phantom items."""
    data_to_load = []
    
    if len(list(pth_input.glob("*.[Dd][Cc][Mm]"))) < 1:
        print(f"No DICOM files found in the directory {pth_input}.")
        return data_to_load
    
    # Handle phantom data
    loading_phantom_item = {
        # "loader_class": BrachyPhantom,
        "args_dict": {"dir_dicom": pth_input}
    }
    
    # Check for segmentation file
    segmentation_file = list(pth_input.glob("[Rr][Ss]*.[Dd][Cc][Mm]"))
    if segmentation_file:
        loading_phantom_item["args_dict"]["pth_structures_file"] = segmentation_file[0]
    else:
        print(f"No segmentation file found in the directory {pth_input}")
    
    data_to_load.append(loading_phantom_item)
    return data_to_load

def _perform_phantom_conversion(item: dict, dir_output: Path, type_out: str):
    """Perform actual phantom conversion."""
    # loader_class = item["loader_class"]
    args_dict = item["args_dict"]
    
    # Convert based on output type
    phantom_obj = BrachyPhantom(
        dir_dicom=args_dict.get("dir_dicom"),
        pth_phantom_file=args_dict.get("pth_phantom_file"),
        pth_structures_file=args_dict.get("pth_structures_file")
        )
    if type_out == ".dcm":
        phantom_obj.export_to(dir_dicom_out=dir_output)
    elif type_out == ".nrrd":
        phantom_obj.export_to(dir_nrrd_out=dir_output)
    else:
        raise ValueError(f"Unsupported output type {type_out} for phantom conversion.")


def convert_phantom_files(
    pth_inputs: List[Union[Path, str]],
    type_out: str = ".nrrd",
    dir_output: Optional[Union[Path, str]] = None,
    multi_proc: bool = False
) -> None:
    """
    Convert phantom (image and segmentation) files to the specified output format.
    
    Args:
        pth_inputs: List of paths to input phantom files. Can be directories or files.
        type_out: Output file type. Options are ".nrrd", ".dcm".
        dir_output: Output directory path (optional).
        multi_proc: Whether to use multiprocessing (default: False).
    """
    # Main conversion logic
    data_to_load = []
    
    # Process each input path
    for pth_input in pth_inputs:
        pth_input = Path(pth_input)
        if not pth_input.exists():
            raise FileNotFoundError(f"Input file {pth_input} does not exist.")
        
        # Handle directories (DICOM)
        if pth_input.is_dir():
            dicom_data = _handle_dicom_directory_phantom(pth_input)
            data_to_load.extend(dicom_data)
        
        # Handle single files
        elif pth_input.is_file():
            base_name = pth_input.stem
            
            # Skip if already processed
            if any(
                base_name in str(item["args_dict"].get("pth_phantom_file", ""))
                for item in data_to_load
            ):
                print(f"Skipping {pth_input} as it is already in the list.")
                continue
                
            data_to_load.append(_prepare_phantom_loading_item(pth_input))
        else:
            raise ValueError(f"Input {pth_input} is neither a file nor a directory.")
    
    # Check if we have valid items to process
    if not data_to_load:
        raise ValueError("No valid phantom files found to convert.")
    
    # Setup output directory
    if dir_output is None:
        dir_output = Path(pth_inputs[0]).parent
    else:
        dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)
    
    # Perform conversion
    if multi_proc:
        # Create partial function with fixed arguments
        partial_conversion = partial(_perform_phantom_conversion, dir_output=dir_output, type_out=type_out)
        with Pool() as pool:
            list(tqdm(pool.imap(partial_conversion, data_to_load), total=len(data_to_load), desc="Converting phantom files"))
    else:
        for item in tqdm(data_to_load):
            _perform_phantom_conversion(item, dir_output, type_out)
