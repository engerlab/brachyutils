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
from opentps.core.data.images import CTImage, MRImage, ROIMask
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
    vtkPoints,
    vtkPolyData,
    vtkTransform,
    vtkTransformPolyDataFilter,
)
from vtk.util import numpy_support
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter


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
            - pth_phantom_file: Path := the path of the phantom .nrrd file.
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
        from brachyutils.egsphant_utils import BrachyEgsphant

        self.egsphant_obj: "BrachyEgsphant" = None

        if dir_dicom is not None:
            self._load_dicom_image_files(self.pth_image)
        elif pth_phantom_file is not None:
            self._load_nrrd_image_file(self.pth_image)
        elif pth_egsphant_file is not None:
            self.egsphant_obj = BrachyEgsphant(pth_egsphant_file=pth_egsphant_file)
        else:
            raise ValueError(
                "No geometry source file provided. Please provide either the directory of the DICOM files or the path of the phantom file."
            )
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
        image_files = glob((str(pth_image) + "/*.dcm"))
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
            self.structure_set.setPatient(
                self.image_obj.patient if self.image_obj is not None else None
            )
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
        mask_type: Union[np.ndarray, ROIContour, ROIMask],
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
        assert (
            os.path.splitext(pth_output)[-1] == ".nrrd"
        ), "the file should have '.nrrd' extension"
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
        no_overlap: Optional[bool] = True,
    ) -> None:
        r"""
        Purpose:
            - To write the structures to a nrrd file. By defualt, we remove the overlap between the structures. the smaller structures
            overwrite the larger structures if there is an overlap.
        Inputs:
            - pth_output: Path := the path to write the structures to.
        """
        assert (
            os.path.splitext(pth_output)[-1] == ".nrrd"
        ), "the file should have '.nrrd' extension"
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        structure_mask_dict: dict = self.get_structure_mask(
            self.structure_names_dcm, mask_type=np.ndarray
        )

        if no_overlap:
            # create the sitk segmentation image
            sorted_by_size = _sort_segementation_dict_by_size(structure_mask_dict)
            all_masks = _convert_many_binary_masks_to_1_int_mask(
                sorted_by_size
            )  # this removes overlap
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
        material_dict: dict | Path = None,
        assign_material_from_ct: bool = None,
    ) -> None:
        r"""
        Purpose:
            - Write the BrachyPhantom object to an Egsphant file.
        Inputs:
            - pth_output: Path := the path to write the Egsphant file to.
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
        assert os.path.exists(pth_input_file), "input file does not exist"
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
            macfile_string += f"/source_world/vertex {float_formatter(vertex[0])} {float_formatter(vertex[1])} {float_formatter(vertex[2])} mm\n"

        # add in the face info
        for face in self.faces:
            macfile_string += f"/source_world/face {face[0]} {face[1]} {face[2]}\n"
        # add in the material info
        macfile_string += f"/source_world/material {self.material}\n"
        # add in the density info
        macfile_string += f"/source_world/density {self.density}\n"
        # add in the origin info
        macfile_string += "/source_world/xPosition 0 mm\n"
        macfile_string += "/source_world/yPosition 0 mm\n"
        macfile_string += "/source_world/zPosition 0 mm\n"
        # add in rotation nfo
        macfile_string += "/source_world/xRotation 0 deg\n"
        macfile_string += "/source_world/yRotation 0 deg\n"
        macfile_string += "/source_world/zRotation 0 deg\n"
        # add in the done flag
        macfile_string += "/source_world/done\n"

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


class CatheterTable(list):
    r"""
    Purpose:
        - This class holds the information regarding the catheter table.
        as well as all the functions to support the necessary catheter table operations.

    Attributes:
    """

    def __init__(self, iterable: list = None, pth_catheter_table: Path = None) -> None:

        if pth_catheter_table is not None:
            assert os.path.exists(
                pth_catheter_table
            ), "The input json file does not exist."
            extension = os.path.splitext(pth_catheter_table)[1]
            if extension == ".json":
                iterable = self.load_from_json(pth_catheter_table)
            elif extension == ".dcm"
                iterable = self.load_from_dicom(pth_catheter_table)
        super().__init__(iterable)

    def load_from_json(self, pth_json: Path) -> list:
        r"""
        Purpose:
            - Load the catheter table from a json file.
        Inputs:
            - pth_json: Path := the path to the json file containing the catheter table.
        Outputs:
            - Void := will update the catheter table based on the json file.
        """
        raw_catheter_table:list = []
        with open(pth_json, "r") as json_file:
            catheter_table_list = json.load(json_file)
            assert isinstance(
                catheter_table_list, list
            ), "The json file, should contain a list of catheters."
            for catheter_dict in catheter_table_list:
                raw_catheter_table.append(
                    {
                        "dwells": [
                            DwellPosition(dwell_dict)
                            for dwell_dict in catheter_dict.get("dwells")
                        ],
                        "id": catheter_dict.get("id"),
                        "points": catheter_dict.get("points", []),
                    }
                )
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
        raise NotImplementedError("to be implemented soon")
    
    def to_dict(self) -> dict:
        r"""
        Purpose:
            - To convert the catheter table to a dictionary.
        Inputs:
            - self := the CatheterTable object.
        Outputs:
            - dict := the dictionary containing the catheter table.
        """
        return [
            {
                "id": catheter.get("id"),
                "points": catheter.get("points"),
                "dwells": [dwell.to_dict() for dwell in catheter.get("dwells")],
            } for catheter in self
            ]

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
        dwell_dict: dict = None,
        angle: float = 0,
        position: np.array = None,
        relativePos: int = None,
        rotation: np.array = None,
        time: float = None,
        weight: float = None,
    ) -> None:

        if dwell_dict is not None:
            angle = dwell_dict.get("angle")
            position = np.array(dwell_dict.get("position"))
            relativePos = dwell_dict.get("relativePos")
            rotation = np.array(dwell_dict.get("rotation"))
            time = dwell_dict.get("time")
            weight = dwell_dict.get("weight")

        self.angle = angle
        self.position = position
        self.relativePos = relativePos
        self.rotation = rotation
        self.time = time
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
            "angle": self.angle,
            "position": self.position.tolist(),
            "relativePos": self.relativePos,
            "rotation": self.rotation.tolist(),
            "time": self.time,
            "weight": self.weight,
        }
