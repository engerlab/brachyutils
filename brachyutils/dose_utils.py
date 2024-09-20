import difflib
import logging
import lzma
import os
import copy
import warnings

from pathlib import Path
# trunk-ignore(bandit/B403)
import pickle
import sys
import tkinter as tk
from array import array
from tkinter import filedialog as fd
from typing import List, Optional, Literal, Union

import numpy as np
import pymedphys
import pyzstd
import SimpleITK as sitk
from matplotlib import pyplot as plt
from numpy import ma, reshape
from scipy.interpolate import RegularGridInterpolator

from opentps.core.data.images import DoseImage
class BrachyDose:
    r"""
    Purpse:
        - This class holds information regarding a dose and uncertainty distribution as well as the fundamental
    functions that are applied on the dose. All doses are expressed in Gy and all the unit length is mm.

    Attributes:
        - dose_image:DoseImage := DoseImage object holding the dose grid ([x, y, z]), as well as
            spacing, origin, and rotation ([x, y, z]) information.
        - uncertainty_image:DoseImage := DoseImage object holding the dose uncertainity grid ([x, y, z]), as well as
            spacing, origin, and rotation ([x, y, z]) information.
        - voxel_edges:np.ndarray := coorindates of voxel edges along x, y and z axis.
        - interpolation_function := RegularGridInterpolator object that allows for sampling of dose at arbitrary points ([x, y, z]).
    Functions:
        - load_file_to_brachydose
        - write_brachydose_to_file
        - load_from_3ddose
        - load_from_nrrd
        - load_from_npz
        - load_from_dicom
        - load_from_minidos
        - create_interpolation_function
        - extract_dose_values_from_coordinates
        - extract_profile_2d
        - extract_profile_1d
        - get_average_uncert
        - get_average_uncert_benchmark
        - pad_3ddose
        - write_to_3ddose
        - write_to_nrrd
        - write_to_npz
        - write_to_minidos
        - crop_by_coordinates
        - crop_by_index
        - crop_by_fraction

    Dependencies:
        - opentps.core
        - matplotlib
        - scipy
        - pymedphys
        - SimpleITK
    """

    def __init__(
        self,
        pth_dose_file: Optional[Path] = None,
        load_uncertainty: Optional[bool] = True,
    ):
        self.path = pth_dose_file
        self.dose_image: DoseImage = None
        self.uncertainty_image: DoseImage = None
        self.voxel_edges: np.ndarray = None
        self.interpolation_function = None
        if pth_dose_file is not None:
            self.load_file_to_brachydose(pth_dose_file, load_uncertainty)
        if self.dose_image is not None:
            self.create_interpolation_function()
        # default dose unit length is mm
        self.unit_length:Literal["mm"] = "mm"
        self.xyz_format: bool = True

    def load_file_to_brachydose(
        self, pth_dose_file: Path, load_uncertainty: Optional[bool] = True
    ) -> None:
        r"""
        Purpose:
            - given the path to a file holding dose information, it will return
        a BrachyDose object with the populated available attributes.

        Inputs:
            - pth_dose_file := path directory where the file containing the dose is. The file
                extension could be ".3ddose", ".nrrd", ".dcm", or ".minidos"
            - load_uncertainty := a boolean flag to load the uncertainty image if it is available.
        Output:
            - None := the contents of the file are loaded into the BrachyDose object.
        Dependencies:
            - load_from_3ddose()
            - load_from_nrrd()
            - load_from_dicom()
            - create_interpolation_function()
        """
        pth_dose_file = os.path.abspath(pth_dose_file)

        file_extension = os.path.splitext(pth_dose_file)[-1]

        if file_extension == ".3ddose":
            self.load_from_3ddose(pth_dose_file, load_uncertainty)
        elif file_extension == ".nrrd":
            self.load_from_nrrd(pth_dose_file)
        elif file_extension == ".dcm":
            self.load_from_dicom(pth_dose_file)
        elif file_extension == ".minidos":
            raise NotImplementedError(
                "loading dose from .minidos file is not currently supported"
            )
        elif file_extension == ".bindose":
            raise NotImplementedError("Writing to .bindose not implemented")
        else:
            raise ValueError("file extension not recognized")
        # voxel_centers = self.get_voxel_centers()
        # print(len(self.voxel_edges))
        if self.interpolation_function is None and self.dose_image is not None:
            self.create_interpolation_function()

    def write_brachydose_to_file(self, pth_dose_file: Path) -> None:
        r"""
        Purpose:
            - To write a brachy dose object to the given file path. this function will automatically
        detect the type of the output file and will call the right brachyDose writer function.
        Inputs:
            - pth_dose_file := path where the BrachyDose contents will be written to. The options
            for output type are "3ddose", "nrrd", "npz", "minidos", "xz", and "zstd".
        Dependencies:
            - write_to_3ddose()
            - write_to_nrrd()
            - write_to_npz()
            - write_to_minidos()
            - write_to_xz()
            - write_to_zstd()
        Output:
            - None := contents of self is written to "pth_dose_file"
        """
        file_extension = os.path.splitext(pth_dose_file)[-1]

        if file_extension == ".3ddose":
            self.write_to_3ddose(pth_dose_file)

        elif file_extension == ".nrrd":
            self.write_to_nrrd(pth_dose_file)

        elif file_extension == ".npz":
            self.write_to_npz(pth_dose_file)

        elif file_extension == ".minidos":
            self.write_to_minidos(pth_dose_file)

        elif file_extension == ".xz":
            self.write_to_xz(pth_dose_file)

        elif file_extension == ".zstd":
            self.write_to_zstd(pth_dose_file)

        elif file_extension == ".bindose":
            raise NotImplementedError("Writing to .bindose not implemented")

        else:
            raise ValueError(
                f"The input file name {pth_dose_file} is not supported. the supported \
            file types are '.3ddose', '.nrrd', '.npz', '.minidos', '.xz', and '.zstd'"
            )

    def load_from_3ddose(self, filename: Path, load_uncertainty: Optional[bool] = True) -> None:
        r"""
        Purpose:
            - Given the path to a 3ddose file, load its content into self:BrachyDose.
        Input:
            - filename := path to a ".3ddose" file
        Outputs:
            - void := contents of self is updated.
        Dependencies:
            - numpy
            - os
        """
        assert (
            os.path.splitext(filename)[-1] == ".3ddose"
        ), "this file should have '3ddose' extension."
        path = filename
        # print("Opening 3ddose at %s" % path)
        with open(path, "rb") as newfile:
            bench_voxels = [int(i) for i in newfile.readline().split()]
            bench_x_pos = np.round(
                np.array(newfile.readline().split(), dtype=np.float32), decimals=6
            )
            bench_y_pos = np.round(
                np.array(newfile.readline().split(), dtype=np.float32), decimals=6
            )
            bench_z_pos = np.round(
                np.array(newfile.readline().split(), dtype=np.float32), decimals=6
            )

            bench_x_spacing = bench_x_pos[1] - bench_x_pos[0]
            bench_y_spacing = bench_y_pos[1] - bench_y_pos[0]
            bench_slice_thick = bench_z_pos[1] - bench_z_pos[0]

            huge_dose_array = np.array(
                newfile.readline().strip().split(), dtype=np.float32
            )
            try:
                bench_dose = reshape(
                    huge_dose_array, (bench_voxels[2], bench_voxels[1], bench_voxels[0])
                )
            except ValueError as e:
                print(f"Error in dose file {filename}: {e}", "\n")
                bench_dose = None

            if load_uncertainty:
                try:
                    huge_uncert_array = np.array(
                        newfile.readline().strip().split(), dtype=np.float32
                    )
                    bench_uncert = reshape(
                        huge_uncert_array,
                        (bench_voxels[2], bench_voxels[1], bench_voxels[0]),
                    )
                    self.uncertainty_image = DoseImage(
                        # convert numpy zyx to xyz in opentps.
                        imageArray = np.swapaxes(bench_uncert, 0, 2),
                        origin = (bench_x_pos[0]*10, bench_y_pos[0]*10, bench_z_pos[0]*10),
                        spacing = (bench_x_spacing*10, bench_y_spacing*10, bench_slice_thick*10),
                    )
                except ValueError:
                    print("Warning: No uncertainty in the 3ddose file", filename, "\n")
            
            self.dose_image = DoseImage(
                # convert numpy zyx to xyz in opentps.
                imageArray = np.swapaxes(bench_dose, 0, 2),
                origin = ( bench_x_pos[0]*10, bench_y_pos[0]*10, bench_z_pos[0]*10),
                spacing = (bench_x_spacing*10, bench_y_spacing*10, bench_slice_thick*10),
            )
            self.voxel_edges = self.calculate_voxel_edges()

    def load_from_nrrd(self, pth_nrrd:Path) -> None:
        r"""
        Purpose:
            - given the path to a nrrd dose file, it will load its content into self:BrachyDose
        Inputs:
            - pth_nrrd := Path to a nrrd file writtern by self.to_nrrd()
        Outputs:
            - void := contents of self is updated.
        Dependencies:
            - SimpleITK
            - calculate_voxel_edges()
        """
        loaded_image_nrrd = sitk.ReadImage(pth_nrrd, imageIO="NrrdImageIO")

        dose_uncertainty = sitk.GetArrayFromImage(loaded_image_nrrd)

        if dose_uncertainty.shape[0] == 2:
            dose_array = dose_uncertainty[0]
            uncertainty_array = dose_uncertainty[1]
            voxel_size = np.round(
                np.array(loaded_image_nrrd.GetSpacing()[1:]).astype(np.float32), 1
            )
            origin_coordinates = np.array(loaded_image_nrrd.GetOrigin()[1:]).astype(
                np.float32
            )
        else: 
            if dose_uncertainty.shape[-1] == 2:
                dose_array = dose_uncertainty[:, :, :, 0]
                # no flipping to have everything xyz.
                # dose_array = np.swapaxes(dose_array, 0, 2).astype(np.float32)
                uncertainty_array = dose_uncertainty[:, :, :, 1]
                # no flipping to have everything xyz.
                # uncertainty_array = np.swapaxes(uncertainty_array, 0, 2).astype(np.float32)
            else:
                print("Uncertainty not found in the nrrd file")
                # no flipping to have everything xyz.
                # dose_array = np.swapaxes(dose_uncertainty, 0, 2).astype(np.float32)
                uncertainty_array = None
                
            voxel_size = np.array(loaded_image_nrrd.GetSpacing()).astype(np.float32)
            origin_coordinates = np.array(loaded_image_nrrd.GetOrigin()).astype(np.float32)
                
        # voxel_size = np.flip(voxel_size)
        # origin_coordinates = np.flip(origin_coordinates)

        self.dose_image = DoseImage(
            imageArray = dose_array,
            origin = origin_coordinates,
            spacing = voxel_size,
        )
        self.uncertainty_image = DoseImage(
            imageArray = uncertainty_array,
            origin = origin_coordinates,
            spacing = voxel_size,
        ) if uncertainty_array is not None else None
        
        self.voxel_edges = self.calculate_voxel_edges()

    def load_from_npz(self, pth_npz:Path) -> None:
        r"""
        Purpose:
            - Given the path to an npz file, load its content into self:BrachyDose.
        Input:
            - filename := path to a ".npz" file
        Outputs:
            - void := contents of self is updated.
        Dependencies:
            - numpy
        """

        assert (
            os.path.splitext(pth_npz)[-1] == ".npz"
        ), "the file extension should be npz"

        loaded_brachydose = np.load(pth_npz, allow_pickle=True)
        self.dose_image = loaded_brachydose.get("dose_image")
        self.uncertainty_image = loaded_brachydose.get("uncertainty_image", None)
        self.calculate_voxel_edges()
        self.create_interpolation_function()

    def load_from_dicom(self, pth_RD_dicom:Path):
        r"""
        Purpose:
            - Given the path to a dicom dose file, load its content into self:BrachyDose.
        Inputs:
            - pth_RD_dicom := path to a dicom dose file. baseename must start with "RD".
        Outputs:
            - void := contents of self is updated.
        Dependencies:
            - pydicom
        """
        from opentps.core.io.dicomIO import readDicomDose

        assert os.path.basename(pth_RD_dicom).startswith(
            "RD"
        ), "the basename should start with RD"
        self.dose_image = readDicomDose(pth_RD_dicom)
        if self.dose_image.spacing[2] == 0:
            self.dose_image.spacing[2] = 1.
        # no flipping to have everything xyz.
        # self.dose_image = DoseImage(
        #     imageArray = np.swapaxes(dose_image_xyz.imageArray, 0, 2),
        #     origin = np.flip(dose_image_xyz.origin),
        #     spacing = np.flip(dose_image_xyz.spacing),
        #     name = dose_image_xyz.name,
        #     angles = np.flip(dose_image_xyz.angles),
        #     seriesInstanceUID=dose_image_xyz.seriesInstanceUID,
        #     sopInstanceUID=dose_image_xyz.sopInstanceUID,
        # )
        self.voxel_edges = self.calculate_voxel_edges()

    def load_from_minidos(self, pth_minidos):
        r"""
        Purpose:
            Given the path to a minidos file, load its content into self:BrachyDose
        Input:
            - filename := path to a ".minidos" file
        """
        raise NotImplementedError(
            "loading dose from .minidos file is not currently supported"
        )
        # assert (
        #     os.path.splitext(pth_minidos)[-1] == ".minidos"
        # ), f"the file {pth_minidos}, should have '.minidos' extension."
        # with open(pth_minidos, "rb") as file:
        #     line_content = np.frombuffer(file.readline())

    def create_interpolation_function(self) -> None:
        r"""
        Purpose:
            - To create an interpolation function for the dose grid.
            it allows for sampling of dose at any arbitrary set of points.
        Inputs:
            - self:BrachyDose := the object must have the grid attribute populated.
        Outputs:
            - void := the interpolation function is stored in the object.
        Dependencies:
            - scipy.interpolate.RegularGridInterpolator()
        """
        voxel_centers = self.get_voxel_centers()
        self.interpolation_function = RegularGridInterpolator(
            (voxel_centers[0], voxel_centers[1], voxel_centers[2]),
            self.dose_image.imageArray,
            bounds_error=False,
            fill_value=0,
        )

    def extract_dose_values_from_coordinates(self, x, y, z):
        r""" """
        raise DeprecationWarning("This function is no longer supported due to migration to open tps. please use self.get_dose_at_coordinates() instead.") 
        self.is_not_empty()
        if self.interpolation_function is None:
            raise ValueError("interpolation function is not defined")
        shape = []
        axis = []
        for coord in [z, y, x]:
            if isinstance(coord, float):
                coord_size = 1
                axis.append([coord])
            elif isinstance(coord, np.ndarray):
                coord_size = coord.size
                axis.append(coord)
            else:
                raise TypeError("x, y, and z should be either floats or numpy arrays")
            shape.append(coord_size)
        coord_grid_z, coord_grid_y, coord_grid_x = np.meshgrid(
            [z], [y], [x], indexing="ij"
        )
        # print(coord_grid.shape())
        dose_grid = self.interpolation_function(
            (coord_grid_z, coord_grid_y, coord_grid_x)
        )
        dose_grid.reshape(shape)
        return dose_grid.squeeze()

    def extract_profile_2d(
        self,
        axis_1_coords: np.ndarray,
        axis_2_coords: np.ndarray,
        plane_coord: float,
        plane: str,
    ):
        if (
            not isinstance(axis_1_coords, np.ndarray)
            or not isinstance(axis_2_coords, np.ndarray)
            or not isinstance(plane_coord, float)
        ):
            raise TypeError(
                "axis_1_coords and axis_2_coords should be numpy arrays and plane_coord should be a float"
            )
        if plane == "xy":
            return self.extract_dose_values_from_coordinates(
                axis_1_coords, axis_2_coords, plane_coord
            )
        elif plane == "xz":
            return self.extract_dose_values_from_coordinates(
                axis_1_coords, plane_coord, axis_2_coords
            )
        elif plane == "yz":
            return self.extract_dose_values_from_coordinates(
                plane_coord, axis_1_coords, axis_2_coords
            )
        elif plane == "yx":
            return self.extract_dose_values_from_coordinates(
                axis_2_coords, axis_1_coords, plane_coord
            )
        elif plane == "zx":
            return self.extract_dose_values_from_coordinates(
                axis_2_coords, plane_coord, axis_1_coords
            )
        elif plane == "zy":
            return self.extract_dose_values_from_coordinates(
                plane_coord, axis_2_coords, axis_1_coords
            )
        else:
            raise ValueError(
                "plane should be one of the following: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy'"
            )

    def extract_profile_1d(
        self,
        axis: str,
        axis_1_coords: np.ndarray,
        axis_2_coords: np.ndarray,
        axis_3_coords: List[float],
    ) -> np.ndarray:
        r"""
        Extracts a 1D line profile from the dose grid along the specified axis, given the coordinates of the other two axes and a list of coordinates along the axis of extraction.

        Parameters:
            - axis (str): the axis along which to extract the line profile. Must be one of 'x', 'y', or 'z'.
            - axis_1_coords (np.ndarray): the coordinates of the first axis, as a 1D numpy array.
            - axis_2_coords (np.ndarray): the coordinates of the second axis, as a 1D numpy array.
            - axis_3_coords (List[float]): the coordinates along the axis of extraction, as a list of floats.

        Returns:
            - profile (np.ndarray): the line profile extracted from the dose grid, as a 1D numpy array.
        """
        if axis not in ["x", "y", "z"]:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")

        if axis == "x":
            x = np.array(axis_3_coords)
            y, z = np.meshgrid(axis_2_coords, axis_1_coords, indexing="ij")
        elif axis == "y":
            y = np.array(axis_3_coords)
            x, z = np.meshgrid(axis_1_coords, axis_2_coords, indexing="ij")
        else:
            z = np.array(axis_3_coords)
            x, y = np.meshgrid(axis_1_coords, axis_2_coords, indexing="ij")

        dose_grid = self.extract_dose_values_from_coordinates(x, y, z)
        profile = np.mean(dose_grid, axis=(1, 2))
        return profile

        # pdd_dict["x_axis"] = z_values
        # pdd_dict["y_axis"] = np.array(dose_values)
        # return pdd_dict

    def get_average_uncert(self) -> float:
        r"""
        Purpose:
            XXX Documentation is missing
        """
        max_dose = self.dose_image.imageArray.max()
        dose_mask = self.dose_image.imageArray < 0.2 * max_dose
        masked_uncert = ma.array(self.uncertainty_image.imageArray, mask=dose_mask)
        masked_dose = ma.array(self.dose_image.imageArray, mask=dose_mask)
        average_uncert = ma.average(masked_uncert / masked_dose) * 100
        return average_uncert

    def get_average_uncert_benchmark(self) -> float:
        r"""
        Purpose:
            Documentation is missing
        """
        max_dose = self.dose_image.imageArray.max()
        dose_mask = self.dose_image.imageArray < 0.2 * max_dose
        masked_uncert = ma.array(self.uncertainty_image.imageArray, mask=dose_mask)
        average_uncert = ma.average(masked_uncert) * 100
        return average_uncert

    def pad_3ddose(self, new_dims: list, new_origin: list) -> "BrachyDose":
        r"""a function to padd the grid and uncertainty in BrachyDose object and bring it to the desired dimensios.
        it will update all the aspects of the dose object to match the new dimensiosn.
        The voxels must have the same size! remember, python does z, y, x.
        inputs:
            self:BrachyDose

            new_dims := a 1 by 3 list containing the new x, y and z dimensions:
                [new_z_dim, new_y_dim, new_x_dim]

            new_origin := coordinates of the new origin_coordinates
                [x, y, z]
        """
        warnings.warn("This function is not tested yet", stacklevel=2)
        assert any(
            new_dims > self.dose_image.gridSize
        ), "since you are padding, the new dimensions should be larger than the input dimensions"

        # calculate distances between the new and old origin_coordinates voxels.
        # if for an axis, the distance of origin_coordinatess is larger than the voxel size, use the new origin_coordinates
        # else, use the old top left
        origin_coordinates_distance = np.abs(new_origin - self.dose_image.origin)
        final_origin_coordinates = np.zeros(3)
        for i, distance in zip(range(3), origin_coordinates_distance):
            final_origin_coordinates[i] = (
                new_origin[i] if distance > self.dose_image.spacing[i] else self.dose_image.origin[i]
            )

        # figure out how much padding to do before and after each axis
        padding = np.zeros([3, 2])
        for i in range(3):
            if final_origin_coordinates[i] == self.dose_image.origin[i]:
                # all padding goes to the end for this dose axis
                pad_before = 0
                pad_after = new_dims[i] - self.dose_image.gridSize[i]
            else:
                # all padding goes to the begining of the dose axis
                pad_before = new_dims[i] - self.dose_image.gridSize[i]
                pad_after = 0
            padding[i] = [pad_before, pad_after]

        # pad the old dose grid to get the new grid!
        new_dose_grid = np.pad(self.dose_image.imageArray, tuple(padding.astype(int)), mode="edge")
        if self.uncertainty_image is not None:
            new_uncert = np.pad(
                self.uncertainty_image.imageArray, tuple(padding.astype(int)), mode="edge"
            )        
        # fillout the new padded dose dictionary
        padded_dose = BrachyDose()
        padded_dose.dose_image = DoseImage(
            imageArray = new_dose_grid,
            origin = final_origin_coordinates,
            spacing = self.dose_image.spacing,
        )
        if self.uncertainty_image is not None:
            padded_dose.uncertainty_image = DoseImage(
                imageArray = new_uncert,
                origin = final_origin_coordinates,
                spacing = self.dose_image.spacing,
            )
        self.calculate_voxel_edges()
        self.create_interpolation_function()
        return padded_dose

    def write_to_3ddose(self, file_name: str):
        r"""
        Purpose:
            This function will write the contents of a BrachyDose onto a text file with .3ddose extension.

        inputs:
            - self := a BrachyDose object containing the following keys:
                grid [z, y, x]
                uncert [z, y, x]
                voxel_size [x, y, z]
                origin_coordinates [x, y, z]
                axis [z, y, x]

            - file_name := the directory path where the file will be written
        """
        file_name = os.path.abspath(file_name)

        # dimensions = " ".join(map(str, np.flip(self.dose_image.gridSize.astype(int)))) + "\n"
        dimensions = " ".join(map(str, self.dose_image.gridSize.astype(int))) + "\n"
        x_axis = " ".join(map(str, self.voxel_edges[0]/10)) + "\n"
        y_axis = " ".join(map(str, self.voxel_edges[1]/10)) + "\n"
        z_axis = " ".join(map(str, self.voxel_edges[2]/10)) + "\n"
        dose_flattened = " ".join(map(str, self.dose_image.imageArray.flatten("C"))) + "\n"
        if self.uncertainty is not None:
            uncertainty_flattened = (
                " ".join(map(str, self.uncertainty_image.imageArray.flatten("C"))) + "\n"
            )
        else:
            uncertainty_flattened = ""

        with open(file_name, "w") as file:
            lines = [
                dimensions,
                x_axis,
                y_axis,
                z_axis,
                dose_flattened,
                uncertainty_flattened,
            ]
            file.writelines(lines)

    def write_to_nrrd(
            self, pth_output: Path,
            metadata: Optional[dict] = None,
            format: Optional[Literal["rapidbrachy", "slicer"]] = "rapidbrachy"):
        r"""
        Purpose:
            To save the contents of BrachyDose into a nrrd file.
        inputs:
            - pth_output := path where the dose nrrd file will be written to.

            - metadata := a dictionary containing the following meta data key values (should be changed later):
                "cancer site":
                "care center":
                "number of dwell positions":
                "number of segmented structures":
                "patient number":
                "Image content": "[3D dose, 3D uncertainty]"
        outputs: Void
            writes [3D dose, 3D uncertainty], voxel size, origin (origin_coordinates), and metadata to the file_name_dose.nrrd
            note that 3D dose files are written in z, y, x, but the sitk image is written in x, y, z.
        """
        # check if the directory exists, if not create it. make sure the file extension is write.
        os.makedirs(os.path.dirname(pth_output), exist_ok=True)
        assert os.path.splitext(pth_output)[-1] == ".nrrd", "the file should have '.nrrd' extension"
        
        # create sitk dose image
        dose_array = self.dose_image.imageArray.astype(np.float32)
        # no flipping to have everything xyz.
        # dose_array = np.swapaxes(self.dose_image.imageArray, 0, 2).astype(np.float32)
        if self.uncertainty_image is not None:
            uncertainty_array = self.uncertainty_image.imageArray.astype(np.float32)
            # no flipping to have everything xyz.
            # uncertainty_array = np.swapaxes(self.uncertainty_image.imageArray, 0, 2).astype(np.float32)
        
        # # old nrrd format
        # image_nrrd = sitk.JoinSeries(
        #     sitk.GetImageFromArray(dose_array), sitk.GetImageFromArray(uncertainty_array)
        # )
        # # new nrrd format
        if format == "rapidbrachy":
            fiter = sitk.ComposeImageFilter()
            if self.uncertainty_image is not None:
                image_nrrd = fiter.Execute(
                    sitk.GetImageFromArray(dose_array), sitk.GetImageFromArray(uncertainty_array)
                )
            else:
                image_nrrd = sitk.GetImageFromArray(dose_array)

            image_nrrd.SetOrigin(self.dose_image.origin).astype(float)
            image_nrrd.SetSpacing(self.dose_image.spacing).astype(float)
        elif format == "slicer":
            raise NotImplementedError("slicer format is not implemented yet")
        else:
            raise ValueError("format should be either 'rapidbrachy' or 'slicer'")

        # set the metadata: all sitk Images belonging to a patient will have the same meta data
        if metadata is not None:
            for key in metadata:
                image_nrrd.SetMetaData(key, metadata[key])

        # write out the files
        sitk.WriteImage(
            image_nrrd, pth_output, useCompression=True, compressionLevel=9
        )

    def write_to_npz(self, file_name: str):
        r"""
        Purpose:
            To save the contents of BrachyDose into a npz file, which is numpy compressed.
        inputs:
            - self := BrachyDose object
            - file_name := path where the dose npz file will be written to.
        outputs: Void
            writes the contents of self:BrachyDose to the file_name.
        """

        assert os.path.splitext(file_name)[-1] == ".npz", "the file name should have '.npz' extension."

        np.savez_compressed(
            file=file_name,
            dose_image = self.dose_image,
            uncertainty_image = self.uncertainty_image if self.uncertainty_image is not None else None, 
            axis=self.voxel_edges,
        )

    def write_to_minidos(self, file_name: str):
        r"""
        Purpose:
            - To save the contents of BrachyDose into a minidos file, which is just a binary file written line by line.
            This code is based on Maude Robitaille's implementation.
            This script was developed by Maude Robitaille.
        inputs:
            - self := BrachyDose object
            - file_name := path where the dose minidos file will be written to.

        outputs: Void
            writes the contents of self:BrachyDose to the file_name.
        """
        assert (
            os.path.splitext(file_name)[-1] == ".minidos"
        ), f"the file name {file_name} should have '.minidos' extension."
        with open(file_name, "wb") as newfile:

            # the first line is the number of voxels along each dimension [x, y , z]
            dims_array = array("i", self.dose_image.gridSize)
            dims_array.tofile(newfile)

            # lines 2,3 and 4 are the voxel sizes x, y, z
            float_array_x = array("f", [self.dose_image.spacing[2]])
            float_array_x.tofile(newfile)
            float_array_y = array("f", [self.dose_image.spacing[1]])
            float_array_y.tofile(newfile)
            float_array_z = array("f", [self.dose_image.spacing[0]])
            float_array_z.tofile(newfile)

            # lines 5, 6, 7 are the origins x, y, and z
            originx_array = array("f", [self.dose_image.origin[2]])
            originy_array = array("f", [self.dose_image.origin[1]])
            originz_array = array("f", [self.dose_image.origin[0]])

            originx_array.tofile(newfile)
            originy_array.tofile(newfile)
            originz_array.tofile(newfile)

            # lines 8 is just a zero
            zero = array("i", [0])
            zero.tofile(newfile)

            # line 9-infinit is the dose per voxel array
            for d in self.dose_image.imageArray.flatten():
                array("f", [d]).tofile(newfile)

    def write_to_xz(self, fileName):
        assert os.path.splitext(fileName)[-1] == ".xz"

        with lzma.open(fileName, "wb") as file:
            pickle.dump(self, file)

    def write_to_zstd(self, file_name):
        assert os.path.splitext(file_name)[-1] == ".zst"

        with pyzstd.open(file_name, "wb", level_or_option=22) as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_voxel_edges(self):
        r"""
        Purpose:
        - will calculate the axes coordinates for a 3ddose dictionary.
        Input:
            - self:BrachyDose
        Output:
            - axes:numpy.array() :=
            [[x_min:voxel_size:x_max],
            [y_min:voxel_size:y_max],
            [z_min:voxel_size:z_max]]
        """
        assert self.dose_image is not None, "dose image is not defined. please load a dose image first"
        voxel_centers = self.get_voxel_centers()
        self.voxel_edges = np.empty(len(voxel_centers), dtype=object)
        for i in range(len(voxel_centers)):
            self.voxel_edges[i] = voxel_centers[i] - self.dose_image.spacing[i] / 2.0
        return self.voxel_edges

    def get_voxel_centers(self):
        r"""
        Purpose:
            - To calculate the voxel centers of the dose grid.
        Inputs:
            - self:BrachyDose
        Outputs:
            - voxel_centers := a numpy array containing the coordinates of the voxel centers for each axis.
        Dependencies:
            - Image3D.getMeshGridPositions()
        """
        assert self.dose_image is not None, "dose image is not defined. please load a dose image first"
        voxel_centers = np.empty(len(self.dose_image.origin), dtype=object)
        for i in range(len(self.dose_image.origin)):
            voxel_centers[i] = (
                self.dose_image.origin[i] 
                + np.arange(self.dose_image.gridSize[i]) * self.dose_image.spacing[i]
            )
        return voxel_centers

    def get_dose_at_coordinates(self, coords:Union[np.ndarray, List[float]]) -> float:
        r"""
        Purpose:
            - Given a set of coordinates, this function will return the dose at that point.
        Inputs:
            - coords := a list of 3 coordinates [z, y, x] or a numpy array of shape (3,)
        Outputs:
            - dose := the dose at the given coordinates in Gy
        """
        assert len(coords) == 3, "coords should be a list of 3 coordinates"
        return self.dose_image.getDataAtPosition(coords)

    def get_uncertainty_at_coordinates(self, coords:Union[np.ndarray, List[float]]) -> float:
        r"""
        Purpose:
            - Given a set of coordinates, this function will return the dose at that point.
        Inputs:
            - coords := a list of 3 coordinates [z, y, x] or a numpy array of shape (3,)
        Outputs:
            - dose := the dose at the given coordinates in Gy
        """
        assert len(coords) == 3, "coords should be a list of 3 coordinates"
        assert self.uncertainty_image is not None, "uncertainty image is not defined"
        return self.uncertainty_image.getDataAtPosition(coords)
    
    def is_equal(self, new_brachy_dose) -> bool:
        r"""
        Purpose:
            - To compare if self:BrachyDose has the same attributes as an input BrachyDose
        Inputs:
            - new_brachy_dose: another BrachyDose object whose attributes may or may not contain
            equal info as the attributes of self.
        Outputs:
            True if attributes of new_brachy_dose are the same as self
            False otherwise
        """
        if not isinstance(new_brachy_dose, BrachyDose):
            warnings.warn("input must be of type BrachyDose", stacklevel=2)
            return False
        elif not np.array_equal(
            self.dose_image.imageArray,
            new_brachy_dose.dose_image.imageArray
            ):
            warnings.warn("dose values are not the same", stacklevel=2)
            return False
        elif not np.array_equal(
            np.concatenate(self.voxel_edges),
            np.concatenate(new_brachy_dose.voxel_edges),
        ):
            warnings.warn("axis is not the same", stacklevel=2)
            return False
        elif not self.uncertainty_image is not None:
            if np.array_equal(
                self.uncertainty_image.imageArray,
                new_brachy_dose.uncertainty_image.imageArray
            ):
                warnings.warn("uncertainty is not the same", stacklevel=2)
                return False
        elif not np.array_equal(
            self.dose_image.gridSize, new_brachy_dose.dose_image.gridSize
        ):
            warnings.warn("num_voxels is not the same", stacklevel=2)
            return False
        elif not np.array_equal(
            self.dose_image.spacing, new_brachy_dose.dose_image.spacing
        ):
            warnings.warn("voxel_size is not the same", stacklevel=2)
            return False
        elif not np.array_equal(
            self.dose_image.origin, new_brachy_dose.dose_image.origin
        ):
            warnings.warn("origin_coordinates is not the same", stacklevel=2)
            return False
        else:
            return True

    def crop_by_coordinates(
        self, coordinate_range: np.array, inplace: Optional[bool] = True
    ) -> Union[None, "BrachyDose"]:
        r"""
        Purpose:
            - given a range of coordinate (mix and max on each axis), this function will crop
            dose and uncertainty maps and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - coordinate_range := a 3 x 2 array holding the min and max on z, y and x axis
                [[z_min, z_max], [y_min, y_max], [x_min, x_max]]
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the coordinate range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            opentps.core.processing.imageProcessing.resampler3D.crop3DDataAroundBox
        """
        from opentps.core.processing.imageProcessing.resampler3D import crop3DDataAroundBox

        self.is_not_empty()
        assert coordinate_range.shape == (3, 2), "coordinate_range should be a 3x2 array in z, y, x order"
        if inplace:
            crop3DDataAroundBox(self.dose_image, coordinate_range)
            if self.uncertainty_image is not None:
                crop3DDataAroundBox(self.uncertainty_image, coordinate_range)
            self.calculate_voxel_edges()
            self.create_interpolation_function()
        else:
            new_dose:BrachyDose = copy.deepcopy(self)
            new_dose.crop_by_coordinates(coordinate_range, inplace=True)
            return new_dose

    def crop_by_fraction(
        self,
        crop_fraction:List[float],
        inplace: Optional[bool] = True
        ) -> Union[None, "BrachyDose"]:
        r"""
        Purpose:
            - given the crop_fraction, this function will crop out 0.5*(1 - crop_fraction)*gridSizeInWorldUnit
            from the edges of the x and y, and z axis of dose and uncertainty maps and will adjust the rest of 
            the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - crop_fraction := 3 floating point between 0 and 1 (one per axis for z, y, x axis), which is the fraction of the image axis 
            that remains in the crop. for example, a crop ratio of [1, 0.5, 0.5] will keep the center of the x and y axis, 
            plus minus 0.25*dimension of the image. The z axis will not be cropped.
                    +++++++++       ---------
                    +++++++++       --+++++--
                    +++++++++  ===> --+++++--
                    +++++++++       --+++++--
                    +++++++++       ---------
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the coordinate range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            -self.crop_by_coordinates()
        """
        crop_fraction = np.array(crop_fraction)
        assert (
            np.all(crop_fraction >= 0) and np.all(crop_fraction <= 1)
        ), "the fraction should be between 0 and 1"
        
        off_set = 0.5 * (1 - crop_fraction) * self.dose_image.gridSizeInWorldUnit
        new_origin_coords = self.dose_image.origin + off_set
        assert np.all(
            new_origin_coords >= self.dose_image.origin
        ), "new origin cannot be smaller than the original origin."
        
        new_ending_coords = self.dose_image.origin + self.dose_image.gridSizeInWorldUnit - off_set
        
        new_coords_range = np.column_stack([new_origin_coords, new_ending_coords])

        return self.crop_by_coordinates(new_coords_range, inplace)

    def crop_by_index(
        self,
        index_range: np.array,
        inplace: Optional[bool] = True
        ) -> Union[None, "BrachyDose"]:
        r"""
        Purpose:
            - given a range of indicies (mix and max on each axis), this function will crop
            dose and uncertainty maps and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - index_range := a 3 x 2 array holding the min and max on z, y and x axis
                [[z_min, z_max], [y_min, y_max], [x_min, x_max]]
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the index range.
        Dependencies:
            - self.crop_by_coordinates()
        """
        
        # convert indicies to coordinates
        new_origin_coords = self.dose_image.getPositionFromVoxelIndex(index_range[:, 0])
        new_ending_coords = self.dose_image.getPositionFromVoxelIndex(index_range[:, 1])
        new_coords_range = np.column_stack([new_origin_coords, new_ending_coords])
        return self.crop_by_coordinates(new_coords_range, inplace)
        
    def is_not_empty(self) -> bool:
        assert self.dose_image is not None, "error dose image is None"
        assert self.voxel_edges is not None, "error axis is None"
        return True

    def info(self) -> None:
        self.is_not_empty()
        print(f"shape of dose grid is: {self.dose_image.gridSize}")
        if self.uncertainty_image is not None:
            print(f"shape of uncertainty matrix is: {self.uncertainty_image.gridSize}")
        # print(f"num voxels attribute is: {self.num_voxels}")
        print(f"the origin of dose image is {self.dose_image.origin}")
        print(f"the voxel size is {self.dose_image.spacing}")
        print(
            f"the size of the x, y and z axes are {self.voxel_edges[0].shape, self.voxel_edges[1].shape, self.voxel_edges[2].shape}"
        )
        print(f"The grid size in world unit is {self.dose_image.gridSizeInWorldUnit}")
        
    def crop_by_dicom_structure(
        self,
        pth_dir_dicom: Path,
        structure_name: str,
        inplace: Optional[bool] = False,
    ) -> Union[None, "BrachyDose"]:
        r"""
        Purpose:
            - based on the given dicom structure file, crop the BrachyDose object such
            that it only contains the smallest bounding box around the structure structures.
            
        Inputs:
            - pth_dir_dicom := pth_dir_dicom := path to the directory with the dicom files of a patient.
                it should contain both images and RTSTRUCT file. this input is optional
            - structure_name_list := a list of strings containing the names of the structures that the dose will be cropped to.
            
        Outputs:
            - Void := will crop the dose and uncertainty maps of self to have the range of the structure.
            """
        from brachyutils import BrachyDicom
        from opentps.core.data.images import ROIMask
        from opentps.core.processing.segmentation.segmentation3D import getBoxAroundROI
        from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D
        # load the dicom object both the image and the mask
        dicom_obj = BrachyDicom(pth_dir_dicom, load_structure=True)
        # load the mask dictionary for the structures in structure_name_list
        mask_dict = dicom_obj.get_strcuture_mask_from_dicom([structure_name], ROIMask)
    
        # Get a cropped dose map that tightly fits each mask.
        resampled_mask = resampleImage3DOnImage3D(mask_dict[structure_name], self.dose_image)
        box_around_mask = np.array(getBoxAroundROI(resampled_mask))
        return self.crop_by_coordinates(box_around_mask, inplace)
        
    def multiply_dose_by_constant(
        self, scale_factor: float, scale_uncert: Optional[bool] = False
    ) -> None:
        r"""
        Purpose:
            - to scale the dose and uncertainty maps by a constant factor.
        Inputs:
            - scale_factor := a floating point number that the dose and uncertainty maps will be scaled by.
        Outputs:
            - Void := will scale the dose and uncertainty maps of self by the scale factor.
        """
        self.is_not_empty()
        self.dose_image.imageArray *= scale_factor
        if scale_uncert and self.uncertainty_image is not None:
            self.uncertainty_image.imageArray *= scale_factor


def dose_with_empty_grid_like(doseObj: BrachyDose):
    r"""
    Purpose:
        - To create a new dose object with the same attributes as the input dose object,
        but with an empty grid and uncertainty.

    Inputs:
        - doseObj: BrachyDose object

    Outputs:
        empty_dose: BrachyDose object with empty grid and uncertainty
    """
    new_dose = BrachyDose()
    new_dose.dose_image = DoseImage(
        imageArray=np.zeros_like(doseObj.dose_image.imageArray),
        gridSize=doseObj.dose_image.gridSize,
        origin=doseObj.dose_image.origin,
    )
    if doseObj.uncertainty_image is not None:
        new_dose.uncertainty_image = DoseImage(
            imageArray=np.zeros_like(doseObj.dose_image.imageArray),
            gridSize=doseObj.dose_image.gridSize,
            origin=doseObj.dose_image.origin,
        )
    new_dose.calculate_voxel_edges()
    new_dose.create_interpolation_function()
    return new_dose


def compare_two_3ddose_files(pth1_3ddose: str, pth2_3ddose: str):
    # old_file_dir = load_3ddose(pth1_3ddose)
    # new_file_dir = load_3ddose(pth2_3ddose)

    with open(pth1_3ddose, "r") as file1, open(pth2_3ddose) as file2:
        contents1 = file1.read()
        contents2 = file2.read()

    if contents1 == contents2:
        print("write 3ddose works fine")
    else:
        print("write 3ddose does not work fine")
        print("here are the differences")
        diff_list = list(difflib.ndiff(contents1.splitlines(), contents2.splitlines()))
        print("\n".join(diff_list))

class DoseComparison:

    def __init__(
        self,
        dose1: BrachyDose,
        dose2: BrachyDose,
        gamma_dose_percent_threshold: float,
        gamma_distance_threshold_mm: float,
        compute_percent_difference=True,
        compute_gamma_index=True,
        prescription_dose: float = None,
        max_gamma=None,
        path=None,
        gamma_kwargs: dict = {
            "lower_percent_dose_cutoff": 5,
            "interp_fraction": 10,
            "local_gamma": False,
            "global_normalisation": None,
            "skip_once_passed": False,
        },
    ):
        # provide no dose to just load a file
        if dose1 is None and dose2 is None:
            self.load_comparison_object(path)
            return

        self.dose1 = dose1
        self.dose2 = dose2
        # axis is taken from the first dose provided
        self.voxel_centers = dose1.get_voxel_centers()
        self.dose_2_grid_resampled = self.dose2.extract_dose_values_from_coordinates(
            self.voxel_centers[2], self.voxel_centers[1], self.voxel_centers[0]
        )
        self.percent_difference: BrachyDose = None
        self.gamma_index: BrachyDose = None
        self.gamma_dose_percent_threshold = gamma_dose_percent_threshold
        self.gamma_kwargs = gamma_kwargs
        # we can index the dose cutoff to the prescription dose
        if isinstance(prescription_dose, float) or isinstance(prescription_dose, int):
            self.gamma_kwargs["global_normalisation"] = prescription_dose
        if isinstance(max_gamma, float) or isinstance(prescription_dose, int):
            self.max_gamma = max_gamma
            self.gamma_kwargs["max_gamma"] = max_gamma
        else:
            self.max_gamma = 2
        # axes values are assumed in cm from the 3ddose formalism
        # gamma distance thresholds are usually provided in mm
        # pymedphys documentation indicates that the threshold unit must match the axis
        # despite the name of the function input containing 'mm'
        self.gamma_distance_threshold = gamma_distance_threshold_mm / 10.0
        if compute_percent_difference:
            self.compute_percent_difference()
        if compute_gamma_index:
            self.compute_gamma_index()

    def plot_2d_dose_comparison(
        self,
        axis_1_coords: np.ndarray,
        axis_2_coords: np.ndarray,
        plane_coord: float,
        plane: str,
        plot_titles: tuple,
    ):
        import itertools

        import matplotlib
        from matplotlib.ticker import (
            AutoMinorLocator,
            FormatStrFormatter,
            MultipleLocator,
        )

        matplotlib.rcParams.update({"font.size": 8})
        plt.rcParams.update({"figure.dpi": 300})
        dose_1_profile = self.dose1.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )
        dose_2_profile = self.dose2.extract_profile_2d(
            axis_1_coords, axis_2_coords, plane_coord, plane
        )
        if self.percent_difference is not None:
            percent_difference_profile = self.percent_difference.extract_profile_2d(
                axis_1_coords, axis_2_coords, plane_coord, plane
            )
        if self.gamma_index is not None:
            gamma_index_profile = self.gamma_index.extract_profile_2d(
                axis_1_coords, axis_2_coords, plane_coord, plane
            )
        else:
            raise NotImplementedError(
                """Plotting of a comparison without computing the percent difference or
            gamma index is not supported"""
            )
        # we will plot a figure that is suitable as a double column figure for medical physics
        mm = 1.0 / 25.4  # define millimeters (relative to inches=1)
        fig, ax = plt.subplots(
            figsize=(180 * mm, 120 * mm), nrows=2, ncols=2, sharex=True, sharey=True
        )
        c00 = ax[0, 0].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            dose_1_profile,
            vmin=0,
            vmax=30,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[0, 0].set_title(plot_titles[0], fontsize=12, pad=5, fontweight="bold")
        ax[0, 0].set_aspect("equal")
        cbar00 = fig.colorbar(c00, ax=ax[0, 0], shrink=0.9, pad=0.04)
        cbar00.set_label(label="Dose [Gy]", size=10, labelpad=5)
        # cbar00.mappable.set_clim(0, max_dose)
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_ylabel("y (cm)", fontsize=10)
        c01 = ax[0, 1].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            dose_2_profile,
            vmin=0,
            vmax=30,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[0, 1].set_title(plot_titles[1], fontsize=12, pad=5, fontweight="bold")
        ax[0, 1].set_aspect("equal")

        cbar01 = fig.colorbar(c01, ax=ax[0, 1], shrink=0.9, pad=0.04)
        cbar01.set_label(label="Dose [Gy]", size=10, labelpad=5)
        # cbar01.mappable.set_clim(0, max_dose)g
        ax[0, 1].invert_yaxis()
        c10 = ax[1, 0].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            percent_difference_profile,
            vmin=0,
            vmax=200,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[1, 0].set_title("Percent Difference", fontsize=12, pad=5, fontweight="bold")
        ax[1, 0].set_aspect("equal")
        cbar10 = fig.colorbar(c10, ax=ax[1, 0], shrink=0.9, pad=0.04)
        cbar10.set_label(label="[%]", size=10, labelpad=5)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_xlabel("x (cm)", fontsize=10)
        ax[1, 0].set_ylabel("y (cm)", fontsize=10)

        c11 = ax[1, 1].pcolormesh(
            axis_1_coords,
            axis_2_coords,
            gamma_index_profile,
            vmin=0,
            vmax=self.max_gamma,
            cmap="turbo",
            rasterized=True,
            antialiased=True,
        )
        ax[1, 1].set_title(
            f"Gamma ({self.gamma_dose_percent_threshold}% / {int(10.*self.gamma_distance_threshold)} mm)",
            fontsize=12,
            pad=5,
            fontweight="bold",
        )
        ax[1, 1].set_aspect("equal")
        #: Pass Rate = {np.round(self.gamma_pass_ratio*100,1)}%"
        cbar11 = fig.colorbar(c11, ax=ax[1, 1], shrink=0.9, pad=0.04)
        cbar11.set_label(label="Gamma", size=10, labelpad=5)
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_xlabel("x (cm)", fontsize=10)
        plt.tight_layout()
        plt.savefig("dose_comparison.eps", dpi=300)
        plt.show()

    def compute_percent_difference(self):
        self.percent_difference = BrachyDose()
        self.percent_difference.grid = (
            np.abs(self.dose1.grid - self.dose_2_grid_resampled) / self.dose1.grid * 100
        )
        self.percent_difference.voxel_edges = self.dose1.voxel_edges
        self.percent_difference.voxel_size = self.dose1.voxel_size
        self.percent_difference.origin_coordinates = self.dose1.origin_coordinates
        self.percent_difference.num_voxels = self.dose1.num_voxels
        self.percent_difference.create_interpolation_function()

    def compute_gamma_index(self):
        print("Computing gamma index may take time")
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.gamma_index = BrachyDose()
        gamma_index_grid = pymedphys.gamma(
            tuple(self.voxel_centers),
            self.dose1.grid,
            tuple(self.voxel_centers),
            self.dose_2_grid_resampled,
            self.gamma_dose_percent_threshold,
            self.gamma_distance_threshold,
            **self.gamma_kwargs,
        )
        # cast the NaNs to 0s
        number_excluded = np.sum(np.isnan(gamma_index_grid))
        gamma_index_grid[np.isnan(gamma_index_grid)] = -1
        self.gamma_index.grid = gamma_index_grid
        self.gamma_index.voxel_edges = self.dose1.voxel_edges
        self.gamma_index.voxel_size = self.dose1.voxel_size
        self.gamma_index.origin_coordinates = self.dose1.origin_coordinates
        self.gamma_index.num_voxels = self.dose1.num_voxels
        self.gamma_pass_ratio = (
            np.sum(self.gamma_index.grid <= 1) - number_excluded
        ) / (self.gamma_index.grid.size - number_excluded)
        self.gamma_index.create_interpolation_function()

    def save_comparison_object(self, path: str = None):
        r"""
        Saves the dose comparison object to a file using the pickle module.

        Returns:
        None
        """
        if not isinstance(path, str):
            root = tk.Tk()
            root.withdraw()
            f = fd.asksaveasfile(
                mode="wb",
                defaultextension=".comp",
                initialdir=os.getcwd(),
                title="Save dose comparison object",
                confirmoverwrite=True,
            )
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            root.destroy()
            f.close()
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load_comparison_object(self, path: str = None):
        r"""
        Opens the dose comparison object file and updates the current object's attributes with the loaded object's attributes.

        Returns:
        None
        """
        if not isinstance(path, str):
            root = tk.Tk()
            root.withdraw()
            f = fd.askopenfile(
                mode="rb",
                parent=root,
                initialdir="$HOME",
                title="Select saved dose comparison file",
            )
            self.__dict__.update(pickle.load(f).__dict__)
            # print(calibration_object_file_path)
            root.destroy()
            f.close()
        else:
            with open(path, "rb") as f:
                self.__dict__.update(pickle.load(f).__dict__)
