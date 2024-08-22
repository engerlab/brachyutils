import difflib
import logging
import lzma
import os

# trunk-ignore(bandit/B403)
import pickle
import sys
import tkinter as tk
import warnings
from array import array
from tkinter import filedialog as fd
from typing import List, Optional

import numpy as np
import pydicom
import pymedphys
import pyzstd
import SimpleITK as sitk
from matplotlib import pyplot as plt
from numpy import ma, reshape
from scipy.interpolate import RegularGridInterpolator

from opentps.core.data.images import DoseImage
from opentps.core.io.dicomIO import readDicomDose

class BrachyDose:
    r"""
    Purpse:
        This class holds information regarding a dose distribution as well as the fundamental
    functions that are applied on the dose. All doses are expressed in Gy.

    Attributes:
    
        - dose_image:DoseImage := DoseImage object holding the dose grid ([z, y, x]), as well as
            spacing, origin, and rotation ([z, y, x]) information.
        - uncertainty_image:DoseImage := DoseImage object holding the dose uncertainity grid([z, y, x]), as well as
            spacing, origin, and rotation ([z, y, x]) information.
        - voxel_edges:np.ndarray := coorindates of voxel edges along z, y and x axis.
        - interpolation_function := RegularGridInterpolator object that allows for sampling of dose at arbitrary points [z, y, x].

        grid:np.ndarray := 3D numpy array holding dose at each voxel. [z, y, x]
        uncertainty:np.ndarray := 3D numpy array holding dose uncertainity at each voxel. [z, y, x]
        num_voxels:np.ndarray := 1D numpy array holding the number of grid points on x, y, z axis.
        voxel_size:np.ndarray := 1D numpy array holding the resolution of each voxel along x, y, z axis in centimeters.
        origin_coordinates:np.ndarray := The spatial coordinate of the "bottom" left corner of the image in centrimeters. [x, y, z]
        

    Functions:
        load_file_to_brachydose()
        load_from_3ddose()
        load_from_nrrd()
        load_from_npz()
        make_profile()
        make_pdd()
        get_average_uncert()
        get_average_uncert_benchmark()
        pad_3ddose()
        multiply_dose_by_constant()
        write_to_3ddose()
        write_to_nrrd()
        write_to_npz()
        write_to_minidos()
        write_to_xz()
        write_to_zstd()
        calculate_voxel_edges()
        is_equal()
        crop_by_coordinates()
        crop_by_fraction()
        crop_by_index()
        is_not_empty()
        info()

    Dependencies:
        numpy
        re
        os
        glob
        SimpleITK
        difflib
        typing
        collections
        pytest
        lzma
        pickle
        pyzstd
        typer
        tqdm
        DicomRTTool
        pydicom
        json
    """

    def __init__(
        self,
        pth_dose_file: Optional[str] = None,
        load_uncertainty: Optional[bool] = True,
    ):

        self.dose_image: DoseImage = None
        self.uncertainty_image: DoseImage = None
        self.voxel_edges: np.ndarray = None
        # # XXX to delete when you are sure!
        # self.grid: np.ndarray = None
        # self.uncertainty: np.ndarray = None
        # self.num_voxels: np.ndarray = None
        # self.voxel_size: np.ndarray = None
        # self.origin_coordinates: np.ndarray = None
        self.interpolation_function = None
        if pth_dose_file is not None:
            self.load_file_to_brachydose(pth_dose_file, load_uncertainty)
        if self.dose_image is not None:
            self.create_interpolation_function()

    def load_file_to_brachydose(
        self, pth_dose_file: str, load_uncertainty: Optional[bool] = True
    ):
        r"""
        Purpose:
        given the path to a file holding dose information, it will return
        a BrachyDose object with the populated available attributes. It will give a warning
        for the missing attributes.

        Inputs:
            - pth_dose_file := path directory where the file containing the dose is. The file
                extension could be ".3ddose", ".nrrd", ".dcm", or ".minidos"

        Output:
        self : BrachyDose
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
        return self

    def write_brachydose_to_file(self, pth_dose_file: str):
        r"""
        Purpose:
        To write a brachy dose object to the given file path. this function will automatically
        detect the type of the output file and will call the right brachyDose writer function.

        Inputs:
            - pth_dose_file := path where the BrachyDose contents will be written to. The options
            for output type are "3ddose", "nrrd", "npz", "minidos", "xz", and "zstd".

        Output:
            - void := contents of self is written to "pth_dose_file"
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

    def load_from_3ddose(self, filename: str, load_uncertainty: Optional[bool] = True):
        r"""
        Purpose:
            Given the path to a 3ddose file, load its content into self:BrachyDose.

        Input:
            - filename := path to a ".3ddose" file
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
                    self.uncertainty = bench_uncert
                except ValueError:
                    print("Warning: No uncertainty in the 3ddose file", filename, "\n")
            self.dose_image = DoseImage(
                imageArray = bench_dose,
                origin = (bench_x_pos[0], bench_y_pos[0], bench_z_pos[0]),
                spacing = (bench_x_spacing, bench_y_spacing, bench_slice_thick),
            )
            self.uncertainty_image = DoseImage(
                imageArray = bench_uncert,
                origin = (bench_x_pos[0], bench_y_pos[0], bench_z_pos[0]),
                spacing = (bench_x_spacing, bench_y_spacing, bench_slice_thick),
            )
            self.voxel_edges = self.calculate_voxel_edges()

    def load_from_nrrd(self, pth_nrrd: str):
        r"""
        Purpose:
            given the path to a nrrd dose file, it will load its content into self:BrachyDose

        Inputs:
            - pth_nrrd := Path to a nrrd file writtern by self.to_nrrd()

        Dependencies:
            - SimpleITK
            - calculate_voxel_edges()
        """
        loaded_image_nrrd = sitk.ReadImage(pth_nrrd, imageIO="NrrdImageIO")

        dose_uncertainty = sitk.GetArrayFromImage(loaded_image_nrrd)
        if dose_uncertainty.shape[0] == 2:
            dose_array = dose_uncertainty[0]
            uncertainty_array = dose_uncertainty[1]
            self.voxel_size = np.round(
                np.array(loaded_image_nrrd.GetSpacing()[1:]).astype(np.float32), 1
            )
            self.origin_coordinates = np.array(loaded_image_nrrd.GetOrigin()[1:]).astype(
                np.float32
            )
        else:
            dose_array = dose_uncertainty[:, :, :, 0]
            uncertainty_array = dose_uncertainty[:, :, :, 1]
            self.voxel_size = np.round(
                np.array(loaded_image_nrrd.GetSpacing()).astype(np.float32), 1
            )
            self.origin_coordinates = np.array(loaded_image_nrrd.GetOrigin()).astype(np.float32)

        dose_array = np.swapaxes(dose_array, 0, 2)
        uncertainty_array = np.swapaxes(uncertainty_array, 0, 2)

        self.uncertainty = uncertainty_array.astype(np.float32)
        self.grid = dose_array.astype(np.float32)
        self.num_voxels = np.array(np.flip((dose_array.shape), axis=0)).astype(
            np.float32
        )

        self.voxel_edges = self.calculate_voxel_edges()

    def load_from_npz(self, pth_npz):
        r"""
        Purpose:
            Given the path to an npz file, load its content into self:BrachyDose.

        Input:
            - filename := path to a ".npz" file
        """

        assert (
            os.path.splitext(pth_npz)[-1] == ".npz"
        ), "the file extension should be npz"

        loaded_brachydose = np.load(pth_npz, allow_pickle=True)
        self.uncertainty = loaded_brachydose["uncertainty"]
        self.grid = loaded_brachydose["grid"]
        self.num_voxels = loaded_brachydose["num_voxels"]
        self.voxel_size = loaded_brachydose["voxel_size"]
        self.origin_coordinates = loaded_brachydose["origin_coordinates"]
        self.voxel_edges = loaded_brachydose["axis"]

    def load_from_dicom(self, pth_RD_dicom):
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
        assert os.path.basename(pth_RD_dicom).startswith(
            "RD"
        ), "the basename should start with RD"
        # XXX to delete when you are sure!
        # dose_dcm = pydicom.dcmread(pth_RD_dicom)
        # if hasattr(dose_dcm, 'DoseGridScaling'):
        #     print(dose_dcm.DoseGridScaling)
        #     self.grid = dose_dcm.pixel_array.astype(np.float32) * dose_dcm.DoseGridScaling
        # else:
        #     self.grid = dose_dcm.pixel_array.astype(np.float32)
        # self.num_voxels = np.flip(
        #     np.array(dose_dcm.pixel_array.shape, dtype=int), axis=0
        # )
        # x_y_spacing = np.array(dose_dcm.PixelSpacing, dtype=np.float32)
        # z_spacing = np.array(dose_dcm.SliceThickness, dtype=np.float32)
        # if z_spacing == 0:
        #     warnings.warn(
        #         "z_spacing is 0, using x_y_spacing as z_spacing", stacklevel=2
        #     )
        #     z_spacing = x_y_spacing[0]

        # self.voxel_size = np.append(x_y_spacing, z_spacing)
        # self.origin_coordinates = np.array(dose_dcm.ImagePositionPatient, dtype=np.float32)
        dose_image_xyz = readDicomDose(pth_RD_dicom)
        self.dose_image = DoseImage(
            imageArray = np.swapaxes(dose_image_xyz.imageArray, 0, 2),
            origin = np.flip(dose_image_xyz.origin),
            spacing = np.flip(dose_image_xyz.spacing),
            name = dose_image_xyz.name,
            angles = np.flip(dose_image_xyz.angles),
            seriesInstanceUID=dose_image_xyz.seriesInstanceUID,
            sopInstanceUID=dose_image_xyz.sopInstanceUID,
        )
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

    def create_interpolation_function(self):
        r"""
        Purpose:
            To create an interpolation function for the dose grid.
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
            Documentation is missing
        """
        max_dose = self.grid.max()
        dose_mask = self.grid < 0.2 * max_dose
        masked_uncert = ma.array(self.uncert, mask=dose_mask)
        masked_dose = ma.array(self.grid, mask=dose_mask)
        average_uncert = ma.average(masked_uncert / masked_dose) * 100
        return average_uncert

    def get_average_uncert_benchmark(self) -> float:
        r"""
        Purpose:
            Documentation is missing
        """
        max_dose = self.grid.max()
        dose_mask = self.grid < 0.2 * max_dose
        masked_uncert = ma.array(self.uncert, mask=dose_mask)
        average_uncert = ma.average(masked_uncert) * 100
        return average_uncert

    def pad_3ddose(self, new_dims: list, new_top_left: list):
        r"""a function to padd the grid and uncertainty in BrachyDose object and bring it to the desired dimensios.
        it will update all the aspects of the dose object to match the new dimensiosn.
        The voxels must have the same size! remember, python does z, y, x.
        inputs:
            self:BrachyDose

            new_dims := a 1 by 3 list containing the new x, y and z dimensions:
                [new_z_dim, new_y_dim, new_x_dim]

            new_top_left := coordinates of the new origin_coordinates
                [x, y, z]
        """
        assert any(
            new_dims > self.grid.shape
        ), "since you are padding, the new dimensions should be larger than the input dimensions"

        # calculate distances between the new and old origin_coordinates voxels.
        # if for an axis, the distance of origin_coordinatess is larger than the voxel size, use the new origin_coordinates
        # else, use the old top left
        origin_coordinates_distance = np.abs(new_top_left - self.origin_coordinates)
        final_origin_coordinates = np.zeros(3)
        for i, distance in zip(range(3), origin_coordinates_distance):
            final_origin_coordinates[i] = (
                new_top_left[i] if distance > self.voxel_size[i] else self.origin_coordinates[i]
            )

        # figure out how much padding to do before and after each axis
        padding = np.zeros([3, 2])
        for i in range(3):
            if final_origin_coordinates[i] == self.origin_coordinates[i]:
                # all padding goes to the end for this dose axis
                pad_before = 0
                pad_after = new_dims[2 - i] - self.grid.shape[2 - i]
            else:
                # all padding goes to the begining of the dose axis
                pad_before = new_dims[2 - i] - self.grid.shape[2 - i]
                pad_after = 0
            padding[2 - i] = [pad_before, pad_after]

        # pad the old dose grid to get the new grid!
        new_dose_grid = np.pad(self.grid, tuple(padding.astype(int)), mode="edge")
        if self.uncertainty is not None:
            new_uncert = np.pad(
                self.uncertainty, tuple(padding.astype(int)), mode="edge"
            )

        # figure out the end coordinates based on the padding
        # self.voxel_size is a list of x, y and z spacing, we want it to be
        # a numpy array of z, y, x spacings.
        voxel_size = np.array(self.voxel_size)[:, np.newaxis][::-1]
        end_coords_distances = (
            padding * np.array([[-1, 1], [-1, 1], [-1, 1]]) * voxel_size
        )

        old_end_coords = np.array(
            [
                [self.voxel_edges[0][0], self.voxel_edges[0][-1]],
                [self.voxel_edges[1][0], self.voxel_edges[1][-1]],
                [self.voxel_edges[2][0], self.voxel_edges[2][-1]],
            ]
        )

        new_end_coords = old_end_coords + end_coords_distances

        # now padd the new axis with respect to the appropriate begin and end coordinates
        new_axis = np.array(
            [np.zeros(new_dims[0]), np.zeros(new_dims[1]), np.zeros(new_dims[2])],
            dtype=object,
        )

        # pad the new axis with linear ramp
        for i in range(new_axis.shape[0]):
            new_axis[i] = np.pad(
                self.voxel_edges[i],
                tuple(padding[i].astype(int)),
                mode="linear_ramp",
                end_values=new_end_coords[i],
            )

        # fillout the new padded dose dictionary
        padded_dose = BrachyDose()

        padded_dose.grid = new_dose_grid
        padded_dose.uncert = new_uncert if self.uncertainty is not None else None
        # voxel size remains unchanged
        padded_dose.voxel_size = self.voxel_size
        padded_dose.origin_coordinates = final_origin_coordinates
        padded_dose.voxel_edges = new_axis

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

        dimensions = " ".join(map(str, self.num_voxels.astype(int))) + "\n"
        x_axis = " ".join(map(str, self.voxel_edges[2])) + "\n"
        y_axis = " ".join(map(str, self.voxel_edges[1])) + "\n"
        z_axis = " ".join(map(str, self.voxel_edges[0])) + "\n"
        dose_flattened = " ".join(map(str, self.grid.flatten("C"))) + "\n"
        if self.uncertainty is not None:
            uncertainty_flattened = (
                " ".join(map(str, self.uncertainty.flatten("C"))) + "\n"
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

    def write_to_nrrd(self, file_name: str, metadata: Optional[dict] = None):
        r"""
        Purpose:
            To save the contents of BrachyDose into a nrrd file.
        inputs:
            - file_name := path where the dose nrrd file will be written to.

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
        # create sitk dose image
        dose_nda = np.swapaxes(self.grid, 0, 2).astype(np.float32)
        try:
            uncertainty_nda = np.swapaxes(self.uncertainty, 0, 2).astype(np.float32)
        except AttributeError:
            uncertainty_nda = np.zeros(dose_nda.shape, dtype=np.float32)
        # # old nrrd format
        # image_nrrd = sitk.JoinSeries(
        #     sitk.GetImageFromArray(dose_nda), sitk.GetImageFromArray(uncertainty_nda)
        # )
        # # new nrrd format
        fiter = sitk.ComposeImageFilter()
        image_nrrd = fiter.Execute(
            sitk.GetImageFromArray(dose_nda), sitk.GetImageFromArray(uncertainty_nda)
        )
        # image_nrrd.SetOrigin(np.append([0], self.origin_coordinates))
        # image_nrrd.SetSpacing(np.append([1], self.voxel_size))
        image_nrrd.SetOrigin(self.origin_coordinates.astype(float))
        image_nrrd.SetSpacing(self.voxel_size.astype(float))

        # set the metadata: all sitk Images belonging to a patient will have the same meta data
        if metadata is not None:
            for key in metadata:
                image_nrrd.SetMetaData(key, metadata[key])

        # write out the files
        file_name_ospth = os.path.abspath(file_name)
        assert os.path.exists(
            os.path.dirname(file_name_ospth)
        ), f"the input folder does not exist: {os.path.dirname(file_name_ospth)}"

        run_number = file_name_ospth.split(".")[0]

        sitk.WriteImage(
            image_nrrd, run_number + ".nrrd", useCompression=True, compressionLevel=9
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

        assert os.path.splitext(file_name)[-1] == ".npz"

        np.savez_compressed(
            file=file_name,
            grid=self.grid,
            uncertainty=self.uncertainty,
            num_voxels=self.num_voxels,
            voxel_size=self.voxel_size,
            origin_coordinates=self.origin_coordinates,
            axis=self.voxel_edges,
        )

    def write_to_minidos(self, file_name: str):
        r"""
        Purpose:
            To save the contents of BrachyDose into a minidos file, which is just a binary file written line by line.
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
            dims_array = array("i", self.num_voxels)
            dims_array.tofile(newfile)

            # lines 2,3 and 4 are the voxel sizes x, y, z
            float_array_x = array("f", [self.voxel_size[0]])
            float_array_x.tofile(newfile)
            float_array_y = array("f", [self.voxel_size[1]])
            float_array_y.tofile(newfile)
            float_array_z = array("f", [self.voxel_size[2]])
            float_array_z.tofile(newfile)

            # lines 5, 6, 7 are the origins x, y, and z
            originx_array = array("f", [self.origin_coordinates[0]])
            originy_array = array("f", [self.origin_coordinates[1]])
            originz_array = array("f", [self.origin_coordinates[2]])

            originx_array.tofile(newfile)
            originy_array.tofile(newfile)
            originz_array.tofile(newfile)

            # lines 8 is just a zero
            zero = array("i", [0])
            zero.tofile(newfile)

            # line 9-infinit is the dose per voxel array
            for d in self.grid.flatten():
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
        Purpose: will calculate the axes coordinates for a 3ddose dictionary.

        Input:
            - self:BrachyDose

        Output:
            - axes:numpy.array() :=
            [[z_min:voxel_size:z_max],
            [y_min:voxel_size:y_max],
            [x_min:voxel_size:x_max]]
        """
        assert self.dose_image is not None, "dose image is not defined. please load a dose image first"
        axes_end = np.array(
            self.dose_image.origin
            + self.dose_image.spacing * self.dose_image.gridSize
            # one voxel size is added because np.arange stops at an index before the end
            + self.dose_image.spacing
        )
        self.voxel_edges = np.empty(len(axes_end), dtype=object)
        for i in range(len(axes_end)):
            self.voxel_edges[i] = np.arange(
                self.dose_image.origin[i],#[len(axes_end) - 1 - i],
                axes_end[i],#[len(axes_end) - 1 - i],
                self.dose_image.spacing[i],#[len(axes_end) - 1 - i],
                dtype=np.float32,
            )
            if np.absolute(
                self.dose_image.gridSize[::-1][i]
                - self.voxel_edges[i].shape[0]
                ) > 1:
                self.voxel_edges[i] = self.voxel_edges[i][:-1]

        return self.voxel_edges

    def get_voxel_centers(self):
        voxel_centers = np.empty(len(self.voxel_edges), dtype=object)
        if self.voxel_edges is not None:
            for i in range(len(self.voxel_edges)):
                voxel_centers[i] = self.voxel_edges[i] + self.dose_image.spacing[i] / 2.0
                # voxel_centers[i] = voxel_centers[i][:-1]
        else:
            raise ValueError("Voxel edges are not calculated yet")
        return voxel_centers

    def is_equal(self, new_brachy_dose):
        r"""
        Purpose:
            To compare if self:BrachyDose has the same attributes as an input BrachyDose

        Inputs:
            - new_brachy_dose: another BrachyDose object whose attributes may or may not contain equal info as the attributes of self.

        Outputs:
            True if attributes of new_brachy_dose are the same as self
            False otherwise
        """
        assert isinstance(
            new_brachy_dose, BrachyDose
        ), "input must be of type BrachyDose"
        assert np.array_equal(
            self.grid, new_brachy_dose.grid
        ), "dose grid is not the same"
        assert np.array_equal(
            np.concatenate(self.voxel_edges),
            np.concatenate(new_brachy_dose.voxel_edges),
        ), "axis is not the same"
        assert np.array_equal(
            self.uncertainty, new_brachy_dose.uncertainty
        ), "uncertainty is not the same"
        assert np.array_equal(
            self.num_voxels, new_brachy_dose.num_voxels
        ), "num_voxels is not the same"
        assert np.array_equal(
            self.voxel_size, new_brachy_dose.voxel_size
        ), "voxel_size is not the same"
        assert np.array_equal(
            self.origin_coordinates, new_brachy_dose.origin_coordinates
        ), "origin_coordinates is not the same"
        # assert np.array_equal(np.round(self.voxel_size, 2), np.round(new_brachy_dose.voxel_size, 2)), "voxel_size is not the same"
        # assert np.array_equal(np.round(self.origin_coordinates, 2), np.round(new_brachy_dose.origin_coordinates), 2), "origin_coordinates is not the same"

        return (
            np.array_equal(self.grid, new_brachy_dose.grid)
            and np.array_equal(
                np.concatenate(self.voxel_edges),
                np.concatenate(new_brachy_dose.voxel_edges),
            )
            and np.array_equal(self.uncertainty, new_brachy_dose.uncertainty)
            and np.array_equal(self.num_voxels, new_brachy_dose.num_voxels)
            and np.array_equal(self.voxel_size, new_brachy_dose.voxel_size)
            and np.array_equal(self.origin_coordinates, new_brachy_dose.origin_coordinates)
        )
        # and np.array_equal(np.round(self.voxel_size, 2), np.round(new_brachy_dose.voxel_size, 2)) \
        # and np.array_equal(np.round(self.origin_coordinates, 2), np.round(new_brachy_dose.origin_coordinates), 2)
        # np.array_equal(np.round(np.concatenate(self.voxel_edges), 2), np.concatenate(new_brachy_dose.voxel_edges)) \

    def crop_by_coordinates(
        self, coord_range: np.array, inplace: Optional[bool] = True
    ):
        r"""
        Purpose:
            given a range of coordinate (mix and max on each axis), this function will crop
            dose and uncertainty maps and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - coord_range := a 3 x 2 array holding the min and max on x, y and axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the coordinate range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            -self.crop_by_index()
        """
        self.is_not_empty()

        # make sure the ending coordinate of the new range is larger than its origin
        for ax in coord_range:
            assert (
                ax[1] > ax[0]
            ), "axis ending coordinate should be larger than its begining coordinate"

        # assert coordinate range falls into the image
        axes_name = ["x axis(2)", "y axis(1)", "z axis(0)"]
        coords_flipped = np.flip(coord_range, 0)
        for i in range(self.voxel_edges.shape[0]):
            assert (
                coords_flipped[i][0] > self.voxel_edges[i][0]
            ), f"cropping lower limit must be larger than the lowest coordinate on {axes_name[i]}"

        # convert new coordinates to indicies for both beggingn and ending (may not be exact)
        new_origin_distance = coord_range[:, 0] - self.origin_coordinates

        new_origin_index = np.floor(new_origin_distance / self.voxel_size).astype(int)

        new_ending_index = np.floor(
            (coord_range[:, 1] - self.origin_coordinates) / self.voxel_size
        ).astype(int)

        new_index_range = np.column_stack([new_origin_index, new_ending_index])

        return self.crop_by_index(new_index_range, inplace)

    def crop_by_fraction(self, crop_fraction, inplace: Optional[bool] = True):
        r"""
        Purpose:
            given the crop_fraction, this function will crop out 0.5*(1 - crop_fraction)*dimension voxels
                from the edges of the x and y axis of dose and uncertainty maps and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - crop_fraction := a floating point between 0 and 1, which is the fraction of the image axis that remains in the crop.
                for example, a crop ratio of 0.5 will keep the center of the x and y axis, plus minus 0.25*dimension of the image.
                The z axis will not be cropped.
                    +++++++++       ---------
                    +++++++++       --+++++--
                    +++++++++  ===> --+++++--
                    +++++++++       --+++++--
                    +++++++++       ---------
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the coordinate range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            -self.crop_by_index()
        """
        assert (
            crop_fraction >= 0 and crop_fraction <= 1
        ), "the fraction should be between 0 and 1"
        off_set = 0.5 * (1 - crop_fraction) * self.num_voxels
        new_origin_index = np.array([off_set[0], off_set[1], 0]).astype(int)
        assert np.all(
            new_origin_index >= 0
        ), "new origin index cannot be negative, please report this bug"

        new_ending_index = np.array(
            [
                self.num_voxels[0] - off_set[0],
                self.num_voxels[1] - off_set[1],
                self.num_voxels[2],
            ]
        ).astype(int)
        assert np.all(
            new_ending_index >= 0
        ), "new ending index cannot be negative, please report this bug"

        new_index_range = np.column_stack([new_origin_index, new_ending_index])

        return self.crop_by_index(new_index_range, inplace)

    def crop_by_index(self, index_range: np.array, inplace: Optional[bool] = True):
        r"""
        Purpose:
            given a range of indicies (mix and max on each axis), this function will crop
            dose and uncertainty maps and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - index_range := a 3 x 2 array holding the min and max on x, y and axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the index range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            - None
        """
        new_origin_index = index_range[:, 0].astype(int)
        assert np.all(
            new_origin_index >= 0
        ), "new origin index cannot be negative, please report this bug"

        new_ending_index = index_range[:, 1].astype(int)
        assert np.all(
            new_ending_index >= 0
        ), "new ending index cannot be negative, please report this bug"

        # update the attributes
        if inplace:
            self.grid = self.grid[
                new_origin_index[2] : new_ending_index[2],  # z
                new_origin_index[1] : new_ending_index[1],  # y
                new_origin_index[0] : new_ending_index[0],  # x
            ]
            self.uncertainty = self.uncertainty[
                new_origin_index[2] : new_ending_index[2],  # z
                new_origin_index[1] : new_ending_index[1],  # y
                new_origin_index[0] : new_ending_index[0],  # x
            ]
            self.origin_coordinates = np.array(
                [
                    self.voxel_edges[2][new_origin_index[0]],  # x
                    self.voxel_edges[1][new_origin_index[1]],  # y
                    self.voxel_edges[0][new_origin_index[2]],  # z
                ]
            )
            self.num_voxels = np.flip(self.grid.shape, 0)
            self.voxel_edges = self.calculate_voxel_edges()
        else:
            new_dose_obj = BrachyDose()
            new_dose_obj.grid = self.grid[
                new_origin_index[2] : new_ending_index[2],
                new_origin_index[1] : new_ending_index[1],
                new_origin_index[0] : new_ending_index[0],
            ]
            new_dose_obj.uncertainty = self.uncertainty[
                new_origin_index[2] : new_ending_index[2],
                new_origin_index[1] : new_ending_index[1],
                new_origin_index[0] : new_ending_index[0],
            ]
            new_dose_obj.origin_coordinates = np.array(
                [
                    self.voxel_edges[2][new_origin_index[0]],  # x
                    self.voxel_edges[1][new_origin_index[1]],  # y
                    self.voxel_edges[0][new_origin_index[2]],  # z
                ]
            )
            new_dose_obj.num_voxels = np.flip(self.grid.shape, 0)
            new_dose_obj.voxel_size = self.voxel_size
            new_dose_obj.voxel_edges = self.calculate_voxel_edges()
            return new_dose_obj

    def is_not_empty(self):
        assert self.dose_image is not None, "error dose image is None"
        assert self.voxel_edges is not None, "error axis is None"
        return True

    def info(self):
        self.is_not_empty()
        print(f"shape of dose grid is: {self.grid.shape}")
        if self.uncertainty is not None:
            print(f"shape of uncertainty matrix is: {self.uncertainty.shape}")
        print(f"num voxels attribute is: {self.num_voxels}")
        print(f"the top left (bottom left in reality) is {self.origin_coordinates}")
        print(f"the voxel size is {self.voxel_size}")
        print(
            f"the size of the z, y and x axes are {self.voxel_edges[0].shape, self.voxel_edges[1].shape, self.voxel_edges[2].shape}"
        )
        print(
            f"the range of the z axis is {self.voxel_edges[0][0], self.voxel_edges[0][-1]}"
        )
        print(
            f"the range of the y axis is {self.voxel_edges[1][0], self.voxel_edges[1][-1]}"
        )
        print(
            f"the range of the x axis is {self.voxel_edges[2][0], self.voxel_edges[2][-1]}"
        )

    def crop_by_body_contour(
        self,
        body_index_range: Optional[np.ndarray] = None,
        body_mask_shape: Optional[np.ndarray] = None,
        pth_dir_dicom: Optional[str] = None,
    ):
        r"""
        Purpose:
            based on the given dicom structure file, crop the BrachyDose object such
                that it only has the body contour.
        Inputs:
            - pth_dir_dicom := pth_dir_dicom := path to the directory with the dicom files of a patient.
                it should contain both images and RTSTRUCT file. this input is optional

            - body_index_range:np.array :=  a 3 x 2 array holding the min and max on x, y and axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]],

            - original_mask_dimensions:np.array := 1 x 3 array holding the dimension of the original mask


        Outputs:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the body contour
                    in the dicom structure file. It will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        """
        from dicom_utils import BrachyDicom

        if body_index_range is None or body_mask_shape is None:
            assert (
                pth_dir_dicom is not None
            ), "Either path to a dicom directory with dicom structure \
                file should be given or body_index_range and body_mask_shape"
            body_mask_shape = BrachyDicom(pth_dir_dicom).structure_index_range_dict[
                "body"
            ][
                "dicom_mask_shape"
            ]  # the body mask may have a different size than the dose map, we normalize range to the dimension
        # of original mask and scale it to the dimension of the dose map to get the body index range on the dose image.
        scaled_body_index_range = (
            body_index_range
            / np.expand_dims(body_mask_shape, axis=1)
            * np.expand_dims(self.num_voxels, axis=1)
        ).astype(int)

        self.crop_by_index(scaled_body_index_range, True)

    def multiply_dose_by_constant(
        self, scale_factor: float, scale_uncert: Optional[bool] = False
    ):
        r"""
        Purpose:
            scale the dose and uncertainty maps by a constant factor.
        Inputs:
            - scale_factor := a floating point number that the dose and uncertainty maps will be scaled by.
        Outputs:
            - Void := will scale the dose and uncertainty maps of self by the scale factor.
        """
        self.is_not_empty()
        self.grid *= scale_factor
        if scale_uncert and self.uncertainty is not None:
            self.uncertainty *= scale_factor


def dose_with_empty_grid_like(doseObj: BrachyDose):
    r"""
    Purpose:
        To create a new dose object with the same attributes as the input dose object,
        but with an empty grid and uncertainty.

    Inputs:
        doseObj: BrachyDose object

    Outputs:
        empty_dose: BrachyDose object with empty grid and uncertainty
    """
    new_dose = BrachyDose()
    new_dose.grid = np.zeros_like(doseObj.grid)
    new_dose.uncertainty = np.zeros_like(doseObj.grid)
    new_dose.num_voxels = doseObj.num_voxels
    new_dose.voxel_size = doseObj.voxel_size
    new_dose.origin_coordinates = doseObj.origin_coordinates
    new_dose.voxel_edges = doseObj.voxel_edges
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
