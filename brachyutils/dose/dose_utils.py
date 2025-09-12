import copy
import difflib

# import logging
import lzma
import os

# trunk-ignore(bandit/B403)
# import sys
import warnings
from array import array

# from glob import glob
from pathlib import Path
from typing import List, Literal, Optional, Union

import nrrd
import nrrd.reader
import numpy as np
import pyzstd
from numpy import ma, reshape
from opentps.core.data.images import DoseImage
from scipy.interpolate import RegularGridInterpolator

from brachyutils.geometry import BrachyPhantom


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
        self.unit_length: Literal["mm"] = "mm"
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
        if self.dose_image is None:
            raise ValueError("dose image not loaded")
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

        elif file_extension == ".dcm":
            self.write_to_dicom(pth_dose_file)

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

    def load_from_3ddose(
        self, filename: Path, load_uncertainty: Optional[bool] = True
    ) -> None:
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
            bench_x_pos = (
                np.round(
                    np.array(newfile.readline().split(), dtype=np.float32), decimals=6
                )
                * 10
            )
            bench_y_pos = (
                np.round(
                    np.array(newfile.readline().split(), dtype=np.float32), decimals=6
                )
                * 10
            )
            bench_z_pos = (
                np.round(
                    np.array(newfile.readline().split(), dtype=np.float32), decimals=6
                )
                * 10
            )

            bench_spacing = np.array(
                [
                    bench_x_pos[1] - bench_x_pos[0],
                    bench_y_pos[1] - bench_y_pos[0],
                    bench_z_pos[1] - bench_z_pos[0],
                ]
            )
            bench_origin = np.array(
                [
                    bench_x_pos[0] + bench_spacing[0] / 2,
                    bench_y_pos[0] + bench_spacing[1] / 2,
                    bench_z_pos[0] + bench_spacing[2] / 2,
                ],
                dtype=np.float32,
            )

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
                        imageArray=np.swapaxes(bench_uncert, 0, 2),
                        origin=bench_origin,
                        spacing=bench_spacing,
                    )
                except ValueError:
                    print("Warning: No uncertainty in the 3ddose file", filename, "\n")

            self.dose_image = DoseImage(
                # convert numpy zyx to xyz in opentps.
                imageArray=np.swapaxes(bench_dose, 0, 2),
                origin=bench_origin,
                spacing=bench_spacing,
            )
            self.voxel_edges = self.get_voxel_edges()

    def load_from_nrrd(self, pth_nrrd: Path) -> None:
        r"""
        Purpose:
            - given the path to a .nrrd dose file, it will load its content into self:BrachyDose
        Inputs:
            - pth_nrrd := Path to a .nrrd file writtern. by default, we assume c index ordering is done.
        Outputs:
            - void := contents of self is updated.
        Dependencies:
            - nrrd
            - get_voxel_edges()
        """
        # nrrd.reader._READ_CHUNKSIZE = 500 * 1024 * 1024  # 500MB
        dose_uncertainty, header = nrrd.read(pth_nrrd, index_order="C")

        if dose_uncertainty.shape[0] == 2:
            dose_array = dose_uncertainty[0]
            uncertainty_array = dose_uncertainty[1]
        elif dose_uncertainty.shape[-1] == 2:
            dose_array = dose_uncertainty[:, :, :, 0]
            uncertainty_array = dose_uncertainty[:, :, :, 1]
            affine = header.get("space directions")[1:]
        else:
            print("Uncertainty not found in the nrrd file")
            dose_array = dose_uncertainty
            uncertainty_array = None
            affine = header.get("space directions")

        voxel_size = affine.diagonal()
        origin_coordinates = np.array(header.get("space origin")).astype(np.float32)

        self.dose_image = DoseImage(
            # imageArray=np.swapaxes(dose_array, 0, 2),
            origin=origin_coordinates,
            spacing=voxel_size,
        )
        self.set_dose_array(dose_array)
        self.uncertainty_image = (
            DoseImage(
                # imageArray=np.swapaxes(uncertainty_array, 0, 2),
                origin=origin_coordinates,
                spacing=voxel_size,
            )
            if uncertainty_array is not None
            else None
        )
        if self.uncertainty_image is not None:
            self.set_uncertainty_array(uncertainty_array)
        self.voxel_edges = self.get_voxel_edges()

    def load_from_npz(self, pth_npz: Path) -> None:
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
        self.get_voxel_edges()
        self.create_interpolation_function()

    def load_from_dicom(self, pth_RD_dicom: Path):
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
        if self.dose_image is None:
            warnings.warn("No dose image found in the dicom file", stacklevel=2)
            return
        if self.dose_image.spacing[2] == 0:
            if self.dose_image.spacing[0] == self.dose_image.spacing[1]:
                self.dose_image.spacing[2] = self.dose_image.spacing[0]
            else:
                self.dose_image.spacing[2] = 1.0
                warnings.warn(
                    "The z spacing is not defined in the dicom file and the x and y spacing are not the same. Z spacing is set to 1mm.",
                    stacklevel=2,
                )

        self.voxel_edges = self.get_voxel_edges()

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
        # r""" """
        # raise DeprecationWarning(
        #    "This function is no longer supported due to migration to open tps. please use self.get_dose_at_coordinates() instead."
        # )
        # self.is_not_empty()
        # if self.interpolation_function is None:
        #    raise ValueError("interpolation function is not defined")
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
        # trunk-ignore(ruff/E731)
        dose_grid_lambda = lambda xs, ys, zs: self.dose_image.getDataAtPosition(
            (xs, ys, zs)
        )
        dose_grid_function = np.vectorize(dose_grid_lambda)
        dose_grid = dose_grid_function(coord_grid_x, coord_grid_y, coord_grid_z)
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

    def get_average_uncertainty(self, mask: Optional[np.ndarray] = None) -> float:
        r"""
        Purpose:
            - To calculate the average uncertainty normalized by dose
            in an optinal mask. The unit of uncertainty is in percentage.
        Inputs:
            - self:BrachyDose
            - mask: Optional[np.ndarray] := a mask to apply to the dose grid in [z, y, x] format.
        Outputs:
            - average_uncert:float := the average uncertainty in the dose grid in percentage.
        """
        # max_dose = self.dose_image.imageArray.max()
        # dose_mask = self.dose_image.imageArray < 0.2 * max_dose
        masked_uncert = ma.array(self.get_uncertainty_array(), mask=mask)
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
        raise NotImplementedError(
            "This function is not needed. use resample from opentps instead!"
        )
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
                new_origin[i]
                if distance > self.dose_image.spacing[i]
                else self.dose_image.origin[i]
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
        new_dose_grid = np.pad(
            self.dose_image.imageArray, tuple(padding.astype(int)), mode="edge"
        )
        if self.uncertainty_image is not None:
            new_uncert = np.pad(
                self.uncertainty_image.imageArray,
                tuple(padding.astype(int)),
                mode="edge",
            )
        # fillout the new padded dose dictionary
        padded_dose = BrachyDose()
        padded_dose.dose_image = DoseImage(
            imageArray=new_dose_grid,
            origin=final_origin_coordinates,
            spacing=self.dose_image.spacing,
        )
        if self.uncertainty_image is not None:
            padded_dose.uncertainty_image = DoseImage(
                imageArray=new_uncert,
                origin=final_origin_coordinates,
                spacing=self.dose_image.spacing,
            )
        self.get_voxel_edges()
        self.create_interpolation_function()
        return padded_dose

    def write_to_3ddose(self, file_name: str):
        r"""
        Purpose:
            This function will write the contents of a BrachyDose onto a text file with .3ddose extension.

        inputs:
            - self := a BrachyDose object containing the following keys:
                grid [x, y, z]
                uncert [x, y, z]
                voxel_size [x, y, z]
                origin_coordinates [x, y, z]
                axis [x, y, z]

            - file_name := the directory path where the file will be written
        """
        file_name = os.path.abspath(file_name)

        # dimensions = " ".join(map(str, np.flip(self.dose_image.gridSize.astype(int)))) + "\n"
        dimensions = " ".join(map(str, self.dose_image.gridSize.astype(int))) + "\n"
        x_axis = " ".join(map(str, (-1 * self.voxel_edges[0]) / 10)) + "\n"
        y_axis = " ".join(map(str, self.voxel_edges[1] / 10)) + "\n"
        z_axis = " ".join(map(str, self.voxel_edges[2] / 10)) + "\n"
        arr_flat = self.get_dose_array().flatten("C")
        formatted_str_array = np.char.mod(f"%.6f", arr_flat)
        dose_flattened = " ".join(formatted_str_array) + "\n"
        if self.uncertainty_image is not None:
            arr_flat = self.get_uncertainty_array().flatten("C")
            formatted_str_array = np.char.mod(f"%.6f", arr_flat)
            uncertainty_flattened = (
                " ".join(formatted_str_array) + "\n"
            )
        else:
            uncertainty_flattened = " ".join(np.ones_like(formatted_str_array)) + "\n"

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
        self,
        pth_output: Path,
        metadata: Optional[dict] = None,
        anatomical_coordinate_system: Literal[
            "LPS", "RAS"
        ] = "LPS",
    ):
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
            - anatomical_coordinate_system := the coordinate system of the dose grid. should be one of the following:
                "left-posterior-superior"
                "right-anterior-superior"
        outputs: Void
            writes [3D dose, 3D uncertainty], voxel size, origin (origin_coordinates), and metadata to the file_name_dose.nrrd
        """
        # check if the directory exists, if not create it. make sure the file extension is write.
        Path.mkdir(pth_output.parent, exist_ok=True)
        assert (
            str(pth_output).endswith(".nrrd")
        ), "the file should have '.nrrd' extension"
        
        from collections import defaultdict
        header = defaultdict(str)
        header = header | metadata if metadata is not None else header
        # # Common metadata
        # header["type"] = "double"
        header["space"] = "left-posterior-superior" if anatomical_coordinate_system == "LPS" else "right-anterior-superior"
        # header["endian"] = "little"
        header["encoding"] = "gzip"
        dose_array = self.get_dose_array()
        # make a dummy uncertainty array and write it to 1
        if self.uncertainty_image is None:
            uncertainty_array = np.ones_like(dose_array, dtype=np.float32)
        else:
            uncertainty_array = self.get_uncertainty_array()
        header["dimension"] = "4"
        header["kinds"] = ["list", "domain", "domain", "domain"]
        header["space origin"] = self.dose_image.origin.tolist()
        header["space directions"] = [
            [np.nan, np.nan, np.nan],
            [self.dose_image.spacing[0], 0.0, 0.0],
            [0.0, self.dose_image.spacing[1], 0.0],
            [0.0, 0.0, self.dose_image.spacing[2]],
        ]
        # header["spacing"] = [np.nan] + self.dose_image.spacing.tolist()
        # header["space units"] = ["None", "mm", "mm", "mm"]
        dose_uncertainty_array = np.stack([dose_array, uncertainty_array], axis=3)
        nrrd.write(str(pth_output), dose_uncertainty_array, header, index_order="C", compression_level=1)

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

        assert (
            os.path.splitext(file_name)[-1] == ".npz"
        ), "the file name should have '.npz' extension."

        np.savez_compressed(
            file=file_name,
            dose_image=self.dose_image,
            uncertainty_image=(
                self.uncertainty_image if self.uncertainty_image is not None else None
            ),
            axis=self.voxel_edges,
        )

    def write_to_xz(self, fileName):
        assert os.path.splitext(fileName)[-1] == ".xz"
        import pickle

        with lzma.open(fileName, "wb") as file:
            pickle.dump(self, file)

    def write_to_zstd(self, file_name):
        assert os.path.splitext(file_name)[-1] == ".zst"
        import pickle

        with pyzstd.open(file_name, "wb", level_or_option=22) as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def write_to_dicom(self, filename:Path | str):
        from opentps.core.io.dicomIO import writeRTDose
        filename=Path(filename)
        writeRTDose(self.dose_image, str(filename.parent), str(filename.name))

    def get_voxel_edges(self):
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
        assert (
            self.dose_image is not None
        ), "dose image is not defined. please load a dose image first"
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
        assert (
            self.dose_image is not None
        ), "dose image is not defined. please load a dose image first"
        voxel_centers = np.empty(len(self.dose_image.origin), dtype=object)
        for i in range(len(self.dose_image.origin)):
            voxel_centers[i] = (
                self.dose_image.origin[i]
                + np.arange(self.dose_image.gridSize[i]) * self.dose_image.spacing[i]
            )
        return voxel_centers

    def get_dose_at_coordinates(self, coords: Union[np.ndarray, List[float]], uncertainty = False) -> float:
        r"""
        Purpose:
            - Given a set of coordinates, this function will return the dose at that point.
        Inputs:
            - coords := a list of 3 coordinates [x, y, z] or a numpy array of shape (3,)
        Outputs:
            - dose := the dose at the given coordinates in Gy
        """
        assert len(coords) == 3, "coords should be a list of 3 coordinates"
        if uncertainty:
            assert self.uncertainty_image is not None, "uncertainty image is not defined"
            return self.uncertainty_image.getDataAtPosition(coords)
        else:
            return self.dose_image.getDataAtPosition(coords)

    def get_uncertainty_at_coordinates(
        self, coords: Union[np.ndarray, List[float]]
    ) -> float:
        r"""
        Purpose:
            - Given a set of coordinates, this function will return the dose at that point.
        Inputs:
            - coords := a list of 3 coordinates [x, y, z] or a numpy array of shape (3,)
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
        elif not np.all(
            np.isclose(
                self.dose_image.imageArray,
                new_brachy_dose.dose_image.imageArray,
                atol=1e-6,
            )
        ):
            warnings.warn("dose values are not the same", stacklevel=2)
            return False
        elif not np.all(
            np.isclose(
                np.concatenate(self.voxel_edges),
                np.concatenate(new_brachy_dose.voxel_edges),
                atol=1e-3,
            )
        ):
            warnings.warn("axis is not the same", stacklevel=2)
            return False
        elif not self.uncertainty_image is not None:
            if np.all(
                np.isclose(
                    self.uncertainty_image.imageArray,
                    new_brachy_dose.uncertainty_image.imageArray,
                    atol=1e-6,
                )
            ):
                warnings.warn("uncertainty is not the same", stacklevel=2)
                return False
        elif not np.array_equal(
            self.dose_image.gridSize, new_brachy_dose.dose_image.gridSize
        ):
            warnings.warn("num_voxels is not the same", stacklevel=2)
            return False
        elif not np.all(
            np.isclose(
                self.dose_image.spacing, new_brachy_dose.dose_image.spacing, atol=1e-3
            )
        ):
            warnings.warn("voxel_size is not the same", stacklevel=2)
            return False
        elif not np.all(
            np.isclose(
                self.dose_image.origin, new_brachy_dose.dose_image.origin, atol=1e-3
            )
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
                [[x_min, x_max], [y_min, y_max], [z_min, z_max],]
        Output:
            - Void := will crop out the dose and uncertainty maps of self to have the range of the coordinate range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            opentps.core.processing.imageProcessing.resampler3D.crop3DDataAroundBox
        """
        from opentps.core.processing.imageProcessing.resampler3D import (
            crop3DDataAroundBox,
        )

        self.is_not_empty()
        assert coordinate_range.shape == (
            3,
            2,
        ), "coordinate_range should be a 3x2 array in x, y, z order"
        if inplace:
            crop3DDataAroundBox(self.dose_image, coordinate_range)
            if self.uncertainty_image is not None:
                crop3DDataAroundBox(self.uncertainty_image, coordinate_range)
            self.get_voxel_edges()
            self.create_interpolation_function()
            self.get_voxel_edges()
        else:
            new_dose: BrachyDose = copy.deepcopy(self)
            new_dose.crop_by_coordinates(coordinate_range, inplace=True)
            return new_dose

    def crop_by_fraction(
        self, crop_fraction: List[float], inplace: Optional[bool] = True
    ) -> Union[None, "BrachyDose"]:
        r"""
        Purpose:
            - given the crop_fraction, this function will crop out 0.5*(1 - crop_fraction)*gridSizeInWorldUnit
            from the edges of the x and y, and z axis of dose and uncertainty maps and will adjust the rest of
            the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - crop_fraction := 3 floating point between 0 and 1 (one per axis for x, y, z axis), which is the fraction of the image axis
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
        assert np.all(crop_fraction >= 0) and np.all(
            crop_fraction <= 1
        ), "the fraction should be between 0 and 1"

        off_set = 0.5 * (1 - crop_fraction) * self.dose_image.gridSizeInWorldUnit
        new_origin_coords = self.dose_image.origin + off_set
        assert np.all(
            new_origin_coords >= self.dose_image.origin
        ), "new origin cannot be smaller than the original origin."

        new_ending_coords = (
            self.dose_image.origin + self.dose_image.gridSizeInWorldUnit - off_set
        )

        new_coords_range = np.column_stack([new_origin_coords, new_ending_coords])

        return self.crop_by_coordinates(new_coords_range, inplace)

    def crop_by_index(
        self, index_range: np.array, inplace: Optional[bool] = True
    ) -> Union[None, "BrachyDose"]:
        r"""
        Purpose:
            - given a range of indicies (mix and max on each axis), this function will crop
            dose and uncertainty maps and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyDose object
            - index_range := a 3 x 2 array holding the min and max on z, y and x axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
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

    def crop_by_contour(
        self,
        phantom_obj: BrachyPhantom,
        contour_name: str,
        inplace: Optional[bool] = True,
    ) -> Union[None, "BrachyDose"]:
        r"""
        Purpose:
            - based on the given phantom object, crop the BrachyDose object such
            that it only contains the smallest bounding box around the contour.
        Inputs:
            - self: BrachyDose object
            - phantom_obj: BrachyPhantom object
            - contour_name: str := the name of the contour to crop the dose around
            - inplace: bool := if True, the function will crop the dose in place, otherwise it will return a new dose object
        Outputs:
            - None or BrachyDose := if inplace is True, the function will return None, otherwise it will return a new BrachyDose object
        """
        from opentps.core.data import ROIMask
        from opentps.core.processing.imageProcessing.resampler3D import (
            resampleImage3DOnImage3D,
        )
        from opentps.core.processing.segmentation.segmentation3D import getBoxAroundROI

        mask_dict = phantom_obj.get_structure_mask([contour_name], mask_type=ROIMask)
        resampled_mask = resampleImage3DOnImage3D(
            mask_dict[contour_name], self.density_image
        )
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

    def get_dose_array(self) -> np.ndarray:
        r"""
        Purpose:
            - To return the dose grid as a numpy array.
        Inputs:
            - self:BrachyDose
        Outputs:
            - dose_array := a numpy array containing the dose grid. in zyx order.
        """
        return np.swapaxes(self.dose_image.imageArray, 0, 2)

    def set_dose_array(self, dose_array: np.ndarray) -> None:
        r"""
        Purpose:
            - To set the dose grid to a numpy array.
        Inputs:
            - self:BrachyDose
            - dose_array := a numpy array containing the dose grid. in zyx order.
        Outputs:
            - Void
        """
        self.dose_image.imageArray = np.swapaxes(dose_array, 0, 2)

    def get_uncertainty_array(self) -> np.ndarray:
        r"""
        Purpose:
            - To return the uncersitainty grid as a numpy array.
        Inputs:
            - self:BrachyDose
        Outputs:
            - dose_array := a numpy array containing the uncertainty grid. in zyx order.
        """
        return np.swapaxes(self.uncertainty_image.imageArray, 0, 2)

    def set_uncertainty_array(self, uncertainty_array: np.ndarray) -> None:
        r"""
        Purpose:
            - To set the uncertainty grid to a numpy array.
        Inputs:
            - self:BrachyDose
            - uncertainty_array := a numpy array containing the uncertainty grid. in zyx order.
        Outputs:
            - Void
        """
        if not (uncertainty_array is None):
            self.uncertainty_image.imageArray = np.swapaxes(uncertainty_array, 0, 2)
        else:
            self.uncertainty_image = None

    @staticmethod
    def dose_with_empty_grid_like(dose_obj: "BrachyDose") -> "BrachyDose":
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
        new_dose.dose_image = DoseImage.createEmptyDoseWithSameMetaData(
            dose_obj.dose_image
        )
        if dose_obj.uncertainty_image is not None:
            new_dose.uncertainty_image = DoseImage.createEmptyDoseWithSameMetaData(
                dose_obj.uncertainty_image
            )
        new_dose.get_voxel_edges()
        new_dose.create_interpolation_function()
        return new_dose

    @staticmethod
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
            diff_list = list(
                difflib.ndiff(contents1.splitlines(), contents2.splitlines())
            )
            print("\n".join(diff_list))


from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
    
def _prepare_dose_loading_item(pth_input: Path) -> dict:
    """Prepare loading item for dose files."""
    full_suffix = "".join(pth_input.suffixes)
    
    if full_suffix in [".3ddose", ".seq.nrrd"]:
        return {
            "loader_class": BrachyDose,
            "args_dict": {"pth_dose_file": pth_input, "load_uncertainty": False}
        }
    else:
        raise ValueError(
            f"Unsupported file type {full_suffix} for dose conversion. "
            "Please provide a .3ddose, .nrrd, or .minidos file."
        )

def _handle_dicom_directory_dose(pth_input: Path) -> List[dict]:
    """Process a directory containing DICOM files, return only dose items."""
    data_to_load = []
    
    if len(list(pth_input.glob("*.dcm"))) < 1:
        print(f"No DICOM files found in the directory {pth_input}.")
        return data_to_load
    
    # Check for dose file
    dose_file = list(pth_input.glob("[Rr][Dd]*.dcm"))
    if dose_file:
        loading_dose_item = {
            # "loader_class": BrachyDose,
            "args_dict": {
                "pth_dose_file": dose_file[0],
                "load_uncertainty": False,
            }
        }
        data_to_load.append(loading_dose_item)
    else:
        print(f"No dose file found in the directory {pth_input}")
    
    return data_to_load
    
def _perform_dose_conversion(item: dict, dir_output: Path, type_out: str):
    """Perform actual dose conversion."""
    # loader_class = BrachyDose if item["loader_class"] is BrachyDose else None
    # if loader_class is None:
    #     raise ValueError("Invalid loader class provided for dose conversion.")
    args_dict = item["args_dict"]
    
    # Extract base name for output files
    if "pth_dose_file" in args_dict:
        full_ext = "".join(args_dict["pth_dose_file"].suffixes)
        base_name = str(args_dict["pth_dose_file"].name).split(full_ext)[0]
    else:
        base_name = "converted"
    
    # Convert based on output type
    dose_obj = BrachyDose(
        pth_dose_file=args_dict["pth_dose_file"],
        load_uncertainty=args_dict.get("load_uncertainty", False)
        )
    if type_out == ".nrrd":
        pth_out = dir_output / f"{base_name}.seq{type_out}"
    elif type_out == ".3ddose":
        pth_out = dir_output / f"{base_name}{type_out}"
    elif type_out == ".dcm":
        pth_out = dir_output / f"RD{type_out}"
    else:
        raise ValueError(f"Unsupported output type {type_out} for dose conversion.")
    
    dose_obj.write_brachydose_to_file(pth_dose_file=pth_out)

# Conversion utilities for dose files
def convert_dose_files(
    pth_inputs: List[Union[Path, str]],
    type_out: Literal[".nrrd", ".dcm", ".3ddose"] = ".nrrd",
    dir_output: Optional[Union[Path, str]] = None,
    multi_proc: bool = False
) -> None:
    """
    Convert dose files to the specified output format.
    
    Args:
        pth_inputs: List of paths to input dose files. Can be directories or files.
        type_out: Output file type. Options are ".nrrd", ".dcm", ".3ddose".
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
            dicom_data = _handle_dicom_directory_dose(pth_input)
            data_to_load.extend(dicom_data)
        
        # Handle single files
        elif pth_input.is_file():
            data_to_load.append(_prepare_dose_loading_item(pth_input))
        else:
            raise ValueError(f"Input {pth_input} is neither a file nor a directory.")
    
    # Check if we have valid items to process
    if not data_to_load:
        raise ValueError("No valid dose files found to convert.")
    
    # Setup output directory
    if dir_output is None:
        dir_output = Path(pth_inputs[0]).parent
    else:
        dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)
    
    # Perform conversion
    if multi_proc:
        # Create partial function with fixed arguments
        partial_conversion = partial(_perform_dose_conversion, dir_output=dir_output, type_out=type_out)
        with Pool() as pool:
            list(tqdm(pool.imap(partial_conversion, data_to_load), total=len(data_to_load), desc="Converting dose files"))
    else:
        for item in tqdm(data_to_load):
            _perform_dose_conversion(item, dir_output, type_out)
