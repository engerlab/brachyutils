import json
import os
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# from dicom_utils import get_structure_index_range
from brachyutils.dicom_utils import BrachyDicom


class BrachyEgsphant:
    r"""
    Purpose:
        An object to allow for loading and manipulating the .egsphant files

    Attributes:
        - material_matrix:np.ndarray
        - density_matrix:np.ndarray
        - num_materials:int := the number of different material composition options a voxel has
        - material_dict:dict := a dictionary containing the name of the elements for each voxel,
            their density and HU lower limit threshold as well as their number coding
        - num_voxels:np.ndarray := 1D numpy array holding the number of grid points
        on x, y, z axis.
        - voxel_size:np.ndarray := 1D numpy array holding the resolution of each voxel
        along x, y, z axis in centimeters.
        - origin_coordinates:np.ndarray := The spatial coordinate of the "bottom" left corner of
        the image in centrimeters. [x, y, z]
        - axis:np.ndarray := coorindates of grid points along z, y and x axis.

    Functions:
        - load_file_to_BrachyEgsphant()     done
        - load_from_ctegsphant()            done
        - load_from_nrrd()                  not implmented
        - calculate_axis()                  done
        - write_to_ctegsphant()             done
        - write_to_nrrd()                   not implemented
        - crop_by_index()                   done
        - crop_by_coordinates()             done
        - crop_by_body_contour()            done
        - assert_BrachyEgsphant_notEmpty()  done
        - info()                            done
        - is_equal()                        done

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

    # each voxel in the material matrix is encoded with a single character
    # from this array that represents a unique material recognized by RapidBrachyMC.
    _materials_encoding_array = [str(i) for i in range(0, 10)] + [
        chr(i) for i in range(ord("A"), ord("Z") + 1)
    ]

    def __init__(
        self,
        pth_egsphant_file: Optional[Path] = None,
        image: Optional[BrachyDicom] = None,
        material_dict: Optional[Union[dict, Path]] = None,
        assign_material_from_ct: Optional[bool] = None,
    ):

        self.material_matrix: np.ndarray = None
        self.material_interpolation_function = None

        self.density_matrix: np.ndarray = None
        self.density_interpolation_function = None

        self.num_materials: int = None
        self.material_dict: defaultdict = defaultdict(dict)
        self.material_dict["Air"] = {
            "encoding": 0,
            "density": 0.001225,
            "HU_limit": -1000.0,
        }

        self.num_voxels: np.ndarray = None
        self.voxel_size: np.ndarray = None
        self.origin_coordinates: np.ndarray = None
        self.voxel_edges: np.ndarray = None
        self._sanity_axis: np.ndarray = None

        if pth_egsphant_file is not None:
            self.load_file_to_BrachyEgsphant(pth_egsphant_file)

        if image is not None and material_dict is not None:

            self.create_egsphant_from_images(
                image=image,
                new_material_dict=(
                    material_dict
                    if isinstance(material_dict, dict)
                    else _load_material_dict(material_dict)
                ),
                assign_material_from_ct=assign_material_from_ct,
            )

        if self.material_matrix is not None:
            self.material_interpolation_function = self.create_interpolation_function(
                self.material_matrix
            )
        if self.density_matrix is not None:
            self.density_interpolation_function = self.create_interpolation_function(
                self.density_matrix
            )

    def load_file_to_BrachyEgsphant(self, pth_egsphant_file):
        pth_egsphant_file = os.path.abspath(pth_egsphant_file)

        assert os.path.exists(
            pth_egsphant_file
        ), "The target egsphant file does not exist!"
        file_extension = os.path.splitext(pth_egsphant_file)[-1]

        if file_extension == ".egsphant":
            self.load_from_ctegsphant(pth_egsphant_file)
        elif file_extension == ".nrrd":
            self.load_from_nrrd(pth_egsphant_file)
        else:
            raise Exception(
                f"Loading from file extension {file_extension} is not supported!"
            )

    def load_from_ctegsphant(self, pth_file: Path):
        r"""
        Purpose:
            to load a file with extension .egsphant into a BrachyEgsphant object
        Input:
            - pth_file := directory path to the .egsphant file
        """
        assert (
            os.path.splitext(pth_file)[-1] == ".egsphant"
        ), "target file does not have .egsphant extension"

        with open(pth_file, "r") as egsphant:
            # first line describes how many materials are used
            self.num_materials = int(egsphant.readline().strip())

            # load each material line by line
            for i in range(self.num_materials):
                self.material_dict[egsphant.readline().strip()] = {
                    "encoding": BrachyEgsphant._materials_encoding_array[i]
                }

            self._sort_materials_by("encoding")

            egsphant.readline()

            # load number of voxels
            self.num_voxels = np.array(
                [int(i) for i in egsphant.readline().strip().split()]
            )

            # load the axis grid points
            self._sanity_axis = np.array(
                [
                    np.array(
                        [float(x) for x in egsphant.readline().strip().split()],
                        dtype=np.float32,
                    ),
                    np.array(
                        [float(y) for y in egsphant.readline().strip().split()],
                        dtype=np.float32,
                    ),
                    np.array(
                        [float(z) for z in egsphant.readline().strip().split()],
                        dtype=np.float32,
                    ),
                ],
                dtype=object,
            )
            self._sanity_axis = np.flip(self._sanity_axis, axis=0)

            self.origin_coordinates = np.array(
                [
                    self._sanity_axis[2][0],
                    self._sanity_axis[1][0],
                    self._sanity_axis[0][0],
                ],
                dtype=np.float32,
            )

            self.voxel_size = np.array(
                [
                    self._sanity_axis[2][1] - self._sanity_axis[2][0],
                    self._sanity_axis[1][1] - self._sanity_axis[1][0],
                    self._sanity_axis[0][1] - self._sanity_axis[0][0],
                ]
            )
            # this line maybe useless in the future
            self.voxel_edges = self.calculate_voxel_edges()
            # {for debugging
            # print(f"The axis calculated from calculate_voxel_edges() are \n {self.voxel_edges}")
            # print(f"The axis from the text file are: \n {self._sanity_axis}")
            # print(f"the size of the axis in the z, y, x for axis from calcAxis() are {self.voxel_edges[0].shape}, {self.voxel_edges[1].shape}, {self.voxel_edges[2].shape}")
            # print(f"the size of the axis in the z, y, x for axis from file are {self._sanity_axis[0].shape}, {self._sanity_axis[1].shape}, {self._sanity_axis[2].shape}")
            # }
            assert np.isclose(
                np.concatenate(self.voxel_edges),
                np.concatenate(self._sanity_axis),
                rtol=1e-1,
            ).all(), "axis is not the same"

            # prepare empty matricies to hold material and density images
            self.material_matrix = np.zeros(
                (self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]), dtype=str
            )
            self.density_matrix = np.zeros(
                (self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]),
                dtype=np.float32,
            )

            # load the material composition data in to the matrix
            for k in range(self.num_voxels[2]):
                for j in range(self.num_voxels[1]):
                    self.material_matrix[k][j] = list(egsphant.readline().strip())
                egsphant.readline()

            # load the density data into the matrix
            for k in range(self.num_voxels[2]):
                for j in range(self.num_voxels[1]):
                    self.density_matrix[k][j] = egsphant.readline().strip().split()
                egsphant.readline()

            self.material_matrix = self._convert_material_matrix_to(dtype=int)

    def _sort_materials_by(self, material_key="encoding"):
        r"""
        Purpose:
            to sort the materials in the material dictionary based on their keys.
        Input:
            - self: BrachyEgsphant object with material_dict attribute. The material_dict
            has to have at least the encoding key for each material.
        Output:
            - Void: will sort the material_dict based on the encoding
        """
        assert material_key in [
            "encoding",
            "density",
            "HU_limit",
            "structure_name",
            "structure_size",
        ], "key is not recognized"

        if material_key == "structure_size":
            sorted_list = sorted(
                self.material_dict.items(),
                key=lambda x: x[1].get(material_key, 0),
                reverse=True,
            )
        else:
            sorted_list = sorted(
                self.material_dict.items(), key=lambda x: x[1][material_key]
            )
        self.material_dict = defaultdict(dict)
        for key, value in sorted_list:
            self.material_dict[key] = value

    def load_from_nrrd(self, pth_file: Path):
        r"""
        Purpose:
            to load a nrrd file containing egsphant data.
        Input:
            - pth_file := directory path to the .nrrd file
        """
        raise Exception("This function is not implemented yet!")

    def calculate_voxel_edges(self):
        r"""
        Purpose: will calculate the axies coordinates for a BrachyEgsphant object.
        Input:
            - self := it should have the following keys and values:
                {"grid":,
                "origin_coordinates":,
                "voxel_size":}
        Output:
            - axes:numpy.array() :=
            [[z_min:voxel_size:z_max],
            [y_min:voxel_size:y_max],
            [x_min:voxel_size:x_max]]
        """
        # calculate the end point of axis in 3D space
        axes_end = np.array(
            # one voxel size is added because np.arange stops at an index before the end
            self.origin_coordinates
            + self.num_voxels * self.voxel_size
            + self.voxel_size
        )

        self.voxel_edges = np.empty(len(axes_end), dtype=object)
        for i in range(len(axes_end)):
            self.voxel_edges[i] = np.arange(
                self.origin_coordinates[len(axes_end) - 1 - i],
                axes_end[len(axes_end) - 1 - i],
                self.voxel_size[len(axes_end) - 1 - i],
                dtype=np.float32,
            )
            if np.absolute(self.num_voxels[::-1][i] - self.voxel_edges[i].shape[0]) > 1:
                self.voxel_edges[i] = self.voxel_edges[i][:-1]
        return self.voxel_edges

    def create_interpolation_function(self, grid):
        voxel_centers = self.get_voxel_centers()
        self.interpolation_function = RegularGridInterpolator(
            (voxel_centers[0], voxel_centers[1], voxel_centers[2]),
            grid,
            bounds_error=False,
            fill_value=0,
        )

    def get_voxel_centers(self):
        voxel_centers = np.empty(len(self.voxel_edges), dtype=object)
        if self.voxel_edges is not None:
            for i in range(len(self.voxel_edges)):
                voxel_centers[i] = self.voxel_edges[i] + self.voxel_size[i] / 2.0
                voxel_centers[i] = voxel_centers[i][:-1]
        else:
            raise ValueError("Voxel edges are not calculated yet")
        return voxel_centers

    def write_to_ctegsphant(self, fileName: Path):
        r"""
        Purpose:
            This function will write the contents of a BrachyEgsphant onto a text
            file with .egsphant extension.

        inputs:
            - self := a BrachyEgsphant object containing the following keys:
                num_materials:int
                material_dict:dict
                num_voxels:np.ndarray       [x, y, z]
                voxel_size:np.ndarray         #Not Written
                origin_coordinates:np.ndarray          #Not Written
                axis:np.ndarray             [z, y, x] -> [x, y, z]
                material_matrix:np.ndarray  [z, y, x] -> [x, y, z]
                density_matrix:np.ndarray   [z, y, x] -> [x, y, z]

            - fileName := the directory path where the file will be written
        """
        assert (
            os.path.splitext(fileName)[-1] == ".egsphant"
        ), "file extension is not .egsphant"
        fileName = os.path.abspath(fileName)
        self._sort_materials_by("encoding")
        num_materials = str(self.num_materials) + "\n"
        materials = "\n".join(self.material_dict.keys()) + "\n"
        spacing = "0 0 0 0 0 0 0 0 0\n"
        dimensions = " ".join(map(str, self.num_voxels.astype(int))) + "\n"
        x_axis = " ".join(map(str, np.round(self.voxel_edges[2], decimals=3))) + "\n"
        y_axis = " ".join(map(str, np.round(self.voxel_edges[1], decimals=3))) + "\n"
        z_axis = " ".join(map(str, np.round(self.voxel_edges[0], decimals=3))) + "\n"
        material_matrix = _to_single_string(
            self._convert_material_matrix_to(dtype=str), ""
        )
        density_matrix = _to_single_string(self.density_matrix.astype(str), " ")

        with open(fileName, "w") as file:
            lines = [
                num_materials,
                materials,
                spacing,
                dimensions,
                x_axis,
                y_axis,
                z_axis,
                material_matrix,
                density_matrix,
            ]
            file.writelines(lines)

    def is_equal(self, new_BrachyEgsphant):
        r"""
        Purpose:
            To compare if self:BrachyEgsphant has the same attributes as an input BrachyEgsphant

        Inputs:
            - new_BrachyEgsphant: another BrachyEgsphant object whose attributes may or may not contain equal info as the attributes of self.

        Outputs:
            True if attributes of new_BrachyEgsphant are the same as self
            False otherwise
        """
        assert isinstance(
            new_BrachyEgsphant, BrachyEgsphant
        ), "input must be of type BrachyEgsphant"
        assert np.array_equal(
            self.material_matrix, new_BrachyEgsphant.material_matrix
        ), "material matrix is not the same"
        assert np.array_equal(
            self.density_matrix, new_BrachyEgsphant.density_matrix
        ), "density matrix is not the same"
        assert np.isclose(
            np.concatenate(self.voxel_edges),
            np.concatenate(new_BrachyEgsphant.voxel_edges),
            rtol=1e-3,
        ).all(), "axis is not the same"
        assert np.array_equal(
            self.num_materials, new_BrachyEgsphant.num_materials
        ), "number of materials is not the same"
        assert (
            self.material_dict == new_BrachyEgsphant.material_dict
        ), "the material dictionary is not the same"
        assert np.array_equal(
            self.num_voxels, new_BrachyEgsphant.num_voxels
        ), "num_voxels is not the same"
        assert np.array_equal(
            self.voxel_size, new_BrachyEgsphant.voxel_size
        ), "voxel_size is not the same"
        assert np.isclose(
            self.origin_coordinates, new_BrachyEgsphant.origin_coordinates, rtol=1e-3
        ).all(), "origin_coordinates is not the same"

        return (
            np.array_equal(self.material_matrix, new_BrachyEgsphant.material_matrix)
            and np.array_equal(self.density_matrix, new_BrachyEgsphant.density_matrix)
            and np.isclose(
                np.concatenate(self.voxel_edges),
                np.concatenate(new_BrachyEgsphant.voxel_edges),
                rtol=1e-3,
            ).all()
            and np.array_equal(self.num_materials, new_BrachyEgsphant.num_materials)
            and self.material_dict == new_BrachyEgsphant.material_dict
            and np.array_equal(self.num_voxels, new_BrachyEgsphant.num_voxels)
            and np.array_equal(self.voxel_size, new_BrachyEgsphant.voxel_size)
            and np.array_equal(
                self.origin_coordinates, new_BrachyEgsphant.origin_coordinates
            )
        )

    def assert_BrachyEgsphant_notEmpty(self):
        r"""
        Purpose:
            to see which field of a brachyEgsphant object is empty
        """
        assert self.material_matrix is not None, "error: material_matrix is None"
        assert self.density_matrix is not None, "error: density_matrix is None"
        assert self.num_materials is not None, "error: num_materials is None"
        assert self.material_dict is not None, "error: material_dict is None"
        assert self.num_voxels is not None, "error: num_voxels is None"
        assert self.voxel_size is not None, "error: voxel_size is None"
        assert self.origin_coordinates is not None, "error: origin_coordinates is None"
        assert self.voxel_edges is not None, "error: axis is None"

    def info(self):
        self.assert_BrachyEgsphant_notEmpty()
        print(f"shape of material matrix is: {self.material_matrix.shape}")
        print(f"shape of density matrix is: {self.density_matrix.shape}")
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
        print(f"The number of materials is {self.num_materials}")
        print(f"the material dictionary is {self.material_dict}")

    def crop_by_index(self, index_range: np.array, inplace: Optional[bool] = True):
        r"""
        Purpose:
            given a range of indicies (mix and max on each axis), this function will crop
            material and density matricies and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyEgsphant object
            - index_range := a 3 x 2 array holding the min and max index on x, y and axis
                [[ix_min, ix_max], [iy_min, iy_max], [iz_min, iz_max]]
        Output:
            - Void := will crop out the material and density maps of self to have the range of the index range.
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
            self.material_matrix = self.material_matrix[
                new_origin_index[2] : new_ending_index[2],  # z
                new_origin_index[1] : new_ending_index[1],  # y
                new_origin_index[0] : new_ending_index[0],  # x
            ]
            self.density_matrix = self.density_matrix[
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
            self.num_voxels = np.flip(self.material_matrix.shape, 0)
            self.voxel_edges = self.calculate_voxel_edges()
        else:
            new_obj = BrachyEgsphant()
            new_obj.material_matrix = self.material_matrix[
                new_origin_index[2] : new_ending_index[2],
                new_origin_index[1] : new_ending_index[1],
                new_origin_index[0] : new_ending_index[0],
            ]
            new_obj.density_matrix = self.density_matrix[
                new_origin_index[2] : new_ending_index[2],
                new_origin_index[1] : new_ending_index[1],
                new_origin_index[0] : new_ending_index[0],
            ]
            new_obj.origin_coordinates = np.array(
                [
                    self.voxel_edges[2][new_origin_index[0]],  # x
                    self.voxel_edges[1][new_origin_index[1]],  # y
                    self.voxel_edges[0][new_origin_index[2]],  # z
                ]
            )
            new_obj.material_dict = self.material_dict
            new_obj.num_voxels = np.flip(new_obj.material_matrix.shape, 0)
            new_obj.voxel_size = self.voxel_size
            new_obj.voxel_edges = new_obj.calculate_voxel_edges()
            new_obj.num_materials = self.num_materials
            return new_obj

    def crop_by_coordinates(
        self, coordinate_range: np.array, inplace: Optional[bool] = True
    ):
        r"""
        Purpose:
            given a range of coordinates (mix and max on each axis), this function will crop
            material and density matricies and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyEgsphant object
            - coordinate_range := a 3 x 2 array holding the min and max on x, y and axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        Output:
            - Void := will crop out the material and density maps of self to have the range of the index range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            - None
        """
        crop_indices = np.zeros((3, 2), dtype=int)

        for i in range(3):
            origin = self.origin_coordinates[i]
            for j in range(2):
                crop_indices[i][j] = int(
                    (coordinate_range[i][j] - origin) / self.voxel_size[i]
                )

        return self.crop_by_index(crop_indices, inplace=inplace)

    def crop_by_body_contour(
        self,
        body_index_range: Optional[np.ndarray] = None,
        body_mask_shape: Optional[np.ndarray] = None,
        pth_dir_dicom: Optional[Path] = None,
    ):
        r"""
        Purpose:
            based on the given dicom structure file, crop the BrachyEgsphant object such
                that it only has the body contour.
        Inputs:
            - body_index_range:np.array :=  a 3 x 2 array holding the min and max on x, y and axis
                [[x_min, x_max], [y_min, y_max], [z_min, z_max]]. If this is not available, provide
                the third input.

            - original_mask_dimensions:np.array := 1 x 3 array holding the dimension of the original mask.
                If this is not available, provide the third input.

            - pth_dir_dicom := pth_dir_dicom := path to the directory with the dicom files of a patient.
                it should contain both images and RTSTRUCT file. this input is used when the first 2 inputs
                are not available

        Outputs:
            - Void := will crop out the material and density maps of self to have the range of the body contour
                    in the dicom structure file. It will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        """

        if body_index_range is None or body_mask_shape is None:
            assert (
                pth_dir_dicom is not None
            ), "Either path to a dicom directory with dicom structure \
                file should be given or body_index_range and body_mask_shape"
            body_mask_info = BrachyDicom(pth_dir_dicom, query_structure_list=["body"])
            body_index_range = body_mask_info["body"]["structure_index_range"]
            body_mask_shape = body_mask_info["body"]["dicom_mask_shape"]
        # the body mask may have a different size than the material map, we normalize range to the dimension
        # of original mask and scale it to the dimension of the material map to get the body index range on the material image.
        scaled_body_index_range = (
            body_index_range
            / np.expand_dims(body_mask_shape, axis=1)
            * np.expand_dims(self.num_voxels, axis=1)
        ).astype(int)
        print(scaled_body_index_range)
        self.crop_by_index(scaled_body_index_range, True)

    def create_egsphant_from_images(
        self,
        image: BrachyDicom,
        new_material_dict: dict = None,
        assign_material_from_ct: bool = True,
    ):
        r"""
        Purpose:
            - To generate an egsphant object from a directory containing images and a structure file.
            if the structure file is not provided, the function will look for it in the directory assuming
            that the dicom structure files start with RS and the NRRD structure files end with seg.nrrd.
        Inputs:
            - image := A brachy image object containing a grid and the structures.
            - new_material_dict := A dictionary containing the material composition of the egsphant object.
            the following keys are required for each material:
                - encoding: a single character string that represents the material in the material matrix
                - density: the density of the material in g/cm^3
                - HU_limit: the lower HU limit of the material
                - structure_name: the name of the structure in the dicom file that represents the material [optional]
                - structure_size: the size of the structure in the dicom file that represents the material [optional]
        Outputs:
            - Void := will generate a BrachyEgsphant object from the images and structure file.
        Dependencies:
            - BrachyDicom
        """
        if not assign_material_from_ct:
            assert image.structure_mask_dict is not None, "No structure mask was found"
        for material in new_material_dict:
            assert {"encoding", "density", "HU_limit"}.issubset(
                set(new_material_dict[material].keys())
            ), "material dictionary is not formatted correctly"

        self.material_dict = new_material_dict

        # get the egsphant dimensions and voxel size from the image.
        self.num_voxels = image.num_voxels
        self.voxel_size = image.voxel_size
        self.origin_coordinates = image.origin_coordinates
        self.voxel_edges = self.calculate_voxel_edges()
        self.material_matrix = np.ones_like(image.grid, dtype=int)
        self.density_matrix = np.ones_like(image.grid, dtype=np.float32)

        # loop through the material, get their binary mask from the ct images apply it to the material
        # density materix.
        materials_list = list(self.material_dict.keys())

        if assign_material_from_ct:
            # sort out the materials and density based on the HU values
            self._sort_materials_by("HU_limit")

            for i, material in enumerate(self.material_dict.keys()):

                # numerically interpolate the density and material based on the HU values
                low_HU_threshold = self.material_dict.get(material).get("HU_limit")
                high_HU_threshold = (
                    self.material_dict.get(materials_list[i + 1]).get("HU_limit")
                    if i + 1 < len(materials_list)
                    else float("inf")
                )
                density_low_bound = self.material_dict.get(material).get("density")
                density_high_bound = (
                    self.material_dict.get(materials_list[i + 1]).get("density")
                    if i + 1 < len(materials_list)
                    else density_low_bound
                )

                slope_density_over_HU = (density_high_bound - density_low_bound) / (
                    high_HU_threshold - low_HU_threshold
                )
                intercept_density_over_HU = density_low_bound - (
                    slope_density_over_HU * low_HU_threshold
                )
                # find region of interest mask based on the HU values
                roi_mask = np.logical_and(
                    np.where(
                        image.grid >= low_HU_threshold,
                        1,
                        0,
                    ).astype(bool),
                    np.where(
                        image.grid < high_HU_threshold,
                        1,
                        0,
                    ).astype(bool),
                )
                # set the density and material of all voxels outside the lowest HU_limit to air
                if i == 0:
                    complementary_roi_mask = np.logical_not(roi_mask)
                    self.density_matrix *= roi_mask
                    self.material_matrix *= roi_mask
                    self.density_matrix += (
                        complementary_roi_mask
                        * self.material_dict.get("Air").get("density")
                    )
                    self.material_matrix += (
                        complementary_roi_mask
                        * BrachyEgsphant._materials_encoding_array.index(
                            self.material_dict.get("Air").get("encoding")
                        )
                    )

                # reset the voxel values for the roi enetries
                self.density_matrix *= np.logical_not(roi_mask)
                self.material_matrix *= np.logical_not(roi_mask)

                # update the density and material matricies
                # interpolate density based on the HU value
                self.density_matrix += (
                    image.grid * roi_mask * slope_density_over_HU
                    + intercept_density_over_HU
                )
                self.material_matrix += (
                    roi_mask
                    * BrachyEgsphant._materials_encoding_array.index(
                        self.material_dict.get(material).get("encoding")
                    )
                )
                # assert np.all(self.density_matrix >= 0), "density matrix has negative values"
        else:
            dicom_structure_list = list(image.structure_mask_dict.keys())
            # get the mask of each material from image
            for material in self.material_dict:
                structure_name = self.material_dict.get(material).get(
                    "structure_name", None
                )
                if structure_name is None:
                    continue
                structure_dicom_name = list(
                    filter(lambda x: structure_name in x, dicom_structure_list)
                )[0]
                self.material_dict.get(material)["structure_size"] = np.sum(
                    image.structure_mask_dict.get(structure_dicom_name)
                )

            # sort the material dictionary based on the size of the mask (from largest to smallest)
            self._sort_materials_by("structure_size")

            for i, material in enumerate(self.material_dict.keys()):
                structure_name = self.material_dict.get(material).get(
                    "structure_name", None
                )
                if structure_name is None:
                    continue
                # get the mask of each material from image
                structure_dicom_name = list(
                    filter(lambda x: structure_name in x, dicom_structure_list)
                )[0]
                roi_mask = image.structure_mask_dict.get(structure_dicom_name).astype(
                    bool
                )

                # set everything outside the largest mask to air
                if i == 0:
                    complementary_roi_mask = np.logical_not(roi_mask)
                    self.density_matrix *= roi_mask
                    self.material_matrix *= roi_mask
                    self.density_matrix += (
                        complementary_roi_mask
                        * self.material_dict.get("Air").get("density")
                    )
                    self.material_matrix += (
                        complementary_roi_mask
                        * BrachyEgsphant._materials_encoding_array.index(
                            self.material_dict.get("Air").get("encoding")
                        )
                    )

                # reset the voxel values for the roi enetries
                self.density_matrix *= np.logical_not(roi_mask)
                self.material_matrix *= np.logical_not(roi_mask)

                # update the density and material matricies
                self.density_matrix += roi_mask * self.material_dict.get(material).get(
                    "density"
                )
                self.material_matrix += (
                    roi_mask
                    * BrachyEgsphant._materials_encoding_array.index(
                        self.material_dict.get(material).get("encoding")
                    )
                )

    def _convert_material_matrix_to(self, dtype: type):
        r"""
        Purpose:
            To convert a numpy array of dtype string to an integer numpy array or the other way around.
            Integer array is the desired data type over string since it allows for more operational functionality.
            String array is desired for outputting the egsphant file.
        Inputs:
            - self.material_matrix:np.array(dtype=str) := a numpy array with string enteries
            - BrachyEgsphant._encoding_array:list := a list of strings that will be used to encode the string enteries
        Outputs:
            - Void := will update the material_dict with the density and HU lower limit thresholds.
        """
        assert dtype in [int, str], "dtype is not recognized"

        flattened_array = self.material_matrix.flatten()

        if dtype is int:

            int_array = np.zeros_like(flattened_array, dtype=int)

            for i, string in enumerate(flattened_array):
                int_array[i] = BrachyEgsphant._materials_encoding_array.index(string)

            return int_array.reshape(self.material_matrix.shape)

        else:
            str_array = np.zeros_like(flattened_array, dtype=str)
            for i, integer in enumerate(flattened_array):
                str_array[i] = BrachyEgsphant._materials_encoding_array[integer]

            return str_array.reshape(self.material_matrix.shape)

    def export_material_dict(self, pth_file: Path):
        r"""
        Purpose:
            To export the material dictionary to a json file.
        Inputs:
            - pth_file:Path := the directory path where the json file will be written.
        Outputs:
            - Void := will write the material dictionary to a json file.
        """
        extension = os.path.splitext(pth_file)[-1]
        if extension == ".json":
            with open(pth_file, "w") as file:
                json.dump(self.material_dict, file, indent=4)
        elif extension == ".txt":
            with open(pth_file, "w") as file:
                for material, values in self.material_dict.items():
                    file.write(f"{material} {values['density']} {values['HU_limit']}\n")


def _to_single_string(matrix: np.ndarray, deliminator: Optional[str] = ""):
    r"""
    Purpose:
        given a 3D matrix with string entries, this function concatenates all the
            entries into a single string to be written to the file.
            "\n" is added at the end of each row and
    Input:
        matrix := 3D ndarray full of string enteries
        deliminator := the string text inbetween the enteries.
    Output:
        a single string containing all the enteries with added \n at the end of each row of
            matrix and an addiation \n added to each slide in the input matrix

    """
    matrix_single_string = []
    for slide in matrix:
        slide_single_string = []
        for row in slide:
            slide_single_string.append(deliminator.join(row) + "\n")
        matrix_single_string.append(deliminator.join(slide_single_string) + "\n")

    return "".join(matrix_single_string)


def _load_json(pth_json: Path):
    assert os.path.exists(
        pth_json
    ), f"no such json file was found at this directory: \n {pth_json}"

    with open(pth_json, "r") as file_json:
        return json.load(file_json)


def _load_material_dict(pth_file: Path):
    r"""
    Purpose:
        To load material dictionary from a ct to density.txt file or from a json file that
        contains the density and HU lower limit threshold for each material.
    Inputs:
        - pth_file := directory path to the ct2density.txt file
    Outputs:
        - dict := a dictionary containing the density and HU lower limit thresholds for each material.
    """
    assert os.path.exists(
        pth_file
    ), f"no such ct2density.txt file was found at this directory: \n {pth_file}"

    extension = os.path.splitext(pth_file)[-1]

    material_dict = defaultdict(dict)

    if extension == ".txt":
        with open(pth_file, "r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            material, density, HU_limit = line.strip().split()
            material_dict[material] = {
                "density": float(density),
                "HU_limit": float(HU_limit),
                "encoding": BrachyEgsphant._materials_encoding_array[i],
            }
    elif extension == ".json":
        material_dict = _load_json(pth_file)
    else:
        raise Exception(
            f"Loading from file extension {extension} is not supported! only .txt and .json are supported."
        )

    return material_dict
