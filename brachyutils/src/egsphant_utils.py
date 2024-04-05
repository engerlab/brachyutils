import json
import os
from typing import Optional

import numpy as np

# from dicom_utils import get_structure_index_range
from brachyutils.src.dicom_utils import BrachyDicom


class BrachyEgsphant:
    r"""
    Purpose:
        An object to allow for loading and manipulating the .egsphant files

    Attributes:
        - material_matrix:np.ndarray
        - density_matrix:np.ndarray
        - num_materials:int := the number of different material composition options a voxel has
        - material_dict:dict := a dictionary containing the name of the elements for each voxel
            and their number coding
        - num_voxels:np.ndarray := 1D numpy array holding the number of grid points
        on x, y, z axis.
        - voxel_size:np.ndarray := 1D numpy array holding the resolution of each voxel
        along x, y, z axis in centimeters.
        - topleft:np.ndarray := The spatial coordinate of the "bottom" left corner of
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

    material_matrix: np.ndarray
    density_matrix: np.ndarray
    num_materials: int
    material_dict: dict
    num_voxels: np.ndarray
    voxel_size: np.ndarray
    topleft: np.ndarray
    axis: np.ndarray
    _sanity_axis: np.ndarray

    def __init__(self, pth_egsphant_file: Optional[str] = None):

        self.material_matrix: np.ndarray = None
        self.density_matrix: np.ndarray = None
        self.num_materials: int = None
        self.material_dict: dict = {}
        self.num_voxels: np.ndarray = None
        self.voxel_size: np.ndarray = None
        self.topleft: np.ndarray = None
        self.axis: np.ndarray = None
        self._sanity_axis: np.ndarray = None

        if pth_egsphant_file is not None:
            self.load_file_to_BrachyEgsphant(pth_egsphant_file)

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

    def load_from_ctegsphant(self, pth_file: str):
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
                self.material_dict[egsphant.readline().strip()] = i

            egsphant.readline()

            # load number of voxels
            self.num_voxels = np.array(
                [int(i) for i in egsphant.readline().strip().split()]
            )

            # load the axis grid points
            self._sanity_axis = np.array(
                [
                    np.array(
                        [float(x)
                         for x in egsphant.readline().strip().split()],
                        dtype=np.float32,
                    ),
                    np.array(
                        [float(y)
                         for y in egsphant.readline().strip().split()],
                        dtype=np.float32,
                    ),
                    np.array(
                        [float(z)
                         for z in egsphant.readline().strip().split()],
                        dtype=np.float32,
                    ),
                ],
                dtype=object,
            )
            self._sanity_axis = np.flip(self._sanity_axis, axis=0)

            self.topleft = np.array(
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
            self.axis = self.calculateAxis()
            # {for debugging
            # print(f"The axis calculated from calculateAxis() are \n {self.axis}")
            # print(f"The axis from the text file are: \n {self._sanity_axis}")
            # print(f"the size of the axis in the z, y, x for axis from calcAxis() are {self.axis[0].shape}, {self.axis[1].shape}, {self.axis[2].shape}")
            # print(f"the size of the axis in the z, y, x for axis from file are {self._sanity_axis[0].shape}, {self._sanity_axis[1].shape}, {self._sanity_axis[2].shape}")
            # }
            assert np.isclose(
                np.concatenate(self.axis), np.concatenate(self._sanity_axis), rtol=1e-1
            ).all(), "axis is not the same"

            # prepare empty matricies to hold material and density images
            self.material_matrix = np.zeros(
                (self.num_voxels[2], self.num_voxels[1],
                 self.num_voxels[0]), dtype=int
            )
            self.density_matrix = np.zeros(
                (self.num_voxels[2], self.num_voxels[1], self.num_voxels[0]),
                dtype=np.float32,
            )

            # load the material composition data in to the matrix
            for k in range(self.num_voxels[2]):
                for j in range(self.num_voxels[1]):
                    self.material_matrix[k][j] = list(
                        egsphant.readline().strip())
                egsphant.readline()

            # load the density data into the matrix
            for k in range(self.num_voxels[2]):
                for j in range(self.num_voxels[1]):
                    self.density_matrix[k][j] = egsphant.readline(
                    ).strip().split()
                egsphant.readline()

    def load_from_nrrd(self, pth_file: str):
        r"""
        Purpose:
            to load a nrrd file containing egsphant data.
        Input:
            - pth_file := directory path to the .nrrd file
        """
        raise Exception("This function is not implemented yet!")

    def calculateAxis(self):
        r"""
        Purpose: will calculate the axies coordinates for a BrachyEgsphant object.
        Input:
            - self := it should have the following keys and values:
                {"grid":,
                "topleft":,
                "voxel_size":}
        Output:
            - axes:numpy.array() :=
            [[z_min:voxel_size:z_max],
            [y_min:voxel_size:y_max],
            [x_min:voxel_size:x_max]]
        """
        # calculate the end point of axis in 3D space
        axes_end = np.array(
            self.topleft
            + self.num_voxels * self.voxel_size
            # one voxel size is added because np.arange stops at an index before the end
            + self.voxel_size
        )
        axes = np.empty(len(axes_end), dtype=object)
        for i in range(len(axes_end)):
            axes[i] = np.arange(
                self.topleft[len(axes_end) - 1 - i],
                axes_end[len(axes_end) - 1 - i],
                self.voxel_size[len(axes_end) - 1 - i],
                dtype=np.float32,
            )

        return axes

    def write_to_ctegsphant(self, fileName: str):
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
                topleft:np.ndarray          #Not Written
                axis:np.ndarray             [z, y, x] -> [x, y, z]
                material_matrix:np.ndarray  [z, y, x] -> [x, y, z]
                density_matrix:np.ndarray   [z, y, x] -> [x, y, z]

            - fileName := the directory path where the file will be written
        """
        fileName = os.path.abspath(fileName)
        num_materials = str(self.num_materials) + "\n"
        materials = "\n".join(self.material_dict.keys()) + "\n"
        spacing = "0 0 0 0 0 0 0 0 0\n"
        dimensions = " ".join(map(str, self.num_voxels.astype(int))) + "\n"
        x_axis = " ".join(map(str, np.round(self.axis[2], decimals=3))) + "\n"
        y_axis = " ".join(map(str, np.round(self.axis[1], decimals=3))) + "\n"
        z_axis = " ".join(map(str, np.round(self.axis[0], decimals=3))) + "\n"
        material_matrix = _to_single_string(self.material_matrix.astype(str))
        density_matrix = _to_single_string(
            self.density_matrix.astype(str), " ")

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
            np.concatenate(self.axis),
            np.concatenate(new_BrachyEgsphant.axis),
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
            self.topleft, new_BrachyEgsphant.topleft, rtol=1e-3
        ).all(), "topleft is not the same"

        return (
            np.array_equal(self.material_matrix,
                           new_BrachyEgsphant.material_matrix)
            and np.array_equal(self.density_matrix, new_BrachyEgsphant.density_matrix)
            and np.isclose(
                np.concatenate(self.axis),
                np.concatenate(new_BrachyEgsphant.axis),
                rtol=1e-3,
            ).all()
            and np.array_equal(self.num_materials, new_BrachyEgsphant.num_materials)
            and self.material_dict == new_BrachyEgsphant.material_dict
            and np.array_equal(self.num_voxels, new_BrachyEgsphant.num_voxels)
            and np.array_equal(self.voxel_size, new_BrachyEgsphant.voxel_size)
            and np.array_equal(self.topleft, new_BrachyEgsphant.topleft)
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
        assert self.topleft is not None, "error: topleft is None"
        assert self.axis is not None, "error: axis is None"

    def info(self):
        self.assert_BrachyEgsphant_notEmpty()
        print(f"shape of material matrix is: {self.material_matrix.shape}")
        print(f"shape of density matrix is: {self.density_matrix.shape}")
        print(f"num voxels attribute is: {self.num_voxels}")
        print(f"the top left (bottom left in reality) is {self.topleft}")
        print(f"the voxel size is {self.voxel_size}")
        print(
            f"the size of the z, y and x axes are {self.axis[0].shape, self.axis[1].shape, self.axis[2].shape}"
        )
        print(
            f"the range of the z axis is {self.axis[0][0], self.axis[0][-1]}")
        print(
            f"the range of the y axis is {self.axis[1][0], self.axis[1][-1]}")
        print(
            f"the range of the x axis is {self.axis[2][0], self.axis[2][-1]}")
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
                it will also update the num_voxels, topleft and axis. only voxel_size will not change
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
                new_origin_index[2]: new_ending_index[2],  # z
                new_origin_index[1]: new_ending_index[1],  # y
                new_origin_index[0]: new_ending_index[0],  # x
            ]
            self.density_matrix = self.density_matrix[
                new_origin_index[2]: new_ending_index[2],  # z
                new_origin_index[1]: new_ending_index[1],  # y
                new_origin_index[0]: new_ending_index[0],  # x
            ]
            self.topleft = np.array(
                [
                    self.axis[2][new_origin_index[0]],  # x
                    self.axis[1][new_origin_index[1]],  # y
                    self.axis[0][new_origin_index[2]],  # z
                ]
            )
            self.num_voxels = np.flip(self.material_matrix.shape, 0)
            self.axis = self.calculateAxis()
        else:
            new_obj = BrachyEgsphant()
            new_obj.material_matrix = self.material_matrix[
                new_origin_index[2]: new_ending_index[2],
                new_origin_index[1]: new_ending_index[1],
                new_origin_index[0]: new_ending_index[0],
            ]
            new_obj.density_matrix = self.density_matrix[
                new_origin_index[2]: new_ending_index[2],
                new_origin_index[1]: new_ending_index[1],
                new_origin_index[0]: new_ending_index[0],
            ]
            new_obj.topleft = np.array(
                [
                    self.axis[2][new_origin_index[0]],  # x
                    self.axis[1][new_origin_index[1]],  # y
                    self.axis[0][new_origin_index[2]],  # z
                ]
            )
            new_obj.material_dict = self.material_dict
            new_obj.num_voxels = np.flip(new_obj.material_matrix.shape, 0)
            new_obj.voxel_size = self.voxel_size
            new_obj.axis = self.calculateAxis()
            new_obj.num_materials = self.num_materials
            return new_obj

    def crop_by_coordinates(self,
                            coordinate_range: np.array, inplace: Optional[bool] = True):
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
                it will also update the num_voxels, topleft and axis. only voxel_size will not change
        Dependencies:
            - None
        """
        crop_indices = np.zeros((3, 2), dtype=int)

        for i in range(3):
            origin = self.topleft[i]
            for j in range(2):
                crop_indices[i][j] = int(
                    (coordinate_range[i][j]-origin)/self.voxel_size[i])

        return self.crop_by_index(crop_indices, inplace=inplace)

    def crop_by_body_contour(
        self,
        body_index_range: Optional[np.ndarray] = None,
        body_mask_shape: Optional[np.ndarray] = None,
        pth_dir_dicom: Optional[str] = None,
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
                    in the dicom structure file. It will also update the num_voxels, topleft and axis. only voxel_size will not change
        """

        if body_index_range is None or body_mask_shape is None:
            assert (
                pth_dir_dicom is not None
            ), "Either path to a dicom directory with dicom structure \
                file should be given or body_index_range and body_mask_shape"
            body_mask_info = BrachyDicom(
                pth_dir_dicom, query_structure_list=["body"])
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
        matrix_single_string.append(
            deliminator.join(slide_single_string) + "\n")

    return "".join(matrix_single_string)


# app = typer.Typer()


def _load_json(pth_json: str):
    assert os.path.exists(
        pth_json
    ), f"no such json file was found at this directory: \n {pth_json}"

    with open(pth_json, "r") as file_json:
        return json.load(file_json)


# if __name__=="__main__":
#     app()
