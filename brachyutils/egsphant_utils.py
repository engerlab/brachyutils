import copy
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk
from opentps.core.data.images import Image3D
from scipy.interpolate import RegularGridInterpolator

from brachyutils.geometry_utils import BrachyPhantom


class BrachyEgsphant:
    r"""
    Purpose:
        - An object to allow for loading and manipulating the .egsphant files
    Attributes:
        - material_image: opentps.core.data.images.Image3D [x, y, z] := a 3D image object holding material per voxel
        - density_image: opentps.core.data.images.Image3D [x, y, z] := a 3D image object holding density per voxel
        - num_materials: int := the number of different material composition options a voxel has
        - material_dict: dict := a dictionary containing the name of the elements for each voxel,
        and the following keys: [
            "density" := the density of the material in g/cm^3,
            "HU_limit" := the lower HU limit threshold of the material,
            "structure_name := {optional} the name of the structure in the dicom file that represents the material,"
            ]
        - unit_length: str := the unit of the length of the axis is mm.
        - voxel_edges: np.ndarray := the edges of the voxels in the material and density matrix
        - xyz_format: bool := if True, the axis is in the format [x, y, z], if False, the axis is in the format [z, y, x]
    Functions:
        - load_file_to_BrachyEgsphant()     done
        - load_from_ctegsphant()            done
        - load_from_nrrd()                  done
        - get_voxel_edges()                 done
        - write_to_ctegsphant()             done
        - write_to_nrrd()                   done
        - crop_by_index()                   done
        - crop_by_coordinates()             done
        - crop_by_contour()                 done
        - is_not_empty()                    done
        - info()                            done
        - is_equal()                        done
        - create_egsphant_from_phantom()    done
        - create_interpolation_function()   done
        - get_voxel_centers()               done
        - sort_materials_by()               done
        - export_material_dict()            done
        - _remove_duplicate_materials()     done

    Dependencies:
        - opentps
    """

    # each voxel in the material matrix is encoded with a single character
    # from this array that represents a unique material recognized by RapidBrachyMC.
    _materials_encoding_array = [str(i) for i in range(1, 10)] + [
        chr(i) for i in range(ord("A"), ord("Z") + 1)
    ]

    def __init__(
        self,
        pth_egsphant_file: Optional[Path] = None,
        phantom: Optional[Union[BrachyPhantom | Path]] = None,
        material_dict: Optional[Union[dict, Path]] = None,
        assign_material_from_ct: Optional[bool] = None,
    ) -> None:
        r"""
        Purpose:
            - to initialize a BrachyEgsphant object, which represents the material and density matrix of a phantom.
            if loading from a file, do not provide any other input other than pth_egsphant_file.
        Inputs:
            - pth_egsphant_file:Path := the directory path to the .egsphant file
            - phantom:BrachyPhantom := a BrachyPhantom object containing the structure mask
            - material_dict: dict := a dictionary containing the name of the elements for each voxel,
                and the following keys: [
                    "density" := the density of the material in g/cm^3,
                    "HU_limit" := the lower HU limit threshold of the material,
                    "structure_name := {optional} the name of the structure in the dicom file that represents the material,"
                ]
            - assign_material_from_ct:bool := if True, the function will assign the material based on the HU values in the CT image.
        Outputs:
            - Void := will initialize a BrachyEgsphant object
        """
        self.unit_length = "mm"
        self.material_image: Image3D = None

        self.density_image: Image3D = None

        self.num_materials: int = None
        self.material_dict: defaultdict = defaultdict(dict)
        self.material_dict["Air"] = {
            "encoding": BrachyEgsphant._materials_encoding_array[0],
            "density": 0.001225,
            "HU_limit": -1000.0,
        }

        self._sanity_axis: np.ndarray = None
        self.voxel_edges: np.ndarray = None
        self.xyz_format: bool = True

        if pth_egsphant_file is not None:
            self.load_file_to_BrachyEgsphant(pth_egsphant_file)

        elif phantom is not None and material_dict is not None:

            if isinstance(material_dict, str):
                if (
                    os.path.splitext(material_dict)[-1] == ".txt"
                    and not assign_material_from_ct
                ):
                    raise Exception(
                        "CT to density text file should be used only when assign_material_from_ct is True.\n \
                        If assigning materials by contours, please provide a json file or a dictionary containing structure_name\
                        for each material."
                    )
                material_dict = Path(material_dict)

            self.material_dict = self.material_dict | _load_material_dict(material_dict)
            self._remove_duplicate_materials()

            self.create_egsphant_from_phantom(
                phantom_obj=(
                    phantom
                    if isinstance(phantom, BrachyPhantom)
                    else BrachyPhantom(phantom)
                ),
                new_material_dict=self.material_dict,
                assign_material_from_ct=assign_material_from_ct,
            )
        else:
            raise Exception(
                "Either provide a path to an egsphant file or a dicom image and a material dictionary"
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
            gridSize = np.array([int(i) for i in egsphant.readline().strip().split()])

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
            # convert sanity axis to z, y, x: no need to flip the axis anymore. everything is xyz
            # self._sanity_axis = np.flip(self._sanity_axis, axis=0)
            # convert sanity axis from cm to mm
            self._sanity_axis = self._sanity_axis * 10
            # remove the last entry in each axis because it is one more than
            # there are desity or material values on that axis.
            self._sanity_axis = np.array(
                [axis[:-1] for axis in self._sanity_axis], dtype=object
            )
            spacing = np.array(
                [
                    self._sanity_axis[0][1] - self._sanity_axis[0][0],
                    self._sanity_axis[1][1] - self._sanity_axis[1][0],
                    self._sanity_axis[2][1] - self._sanity_axis[2][0],
                ]
            )
            origin = np.array(
                [
                    self._sanity_axis[0][0] + spacing[0] / 2,
                    self._sanity_axis[1][0] + spacing[0] / 2,
                    self._sanity_axis[2][0] + spacing[0] / 2,
                ],
                dtype=np.float32,
            )

            # prepare empty matricies to hold material and density images
            material_matrix = np.zeros(
                (gridSize[2], gridSize[1], gridSize[0]), dtype=str
            )
            density_matrix = np.zeros(
                (gridSize[2], gridSize[1], gridSize[0]),
                dtype=np.float32,
            )

            # load the material composition data in to the matrix
            for k in range(gridSize[2]):
                for j in range(gridSize[1]):
                    material_matrix[k][j] = list(egsphant.readline().strip())
                egsphant.readline()

            # load the density data into the matrix
            for k in range(gridSize[2]):
                for j in range(gridSize[1]):
                    density_matrix[k][j] = egsphant.readline().strip().split()
                egsphant.readline()

            material_matrix = _convert_material_matrix_to(material_matrix, dtype=int)

            self.material_image = Image3D(
                # convert array from zyx to xyz.
                imageArray=np.swapaxes(material_matrix, 0, 2),
                origin=origin,
                spacing=spacing,
            )
            self.density_image = Image3D(
                # convert array from zyx to xyz.
                imageArray=np.swapaxes(density_matrix, 0, 2),
                origin=origin,
                spacing=spacing,
            )
            # this line maybe useless in the future
            self.voxel_edges = self.get_voxel_edges()
            # {for debugging
            # print(f"The axis calculated from get_voxel_edges() are \n {self.voxel_edges}")
            # print(f"The axis from the text file are: \n {self._sanity_axis}")
            # print(f"the size of the axis in the z, y, x for axis from calcAxis() are {self.voxel_edges[0].shape}, {self.voxel_edges[1].shape}, {self.voxel_edges[2].shape}")
            # print(f"the size of the axis in the z, y, x for axis from file are {self._sanity_axis[0].shape}, {self._sanity_axis[1].shape}, {self._sanity_axis[2].shape}")
            # }
            assert np.isclose(
                np.concatenate(self.voxel_edges),
                np.concatenate(self._sanity_axis),
                rtol=1e-1,
            ).all(), "axis is not the same"

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

        if len(self.material_dict.keys()) == 1:
            return  # don't sort when there's only one material
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

    def load_from_nrrd(self, filePath: Path):
        r"""
        Purpose:
            to load a nrrd file containing egsphant data.
        Input:
            - pth_file := directory path to the .nrrd file
        """
        if not os.path.exists(filePath):
            raise ValueError(f"The target nrrd file {filePath} does not exist!")

        image = sitk.ReadImage(filePath)
        # gridSize = np.array(image.GetSize(), dtype=int)  # [::-1]
        spacing = np.array(image.GetSpacing(), dtype=float)  # [::-1]
        # origin_coordinates is the bottom left corner of the image
        # but in sitk, it's the center of the first voxel
        origin = (
            np.array(image.GetOrigin(), dtype=float) - 0.5 * spacing
        )  # [::-1] - 0.5 * self.voxel_size

        material_matrix = sitk.GetArrayFromImage(image)[
            :, :, :, 0
        ]  # np.swapaxes([:, :, :, 0], 0, 2)
        density_matrix = sitk.GetArrayFromImage(image)[
            :, :, :, 1
        ]  # np.swapaxes(sitk.GetArrayFromImage(image)[:, :, :, 1], 0, 2)
        # self.num_materials = np.max(self.material_matrix).astype(int)

        try:  # try to load the material dictionary from the metadata
            self.material_dict = _load_material_dict(json.loads(image.GetMetaData("material_dict")))
            self.num_materials = len(self.material_dict.keys())  # if the material dict
            # is found, update a more accurate count of the number of materials
        except Exception:
            warnings.warn(
                "Material dictionary not found in the nrrd file. \
            Please provide the dictionary manually before exporting the file in \
            .egsphant format",
                stacklevel=2,
            )
            
        self.material_image = Image3D(
            # convert array from zyx to xyz.
            imageArray=np.swapaxes(material_matrix, 0, 2),
            origin=origin,
            spacing=spacing,
        )
        self.density_image = Image3D(
            # convert array from zyx to xyz.
            imageArray=np.swapaxes(density_matrix, 0, 2),
            origin=origin,
            spacing=spacing,
        )
        
        self.get_voxel_edges()
        self.unit_length = "mm"

    def get_voxel_edges(self):
        r"""
        Purpose: will calculate the axies coordinates for a BrachyEgsphant object.
        Input:
            - self := it should have the following keys and values:
                {"grid":,
                "origin_coordinates":,
                "voxel_size":}
        Output:
            - axes:numpy.array() :=
            [[x_min:voxel_size:x_max],
            [y_min:voxel_size:y_max],
            [z_min:voxel_size:z_max]]
        """
        assert self.density_image is not None, "density matrix is not loaded"
        voxel_centers = self.get_voxel_centers()
        self.voxel_edges = np.empty(len(voxel_centers), dtype=object)
        for i in range(len(voxel_centers)):
            self.voxel_edges[i] = voxel_centers[i] - self.density_image.spacing[i] / 2.0
        return self.voxel_edges

    def create_interpolation_function(self, grid):
        voxel_centers = self.get_voxel_centers()
        interpolation_function = RegularGridInterpolator(
            (voxel_centers[0], voxel_centers[1], voxel_centers[2]),
            grid,
            bounds_error=False,
            fill_value=0,
        )
        return interpolation_function

    def get_voxel_centers(self):
        r"""
        Purpose:
            - to calculate the center of each voxel in the BrachyEgsphant object.
        Output:
            - voxel_centers:np.ndarray := the center of each voxel in the BrachyEgsphant object.
        """
        assert self.density_image is not None, "density matrix is not loaded"
        voxel_centers = np.empty(len(self.density_image.origin), dtype=object)
        for i in range(len(self.density_image.origin)):
            voxel_centers[i] = (
                self.density_image.origin[i]
                + np.arange(self.density_image.gridSize[i])
                * self.density_image.spacing[i]
            )
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
                density_image.gridSize:np.ndarray       [x, y, z]
                voxel_edges:np.ndarray                  [x, y, z]
                material_matrix:np.ndarray              [x, y, z]
                density_matrix:np.ndarray               [x, y, z]

            - fileName := the directory path where the file will be written
        """
        assert (
            os.path.splitext(fileName)[-1] == ".egsphant"
        ), "file extension is not .egsphant"
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        egsphant_voxel_edges = (
            np.array(
                [
                    np.append(axis, axis[-1] + self.density_image.spacing[i])
                    for i, axis in enumerate(self.voxel_edges)
                ],
                dtype=object,
            )
            / 10
        )
        self._sort_materials_by("encoding")
        num_materials = str(self.num_materials) + "\n"
        materials = "\n".join(self.material_dict.keys()) + "\n"
        spacing = "0 0 0 0 0 0 0 0 0\n"
        dimensions = " ".join(map(str, self.density_image.gridSize.astype(int))) + "\n"
        x_axis = (
            " ".join(
                map(str, np.round(egsphant_voxel_edges[0].astype(float), decimals=3))
            )
            + "\n"
        )
        y_axis = (
            " ".join(
                map(str, np.round(egsphant_voxel_edges[1].astype(float), decimals=3))
            )
            + "\n"
        )
        z_axis = (
            " ".join(
                map(str, np.round(egsphant_voxel_edges[2].astype(float), decimals=3))
            )
            + "\n"
        )
        material_matrix = self.get_material_array()
        material_matrix = _to_single_string(
            _convert_material_matrix_to(material_matrix, dtype=str), ""
        )
        density_matrix = self.get_density_array()
        density_matrix = _to_single_string(density_matrix.astype(str), " ")

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

    def write_to_nrrd(self, fileName: Path, metadata: Optional[dict] = None):
        r"""
        Purpose:
            To save the contents of an egsphant as a nrrd file.
        inputs:
            - fileName := path where the dose nrrd file will be written to.

            - metadata := a dictionary containing the following meta data key values:
                "material_dict:" {material_name: {"encoding": int, "density": float, "HU_limit": float}}
                "Image content": "[material_matrix, density_matrix]"
        outputs: Void
            writes [material_matrix, density_matrix], voxel size, origin (origin_coordinates), and metadata to the file_name_dose.nrrd
            note that 3D dose files are written in z, y, x, but the sitk image is written in x, y, z.
        """
        # create sitk dose image
        material_grid = (
            self.get_material_array().astype(np.float32)
        )  # np.swapaxes(self.material_matrix, 0, 2).astype(np.float32)
        density_grid = (
            self.get_density_array().astype(np.float32)
        )  # np.swapaxes(self.density_matrix, 0, 2).astype(np.float32)

        compose_filter = sitk.ComposeImageFilter()
        image_nrrd = compose_filter.Execute(
            sitk.GetImageFromArray(material_grid), sitk.GetImageFromArray(density_grid)
        )

        image_nrrd.SetOrigin(
            (self.density_image.origin + self.density_image.spacing / 2).astype(float)
        )  # [::-1])
        image_nrrd.SetSpacing(self.density_image.spacing.astype(float))  # [::-1])

        # write the static metadata that will be written to all nrrd files
        static_metadata = {}
        static_metadata["Image content"] = "[material_matrix, density_matrix]"
        if self.material_dict is not None:
            static_metadata["material_dict"] = self.material_dict

        for key in static_metadata:
            image_nrrd.SetMetaData(key, json.dumps(static_metadata[key]))
        # set the metadata: all sitk Images belonging to a patient will have the same meta data
        if metadata is not None:
            for key in metadata:
                image_nrrd.SetMetaData(key, metadata[key])

        # write out the files
        if not os.path.exists(os.path.dirname(fileName)):
            raise ValueError(
                f"the input folder does not exist: {os.path.dirname(fileName)}"
            )

        sitk.WriteImage(image_nrrd, fileName, useCompression=True, compressionLevel=9)

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
        if not isinstance(new_BrachyEgsphant, BrachyEgsphant):
            warnings.warn("input must be of type BrachyEgsphant", stacklevel=2)
            return False
        elif not np.array_equal(
            self.material_image.imageArray, new_BrachyEgsphant.material_image.imageArray
        ):
            warnings.warn("material matrix is not the same", stacklevel=2)
            return False
        elif not np.array_equal(
            self.density_image.imageArray, new_BrachyEgsphant.density_image.imageArray
        ):
            warnings.warn("density matrix is not the same", stacklevel=2)
            return False
        elif not np.isclose(
            np.concatenate(self.voxel_edges),
            np.concatenate(new_BrachyEgsphant.voxel_edges),
            rtol=1e-3,
        ).all():
            warnings.warn("axis is not the same", stacklevel=2)
            return False
        elif not np.array_equal(self.num_materials, new_BrachyEgsphant.num_materials):
            warnings.warn("number of materials is not the same", stacklevel=2)
            return False
        elif not (self.material_dict == new_BrachyEgsphant.material_dict):
            warnings.warn("the material dictionary is not the same", stacklevel=2)
            return False
        elif not np.array_equal(
            self.density_image.gridSize, new_BrachyEgsphant.density_image.gridSize
        ):
            warnings.warn("num_voxels is not the same", stacklevel=2)
            return False
        elif not np.isclose(
            self.density_image.spacing,
            new_BrachyEgsphant.density_image.spacing,
            atol=1e-3,
        ).all():
            warnings.warn("voxel_size is not the same", stacklevel=2)
            return False
        elif not np.isclose(
            self.density_image.origin,
            new_BrachyEgsphant.density_image.origin,
            rtol=1e-3,
        ).all():
            warnings.warn("origin_coordinates is not the same", stacklevel=2)
            return False
        else:
            return True

    def is_not_empty(self):
        r"""
        Purpose:
            to see which field of a brachyEgsphant object is empty
        """
        assert self.material_image is not None, "error: material_matrix is None"
        assert self.density_image is not None, "error: density_matrix is None"
        assert self.num_materials is not None, "error: num_materials is None"
        assert self.material_dict is not None, "error: material_dict is None"
        assert self.voxel_edges is not None, "error: axis is None"

    def info(self):
        self.is_not_empty()
        print(
            f"grid size of material density matrix are {self.material_image.gridSize, self.density_image.gridSize}"
        )
        print(f"grid size in world units is {self.density_image.gridSizeInWorldUnit}")
        print(
            f"spacing of material and density matrix is {self.material_image.spacing, self.density_image.spacing}"
        )
        print(
            f"origin of material and density matrix is {self.material_image.origin, self.density_image.origin}"
        )
        print(
            f"the size of the x, y and z axes are {self.voxel_edges[0].shape, self.voxel_edges[1].shape, self.voxel_edges[2].shape}"
        )
        print(
            f"the range of the x axis is {self.voxel_edges[0][0], self.voxel_edges[0][-1]}"
        )
        print(
            f"the range of the y axis is {self.voxel_edges[1][0], self.voxel_edges[1][-1]}"
        )
        print(
            f"the range of the z axis is {self.voxel_edges[2][0], self.voxel_edges[2][-1]}"
        )
        print(f"The number of materials is {self.num_materials}")
        print(f"the material dictionary is {self.material_dict}")

    def crop_by_index(
        self, index_range: np.ndarray, inplace: Optional[bool] = True
    ) -> Union[None, "BrachyEgsphant"]:
        r"""
        Purpose:
            given a range of indicies (mix and max on each axis), this function will crop
            material and density matricies and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyEgsphant object
            - index_range := a 3 x 2 array holding the min and max index on x, y and z axis
                [[ix_min, ix_max], [iy_min, iy_max], [iz_min, iz_max]]
        Output:
            - Void := will crop out the material and density maps of self to have the range of the index range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            - self.crop_by_coordinates()
        """
        assert index_range.shape == (
            3,
            2,
        ), "index_range should be a 3x2 array in x, y, z order"
        assert np.all(
            self.density_image.gridSize == self.material_image.gridSize
        ), "material and density matrix should have the same size"
        new_origin_coords = self.density_image.getPositionFromVoxelIndex(
            index_range[:, 0]
        )
        new_ending_coords = self.density_image.getPositionFromVoxelIndex(
            index_range[:, 1]
        )
        new_coords_range = np.column_stack([new_origin_coords, new_ending_coords])
        return self.crop_by_coordinates(new_coords_range, inplace)

    def crop_by_coordinates(
        self, coordinate_range: np.array, inplace: Optional[bool] = True
    ) -> Union[None, "BrachyEgsphant"]:
        r"""
        Purpose:
            given a range of coordinates (mix and max on each axis), this function will crop
            material and density matricies and will adjust the rest of the attributes accordingly.
        Inputs:
            - self: BrachyEgsphant object
            - coordinate_range := a 3 x 2 array holding the min and max on x, y and z axis
                [[ix_min, ix_max], [iy_min, iy_max], [iz_min, iz_max]]
        Output:
            - Void := will crop out the material and density maps of self to have the range of the index range.
                it will also update the num_voxels, origin_coordinates and axis. only voxel_size will not change
        Dependencies:
            - None
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
            crop3DDataAroundBox(self.material_image, coordinate_range)
            crop3DDataAroundBox(self.density_image, coordinate_range)
            self.get_voxel_edges()
        else:
            new_egsphant: BrachyEgsphant = copy.deepcopy(self)
            new_egsphant.crop_by_coordinates(coordinate_range, inplace=True)
            return new_egsphant

    def crop_by_contour(
        self,
        phantom_obj: BrachyPhantom,
        contour_name: str,
        inplace: Optional[bool] = True,
    ) -> Union[None, "BrachyEgsphant"]:
        r"""
        Purpose:
            - to crop the material and density matrix based on the contour of a structure in the phantom object.
        Inputs:
            - phantom_obj:BrachyPhantom := a BrachyPhantom object containing the structure mask
            - contour_name:str := the name of the structure in the phantom object
            - inplace:bool := if True, the function will crop the current object, if False, it will return a new object
        Output:
            - None or BrachyEgsphant := if inplace is True, the function will crop the current object, if False, it will return a new object
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

    def get_material_array(self):
        r"""
        Purpose:
            - to get the material matrix as a numpy array in [z, y, x].
        Output:
            - material_matrix:np.ndarray := the material matrix in [z, y, x]
        """
        return np.swapaxes(self.material_image.imageArray, 0, 2)

    def get_density_array(self):
        r"""
        Purpose:
            - to get the density matrix as a numpy array in [z, y, x].
        Output:
            - density_matrix:np.ndarray := the density matrix in [z, y, x]
        """
        return np.swapaxes(self.density_image.imageArray, 0, 2)

    def create_egsphant_from_phantom(
        self,
        phantom_obj: BrachyPhantom,
        new_material_dict: dict = None,
        assign_material_from_ct: bool = True,
        background_material: str = "Air",
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
            - BrachyPhantom
        """
        if not assign_material_from_ct:
            assert (
                phantom_obj.structure_set is not None
            ), "No structure mask was found. please load structure file into the phantom object"
        for material in new_material_dict:
            assert {"encoding", "density", "HU_limit"}.issubset(
                set(new_material_dict[material].keys())
            ), "material dictionary is not formatted correctly"
        if background_material not in new_material_dict:
            warnings.warn(
                f"Background material {background_material} not found in the material dictionary. Will default to Air",
                stacklevel=2,
            )
        self.material_dict = new_material_dict
        material_matrix = np.ones_like(phantom_obj.get_image_array(), dtype=int)
        density_matrix = np.ones_like(phantom_obj.get_image_array(), dtype=np.float32)
        self.num_materials = len(self.material_dict)

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
                        phantom_obj.get_image_array() >= low_HU_threshold,
                        1,
                        0,
                    ).astype(bool),
                    np.where(
                        phantom_obj.get_image_array() < high_HU_threshold,
                        1,
                        0,
                    ).astype(bool),
                )
                # set the density and material of all voxels outside the lowest HU_limit to air
                if i == 0:
                    complementary_roi_mask = np.logical_not(roi_mask)
                    density_matrix *= roi_mask
                    material_matrix *= roi_mask
                    density_matrix += complementary_roi_mask * self.material_dict.get(
                        background_material, "Air"
                    ).get("density")
                    material_matrix += (
                        complementary_roi_mask
                        * BrachyEgsphant._materials_encoding_array.index(
                            self.material_dict.get(background_material, "Air").get(
                                "encoding"
                            )
                        )
                    )

                # reset the voxel values for the roi enetries
                density_matrix *= np.logical_not(roi_mask)
                material_matrix *= np.logical_not(roi_mask)

                # update the density and material matricies
                # interpolate density based on the HU value
                density_matrix += (
                    phantom_obj.get_image_array() * roi_mask * slope_density_over_HU
                    + intercept_density_over_HU
                )
                material_matrix += (
                    roi_mask
                    * BrachyEgsphant._materials_encoding_array.index(
                        self.material_dict.get(material).get("encoding")
                    )
                )
                # assert np.all(density_matrix >= 0), "density matrix has negative values"
        else:
            # dicom_structure_list = list(phantom_obj.structure_mask_dict.keys())
            # find the materials that have a structure name with them.
            query_structure_list = []
            for material in self.material_dict.values():
                if material.get("structure_name") is None:
                    continue
                else:
                    query_structure_list.append(material.get("structure_name"))

            # get the mask of each material from image
            mask_dict = phantom_obj.get_structure_mask(
                query_structure_list, mask_type=np.ndarray
            )
            for material in self.material_dict:
                if self.material_dict.get(material).get("structure_name") is None:
                    continue
                self.material_dict.get(material)["structure_size"] = np.sum(
                    mask_dict.get(
                        self.material_dict.get(material).get("structure_name")
                    )
                )

            # sort the material dictionary based on the size of the mask (from largest to smallest)
            self._sort_materials_by("structure_size")

            for i, material in enumerate(self.material_dict.keys()):
                if self.material_dict.get(material).get("structure_name") is None:
                    continue
                roi_mask = mask_dict.get(
                    self.material_dict.get(material).get("structure_name")
                ).astype(bool)

                # set everything outside the largest mask to air
                if i == 0:
                    complementary_roi_mask = np.logical_not(roi_mask)
                    density_matrix *= roi_mask
                    material_matrix *= roi_mask
                    density_matrix += complementary_roi_mask * self.material_dict.get(
                        background_material, "Air"
                    ).get("density")
                    material_matrix += (
                        complementary_roi_mask
                        * BrachyEgsphant._materials_encoding_array.index(
                            str(
                                self.material_dict.get(background_material, "Air").get(
                                    "encoding"
                                )
                            )
                        )
                    )

                # reset the voxel values for the roi enetries
                density_matrix *= np.logical_not(roi_mask)
                material_matrix *= np.logical_not(roi_mask)

                # update the density and material matricies
                density_matrix += roi_mask * self.material_dict.get(material).get(
                    "density"
                )
                material_matrix += (
                    roi_mask
                    * BrachyEgsphant._materials_encoding_array.index(
                        str(self.material_dict.get(material).get("encoding"))
                    )
                )

        self.material_image = Image3D(
            imageArray=np.swapaxes(material_matrix, 0, 2),
            origin=phantom_obj.image_obj.origin,
            spacing=phantom_obj.image_obj.spacing,
            angles=phantom_obj.image_obj.angles,
        )
        self.density_image = Image3D(
            imageArray=np.swapaxes(density_matrix, 0, 2),
            origin=phantom_obj.image_obj.origin,
            spacing=phantom_obj.image_obj.spacing,
            angles=phantom_obj.image_obj.angles,
        )
        self.voxel_edges = self.get_voxel_edges()

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

    def _remove_duplicate_materials(self):
        r"""
        Purpose:
            To remove duplicate materials from the material dictionary.
        Inputs:
            - self:BrachyEgsphant := a BrachyEgsphant object with a material dictionary
        Outputs:
            - Void := will remove duplicate materials from the material dictionary.
        """
        assert self.material_dict is not None, "material dictionary is not available"

        material_list = list(self.material_dict.keys())
        material_list = [x.lower() for x in material_list]

        # remove duplicated materials self.material_dict
        for material in self.material_dict:
            if material_list.count(material.lower()) > 1 and material != "Air":
                print(
                    f"Duplicate material {material} found in the material dictionary and removed"
                )
                self.material_dict.pop(material)

        # reset the encoding of the materials in the material dictionary
        for i, material in enumerate(self.material_dict):
            self.material_dict.get(material)["encoding"] = (
                BrachyEgsphant._materials_encoding_array[i]
            )


def _convert_material_matrix_to(
    material_matrix: np.ndarray, dtype: Union[int, str]
) -> np.ndarray:
    r"""
    Purpose:
        To convert a numpy array of dtype string to an integer numpy array or the other way around.
        Integer array is the desired data type over string since it allows for more operational functionality.
        String array is desired for outputting the egsphant file.
    Inputs:
        - self.material_matrix:np.array(dtype=str) := a numpy array with string enteries
        - BrachyEgsphant._encoding_array:list := a list of strings that will be used to encode the string enteries
    Outputs:
        - np.array(dtype=int) := a numpy array with integer or string enteries
    """
    # assert dtype in [int, str], "dtype is not recognized"

    flattened_array = material_matrix.flatten()

    if dtype is int:

        int_array = np.zeros_like(flattened_array, dtype=int)

        for i, string in enumerate(flattened_array):
            int_array[i] = BrachyEgsphant._materials_encoding_array.index(string)

        return int_array.reshape(material_matrix.shape)

    elif dtype is str:
        str_array = np.zeros_like(flattened_array, dtype=str)
        for i, integer in enumerate(flattened_array):
            str_array[i] = BrachyEgsphant._materials_encoding_array[integer]

        return str_array.reshape(material_matrix.shape)
    else:
        raise Exception("dtype is not recognized")


def _to_single_string(
    matrix: np.ndarray,
    delimiter: Optional[str] = "",
    add_terminating_newline: Optional[bool] = True,
):
    r"""
    Purpose:
        given a 3D matrix with string entries, this function concatenates all the
            entries into a single string to be written to the file.
            "\n" is added at the end of each row and
    Input:
        matrix := 3D ndarray full of string enteries
        delimiter := the string text inbetween the enteries.
        add_terminating_newline := if True, an additional \n will be added at the end of the string
    Output:
        a single string containing all the entries with added \n at the end of each row of
            matrix and an additional \n added to each slice in the input matrix

    """
    zslice_strings = []
    for zslice in matrix:
        yrow_strings = []
        for yrow in zslice:
            x_row_str = delimiter.join(yrow)
            yrow_strings.append(x_row_str)
        zslice_strings.append("\n".join(yrow_strings))
    full_string = "\n\n".join(zslice_strings)
    if add_terminating_newline:
        full_string += "\n\n"
    return full_string


def _load_json(pth_json: Path):
    assert os.path.exists(
        pth_json
    ), f"no such json file was found at this directory: \n {pth_json}"

    with open(pth_json, "r") as file_json:
        return json.load(file_json)


def _load_material_dict(material_source: Union[Path, dict]):
    r"""
    Purpose:
        To load material dictionary and give it the proper keys from simple material dictionary,
        a ct to density.txt file or from a json file that contains the density and HU lower
        limit threshold for each material.
    Inputs:
        - material_source := directory path to the ct2density.txt file, json file or the material dictionary
    Outputs:
        - dict := a dictionary containing the density and HU lower limit thresholds for each material.
    """
    if isinstance(material_source, Path) or isinstance(material_source, str):
        pth_file = material_source
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
            raise Exception("file extension is not recognized")
    elif isinstance(material_source, dict):
        material_dict = material_source
    else:
        raise Exception(
            "material source is not recognized, please provide the dictionary, json file or ct2density.txt file"
        )

    for i, material in enumerate(material_dict):
        assert (
            material_dict.get(material).get("density") is not None
        ), "density is not available"

        if material_dict.get(material).get("HU_limit") is None:
            warnings.warn(
                f"no HU limit was found for {material}, material assignment by ct will not be possible",
                stacklevel=2,
            )
            material_dict.get(material)["HU_limit"] = float("-inf")

        if material_dict.get(material).get("encoding") is None:
            warnings.warn(
                f"no encoding was found for {material}, encoding will be set by the order of the material in the json file",
                stacklevel=2,
            )
            material_dict.get(material)["encoding"] = (
                BrachyEgsphant._materials_encoding_array[i]
            )

    return material_dict
