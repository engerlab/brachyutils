import numpy as np
import os
import json
from pathlib import Path
from typing import Union, List
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

class BrachyApplicator:
    r"""
    Purpose:
        - This class holds the information regarding the brachytherapy applicator.
        as well as all the functions to support the necessary applicator operations.

    Attributes:
        - path:str := path to the applicator geometry file.
        - name:str := name of the applicator, which is taken as the basename of the path.
        - applicator_mesh := the vtk mesh of the applicator.
        - vertices:np.array := the vertices of the applicator mesh.
        - faces:np.array := the faces of the applicator mesh.
        - origin:np.array := the origin of the applicator.
        - rotation:np.array := the rotation of the applicator.
        - material:str := the material of the applicator.
        - density:float := the density of the applicator.

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
            - catheter_trajectory: (list, optional): The list of start dwell poisition and end dwell position of the catheter inside
            the applicator [[x,y,z,x,y,z]]. Defaults to None.
        Outputs:
            - None: an applicator object is created dependeing on the inputs.
        """
        assert os.path.exists(
            pth_input_file
        ), f"input file {pth_input_file} does not exist"
        self.path = pth_input_file
        self.name = os.path.splitext(os.path.basename(self.path))[0] 
        self.applicator_mesh: vtkPolyData = None
        self.vertices: np.array = None
        self.faces: np.array = None
        self.origin: np.array = np.array([0, 0, 0])  # [x, y, z]
        self.rotation: np.array = np.array([0, 0, 0, 0])  # [w, x, y, z]
        self.coordinates: np.array = np.array([0, 0, 0])  # [x, y, z]
        self.material: str = None
        self.density: float = None
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
        if catheter_trajectory is not None:
            self.catheter_trajectory = catheter_trajectory

    def load_stl(self, pth_input: str) -> None:
        r"""
        Purpose:
            - To load the applicator geometry from an stl file.
        Inputs:
            - pth_input:str := path to the stl file containing the applicator geometry.
        Outputs:
            - None := will update the BrachyApplicator object based on the stl file.
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
            - None := will update the BrachyApplicator object based on the json file.
        """
        with open(pth_input, "r") as json_file:
            applicator_dict = json.load(json_file)

        self.vertices = np.array(applicator_dict["vertices"], dtype=np.float32)
        self.faces = np.array(applicator_dict["faces"], dtype=np.int32)
        self.set_origin(np.array(applicator_dict["origin"]))
        self.set_rotation(np.array(applicator_dict["rotation"]))
        self.set_coordinates(np.array(applicator_dict["coordinates"]))
        self.material = applicator_dict["material"]
        self.density = applicator_dict["density"]

    def info(self) -> None:
        r"""
        Purpose:
            - To print the information about the applicator.
        Inputs:
            - self := the BrachyApplicator object.
        Outputs:
            - None := will print the information about the applicator.
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
        if not np.isclose(self.vertices, other.vertices, atol=1e-6).all():
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
            - To update the applicator mesh from the vertices and faces.
        Inputs:
            - self := the BrachyApplicator object.
        Outputs:
            - None := will update the applicator mesh from the vertices and faces.
        """
        points = vtkPoints()
        for vertex in self.vertices:
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
            - None := will update the brachy applicator from the applicator mesh.
        """
        self.vertices = numpy_support.vtk_to_numpy(
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
            - None := will update the applicator vertices based on the new origin.
        """
        old_origin = self.origin
        change_in_origin = np.ones_like(self.vertices) * (origin - old_origin)
        self.origin = origin
        self.vertices += change_in_origin
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
            - None := will update the applicator vertices based on the new rotation.
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
            - None := will update the applicator vertices based on the new coordinates.
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
            - None := will update the catheter trajectory.
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
            # "vertices": self.vertices.tolist(),
            # "faces": self.faces.tolist(),
            "origin": self.origin,
            "rotation": self.rotation,
            "material": self.material,
            "density": self.density,
            "catheter_trajectory": self.catheter_trajectory,
        }

    def to_json(self, pth_output: str) -> None:
        r"""
        Purpose:
            - To save the applicator geometry to a json file.
        Inputs:
            - pth_output:str := path to the output json file.
        Outputs:
            - None := will save the applicator geometry to a json file.
        """
        applicator_dict = self.to_dict()

        with open(pth_output, "w") as json_file:
            json.dump(applicator_dict, json_file, indent=4)

    def to_stl(self, pth_output: str) -> None:
        r"""
        Purpose:
            - To save the applicator geometry to an stl file.
        Inputs:
            - pth_output:str := path to the output stl file.
        Outputs:
            - None := will save the applicator geometry to an stl file.
        """
        self._update_applicator_mesh_from_brachy_applicator()
        # write the polydata to an stl file
        stl_writer = vtkSTLWriter()
        stl_writer.SetFileName(pth_output)
        stl_writer.SetInputData(self.applicator_mesh)
        stl_writer.Write()

def write_applicator_geometry(self, pth_output: str | Path = Path("./applicator_geometry.json")) -> None:
    r"""
    ### Purpose:
    - To export the applicator geometries.

    ### Inputs:
    - dir_export := path to the directory where the export happens

    ### Outputs:
    - None := will export the applicator geometries into the specified export directory.

    ### Dependencies:
    - None
    """

        # initialize the fields of the json file:
    out_json = {
    "densities": [],
    "filenames": [],
    "materials": [],
    "points": [],
    "wRot": 0,
    "x": 0,
    "xRot": 0,
    "y": 0,
    "yRot": 0,
    "z": 0,
    "zRot": 0,
}
    counter = 0
    for applicator in self.applicator_list:

        out_json["densities"].append(applicator.density)
        out_json["filenames"].append(applicator.path)
        out_json["materials"].append(applicator.material)
        out_json["points"].append(
            applicator.catheter_trajectory.flatten().tolist()
        )

        subscript = counter + 1 if counter >= 1 else ""
        out_json[f"wRot{subscript}"] = float(applicator.rotation[0])
        out_json[f"xRot{subscript}"] = float(applicator.rotation[1])
        out_json[f"yRot{subscript}"] = float(applicator.rotation[2])
        out_json[f"zRot{subscript}"] = float(applicator.rotation[3])

        out_json["x"] = float(applicator.coordinates[0])
        out_json["y"] = float(applicator.coordinates[1])
        out_json["z"] = float(applicator.coordinates[2])
        counter += 1

    with open(pth_output, "w") as file:
        json.dump(out_json, file, indent=4)

    print("applicator geometry file was exported successfully")


def load_applicator_list(
        self,
        applicator_list_pth: Union[Path, str],
    ) -> List[BrachyApplicator]:
        r"""
        ### Purpose:
        - To load the applicator list from a json file containing the applicator geometry.

        ### Inputs:
        - applicator_list_pth:str := path to the json file containing the applicator list with N applicators.
        The items inside this list have the attributes bellow. If any left empty, the default value will be used.
        these attributes could be changed later using the setter functions.

        if the format is RapidBrachy, the attributes are:
            - "densities": list of densities of the applicator.
            - "filenames": list of filenames of the applicator.
            - "materials": list of materials of the applicator.
            - "points": list of points (x,y,z,x,y,z) describing the first and last dwell positions
            on the applicator in the frame of the applicator.
            - "wRot": list of wRot of the applicator.
            - "x": list of x of the applicator.
            - "xRoti": list of xRot of the applicator i in [1, N].
            - "y": list of y of the applicator.
            - "yRoti": list of yRot of the applicator i in [1, N].
            - "z": list of z of the applicator.
            - "zRoti": list of zRot of the applicator i in [1, N].

        ### Outputs:
            - None := will update the BrachyPlan.applicator_list attribute
        """
        if isinstance(applicator_list_pth, Path) or isinstance(
            applicator_list_pth, str
        ):
            with open(applicator_list_pth, "r") as json_file:
                applicator_list = json.load(json_file)
            num_applicators = len(applicator_list["densities"])

            for i in range(num_applicators):

                j = i + 1 if i >= 1 else ""

                rotation = np.array(
                    [
                        (
                            applicator_list[f"wRot{j}"]
                            if f"wRot{j}" in applicator_list
                            else 0
                        ),
                        (
                            applicator_list[f"xRot{j}"]
                            if f"xRot{j}" in applicator_list
                            else 0
                        ),
                        (
                            applicator_list[f"yRot{j}"]
                            if f"yRot{j}" in applicator_list
                            else 0
                        ),
                        (
                            applicator_list[f"zRot{j}"]
                            if f"zRot{j}" in applicator_list
                            else 0
                        ),
                    ]
                )

                applicator_obj = BrachyApplicator(
                    pth_input_file=applicator_list["filenames"][i],
                    material=applicator_list["materials"][i],
                    density=applicator_list["densities"][i],
                    origin=self.patient_origin,
                    rotation=rotation,
                    rotation_origin=np.array(
                        [
                            applicator_list["x"],
                            applicator_list["y"],
                            applicator_list["z"],
                        ]
                    ),
                    coordinates=np.array(
                        [
                            applicator_list["x"],
                            applicator_list["y"],
                            applicator_list["z"],
                        ]
                    ),
                    # for now RapidBrachy exports only one catheter trajectory.
                    # in future, more catheter trajectories may be possible.
                    # use i instead of 0 to get the ith catheter trajectory.
                    catheter_trajectory=np.array(
                        [
                            applicator_list["points"][0][0:3],
                            applicator_list["points"][0][3:6],
                        ]
                    ),
                )

                self.applicator_list.append(applicator_obj)

def load_applicator_materials(pth_applicator_materials : Path | None = None ) -> dict:
    r"""
    Purpose:
        - To load the applicator material candidates from the constants json file.
    Inputs:
        - pth_applicator_materials:Path := the path to the applicator materials
    Outputs:
        - dict := the dictionary containing the applicator materials.
    """
    if pth_applicator_materials is None:
        pth_applicator_materials = Path(__file__).parent.parent.parent / "admin/constants/applicator_materials.json"
    with open(pth_applicator_materials, "r") as json_file:
        applicator_materials_dict = json.load(json_file)
    return applicator_materials_dict