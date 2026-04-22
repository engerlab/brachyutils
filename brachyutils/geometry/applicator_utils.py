import numpy as np
import os
import json

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
            - None: an applicator object is created dependeing on the inputs.
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
            - None := will update the BrachyApplicator object based on the mac file.
        """
        raise NotImplementedError("to be implemented soon")

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
            - None := will update the applicator mesh from the verticies and faces.
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
            - None := will update the brachy applicator from the applicator mesh.
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
            - None := will update the applicator verticies based on the new origin.
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
            - None := will update the applicator verticies based on the new rotation.
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
            - None := will update the applicator verticies based on the new coordinates.
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
            - None := will save the applicator geometry to a json file.
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
            - None := will save the applicator geometry to a mac file.
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
            - None := will save the applicator geometry to an stl file.
        """
        self._update_applicator_mesh_from_brachy_applicator()
        # write the polydata to an stl file
        stl_writer = vtkSTLWriter()
        stl_writer.SetFileName(pth_output)
        stl_writer.SetInputData(self.applicator_mesh)
        stl_writer.Write()