import gc
import json
import os
import re
import warnings
from copy import deepcopy
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np

# from multipledispatch import dispatch
from scipy import interpolate, ndimage

# from typing import Optional
from tqdm import tqdm
from vtk import (
    vtkCellArray,
    vtkPoints,
    vtkPolyData,
    vtkTransform,
    vtkTransformPolyDataFilter,
)
from vtk.util import numpy_support
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter

from brachyutils.dicom_utils import BrachyDicom
from brachyutils.dose_utils import BrachyDose, dose_with_empty_grid_like
from brachyutils.egsphant_utils import BrachyEgsphant
from brachyutils.simulation_utils import BrachySimulation


class BrachyStructure:
    r"""
    Purpose:
        - this class holds the information regarding a structure inside a brachytherapy
        treatment plan.

    Attributes:

        Basic Attributes
        - name:str
        - mask
        - target_volume

        DVH Attributes:
        - in_dvh
        - dvh_metric_name
        - dvh_metric_clinical_goal
        - dvh_metric_observed
        - normalized_cummulative_dvh

        Uncertainty Attributes:
        - uvh
        - uncertainty_mean
        - uncertainty_std
        - uncertainty_max
        - uncertainty_min

        Optimization Attributes:
        - name_in_gurobiModel
        - bound_coordinates_in_gurobiModel
        - penalty_weight_linear
        - penalty_weight_quadratic
        - penalty_weight_uniformity
        - dose_limit
        - max_dose
        - min_dose

        Simulation attributes:
        - density
        - density_mode
        - material

    Functions:
        - get_dvh_metric(combined_dose:BrachyDose)
        - to_dict(export_format:str)
    """

    def __init__(self):
        self.name: str = None
        self.mask: np.array = None  # shape: (z, y, x)
        self.target_volume: bool = None

        # dose volume histogram
        self.in_dvh: bool = None
        self.dvh_metric_name: str = None
        self.dvh_metric_clinical_goal: float = None
        self.dvh_metric_observed: float = None
        self.normalized_cummulative_dvh: np.array = None

        # uncertainty volume histogram
        self.uvh: np.array = None
        self.uncertainty_mean: float = None
        self.uncertainty_std: float = None
        self.uncertainty_max: float = None
        self.uncertainty_min: float = None

        # optimization attributes
        self.name_in_gurobiModel: str = None
        self.bound_coordinates_in_gurobiModel: list = None
        self.penalty_weight_linear: float = None
        self.penalty_weight_quadratic: float = None
        self.penalty_weight_uniformity: float = None
        self.dose_limit: float = None
        self.max_dose: float = 500
        self.min_dose: float = 0

        # simulation attributes
        self.density: float = None  # 0
        self.density_mode: str = None  # ""
        self.material: str = None  # "CT Material"

    def get_dvh_metric(self, combined_dose: BrachyDose):
        assert self.mask is not None, "mask is not loaded"
        assert self.dvh_metric_name is not None, "dvh metric name is not set"
        assert (
            self.dvh_metric_clinical_goal is not None
        ), "dvh metric clinical goal is not set"

        num_bins = int(combined_dose.grid.max() * 10) + 1
        total_dose_max = combined_dose.grid.max()

        structure_dose = combined_dose.grid * self.mask
        structure_dose = structure_dose[structure_dose != 0].flatten()
        voxel_volume = np.prod(combined_dose.voxel_size)
        num_voxels_in_structure = np.sum(self.mask)

        if "%" in self.dvh_metric_name:
            histogram_limit = float(*re.findall(r"-?\d+\.?\d*", self.dvh_metric_name))
        elif "cc" in self.dvh_metric_name:
            histogram_limit = (
                float(*re.findall(r"-?\d+\.?\d*", self.dvh_metric_name))
                / (voxel_volume * num_voxels_in_structure)
                * 100
            )
        else:
            raise ValueError(
                "invalid name for DVH metric name. \
                The metric should have percent sign (%) or cc."
            )

        self.dvh_metric_observed, self.normalized_cummulative_dvh = dvh_metric(
            structure_dose, num_bins, total_dose_max, histogram_limit, voxel_volume
        )

    def to_dict(self, export_format: str):
        r"""
        Purpose:
            - To export the BrachyStructure object into a dictionary of a certain format.
        Inputs:
            - export_format := the export_format of the exported plan. an example is:
                - "RapidBrachy":{
                    "density": 0,
                    "density_mode": "",
                    "dose_limit": 0,
                    "dvhConstraints": "",
                    "in_dvh": true,
                    "linear_weight": 1,
                    "material": "CT Material",
                    "max_dose": 500,
                    "min_dose": 0,
                    "name": "BODY",
                    "quadratic_weight": 1,
                    "type": "" or "Target volume" or "Organ at risk",
                    "uniformity_weight": 1}

                - "WebApp": Not implemented yet
        """
        if export_format == "WebApp":
            raise NotImplementedError("export to WebApp is not implemented yet")
        elif export_format == "RapidBrachy":
            return {
                "density": self.density,
                "density_mode": self.density_mode,
                "dose_limit": self.dose_limit,
                "dvhConstraints": "",
                "in_dvh": self.in_dvh,
                "linear_weight": self.penalty_weight_linear,
                "material": self.material,
                "max_dose": self.max_dose,
                "min_dose": self.min_dose,
                "name": self.name,
                "quadratic_weight": self.penalty_weight_quadratic,
                "type": "Target volume" if self.target_volume else "Organ at risk",
                "uniformity_weight": self.penalty_weight_uniformity,
            }


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
            - catheter_trajector: (list, optional): The list of start dwell poisition and end dwell position of the catheter inside
            the applicator [[x,y,z,x,y,z]]. Defaults to None.
        Outputs:
            - Void: an applicator object is created dependeing on the inputs.
        """
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
        self.catheter_trajectory: list = []

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
        if type(self) != type(other):
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

    def _update_catheter_trajectory(self,) -> None:
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
            "verticies": self.verticies.tolist(),
            "faces": self.faces.tolist(),
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
        # raise NotImplementedError("to be implemented soon")
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


class BrachyPlan:
    r"""
    Purpose:
        - This class holds the information regarding the brachytherapy treatment plan
        as well as all the functions to support the necessary plan operations.

    Attributes:
        - num_dwells:int := the number of dwell positions in the plan
        - catheter_table:list := a list of catheter dictionaries. each catheter dictionary
        contains the keys "dwells", "id", and points. the value belonging to the "dwells" key
        is a list of dwell position dictionary. The dwell position dictionary contains the keys:
        "angle", "position", "relativePos", "rotation", "time", and "weight". for more info,
        look at the function BrachyPlan.load_catheterTable_json()
        - dwell_numbers:np.array := the dwell number of each dwell position in the plan
        - dwell_times:np.array := the dwell time of each dwell position in the plan
        - dwell_coordinates:list := a list of dictionaries. each dictionary contains the
        keys "position", "rotation", and "relativePos"
        - organ_bounds:dict
        - dose_rate_tensor:np.array := dose rate from dwell position 1 to num_dwells.
        matches the dwell_number_list. shape: (num_dwells, z, y, x)
        - uncertainty_tensor:np.array := uncertainty from dwell position 1 to num_dwells.
        shape: (num_dwells, z, y, x)
        - brachy_structure:list[BrachyStructure] := the list of patient structures in the plan.

    Functions:
        - load_catheterTable_json()
        - _extract_dwell_numbers_times_coordinates_from_catheterTable()
        - _update_catheter_table_from_plan()
        - _update_dose_after_change_in_plan()
        - load_dose_rate_or_uncertainty_tensor()
        - _calculate_combined_dose()
        - set_dvh_metric_goals()
        - create_structures()
        - calculate_dvh_metrics()
        - _calculate_combined_uncertainty()
        - calculate_uncertainty_per_structure()
        - export_brachy_plan ()
    """

    def __init__(
        self,
        # for loading dicom
        dir_dicom: str = None,
        # for structure creation:
        dvh_metric_goals: dict = None,
        pth_structure_source: str = None,
        dose_cropped_by_body: bool = False,
        # for loading catheter table:
        pth_catheter_table_json: str = None,
        # for loading dose or uncertainty:
        dir_dose_rate: str = None,
        type_dose_file: str = ".nrrd",
        load_dose_or_uncertainty: str = "dose",
        multi_processing: bool = False,
        # for simulation setup:
        combined_simulation_dict: dict = None,
        dir_egsphant: str = None,
        # for applicator setup
        pth_applicator_list_json: str = None,
        applicator_format: str = "RapidBrachy",
    ):
        r"""
        Purpose:
            - To initialize the BrachyPlan object.
        Inputs:
            # for loading catheter table:
            - pth_catheter_table_json:str := path to a json file containing the information of the catheter table.
            # for loading dose or uncertainty:
            - dir_dose_rate:str := path to the directory containing the dose rate files for a patient.
            - type_dose_file:str = ".nrrd" := the type of dose file to load (default is ".nrrd").
            - load_dose_or_uncertainty:str = "dose" := specify whether to load "dose" or "uncertainty" or "both" (default is "dose").
            - multi_processing:bool = False := flag to enable multi-processing for loading dose or uncertainty (default is False).
            # for structure creation:
            - dvh_metric_goals:dict = None := dictionary containing the DVH metric goals (default is None).
            - pth_structure_source:str = None := path to the directory containing the structures (default is None).
            - dose_cropped_by_body:bool = True := flag to indicate whether the dose is cropped by body (default is True).
            - combined_simulation_dict = None := dictionary containing the simulation setup,
            - dir_egsphant = None := path to the directory containing the egsphant file,
            - pth_applicator_list_json := path to the json file containing the applicator list. See load_applicator_list() for more info.
            - applicator_format:str = "RapidBrachy" := the format of the applicator list (default is "RapidBrachy"). See load_applicator_list() for more info.
        Outputs:
            - Void := will initialize the BrachyPlan object
        Dependencies:
            -
        """
        # declare the attributes
        # patient origin is used as a reference point for the catheter table,
        # the dwell coordinates, image origin, egsphant, and the dose objects.
        # XXX: figure out how to sort out patient origin to match all above.
        self.patient_origin = np.array([0, 0, 0])  # x,y,z
        # catheter table attributes
        self.catheter_table = None
        self.num_catheters = None
        self.catheter_numbers = np.array([], dtype=int)  # shape: (num_catheters, 1)
        self.num_dwells = None
        self.dwell_numbers = np.array([], dtype=int)  # shape: (num_dwells, 1)
        self.dwell_times = np.array([], dtype=np.float32)  # shape: (num_dwells, 1)
        self.dwell_coordinates = []  # shape: (num_dwells, 3)

        # dose attributes
        self.dose_rate_tensor = np.array(
            [], dtype=np.float32
        )  # shape: (num_dwells, z, y, x)
        self.combined_dose: BrachyDose = None
        self.uncertainty_tensor = np.array(
            [], dtype=np.float32
        )  # shape: (num_dwells, z, y, x)

        # sturctures attributes
        # self.organ_bounds = None
        self.dvh_metric_goals: dict = None
        self.dvh_metric_observed: dict = None
        self.structure_list: List[BrachyStructure] = []

        # dicom image
        self.dicom_obj: BrachyDicom = None

        # simulation attributes
        self.simulation_setup: BrachySimulation = None
        self.egsphant: BrachyEgsphant = None
        self.applicator_list: List[BrachyApplicator] = []
        # XXX: figure out if the two below are dwell or applicator attributes?
        self.applicator_rotation_axis: np.array = np.array([0, 0, 1])  # x,y,z
        self.applicator_rotation_origin: float = np.array([0, 0, 0])  # x,y,z

        # optimization attributes
        self.optimizer = None

        # fill the attributes depending on the inputs to the constructor
        # set the dvh metric goals if provided
        (
            self.set_dvh_metric_goals(dvh_metric_goals)
            if dvh_metric_goals is not None
            else None
        )

        # load the dicom plan if the path is provided
        if dir_dicom is not None:
            self.load_brachy_plan_from_dicom(dir_dicom, dose_cropped_by_body)

        # load the catheter table if the path is provided
        if pth_catheter_table_json is not None:
            self.load_catheterTable_json(pth_catheter_table_json)

        # load the dose rate tensor if the path is provided
        if dir_dose_rate is not None:
            self.load_dose_rate_or_uncertainty_tensor(
                dir_dose_rate,
                type_dose_file=type_dose_file,
                load_dose_or_uncertainty=load_dose_or_uncertainty,
                multi_processing=multi_processing,
            )

        # create the structures if the path is provided
        if pth_structure_source is not None:
            self.create_structures(
                dir_structures_source=pth_structure_source,
                dose_cropped_by_body=dose_cropped_by_body,
            )

        # load the simulation setup if the dictionary is provided
        if dir_egsphant is not None:
            self.egsphant = BrachyEgsphant(dir_egsphant)
        if combined_simulation_dict is not None:
            self.combined_simulation_setup = BrachySimulation(combined_simulation_dict)

        # load the applicator list if the path is provided
        if pth_applicator_list_json is not None:
            self.load_applicator_list(pth_applicator_list_json, applicator_format)

    def load_brachy_plan_from_dicom(
        self, dir_dicom: str, dose_cropped_by_body: bool = False
    ):
        r"""
        Purpose:
            - To load the brachytherapy plan from a directory containing the dicom files.
            depending on the availability of the RP, RS and RD dicom files, the plan will be
            loaded in different ways.
        Inputs:
            - dir_dicom := path to the directory containing the dicom files.
        Outputs:
            - Void := will update the BrachyPlan.dicom_obj as well other attribute
        Dependencies:
            - BrachyDicom
        """
        file_list_dcm = glob(os.path.join(dir_dicom, "*.dcm"))
        file_list_dcm = [os.path.basename(file).split(".")[0] for file in file_list_dcm]
        all_names = ",".join(file_list_dcm)

        load_structure = True if "RS" in all_names else False
        load_plan = True if "RP" in all_names else False
        load_dose = True if "RD" in all_names else False

        try:
            self.dicom_obj = BrachyDicom(
                pth_dir_dicom=dir_dicom,
                load_structure=load_structure,
                load_plan=load_plan,
                load_dose=load_dose,
            )
        except Exception as e:
            print(f"Error in loading all dicom files: {e}")
            try:
                self.dicom_obj = BrachyDicom(
                    pth_dir_dicom=dir_dicom,
                    load_structure=load_structure,
                    load_plan=load_plan,
                    load_dose=False,
                )
            except Exception as e:
                print(f"Error in loading dicom dose file: {e}")
                try:
                    self.dicom_obj = BrachyDicom(
                        pth_dir_dicom=dir_dicom,
                        load_structure=load_structure,
                        load_plan=False,
                        load_dose=False,
                    )
                except Exception as e:
                    print(f"Error in loading dicom plan file: {e}")
                    self.dicom_obj = BrachyDicom(
                        pth_dir_dicom=dir_dicom,
                        load_structure=False,
                        load_plan=False,
                        load_dose=False,
                    )

        if load_structure:
            self.create_structures(
                structure_mask_dict=self.dicom_obj.structure_mask_dict,
                dose_cropped_by_body=dose_cropped_by_body,
            )

        if load_plan:
            self.catheter_table = self.dicom_obj.catheter_table
            self._extract_dwell_numbers_times_coordinates_from_catheterTable()

        if load_dose:
            self.combined_dose = self.dicom_obj.dose

    def load_catheterTable_json(self, pth_catheter_table_json: str):
        r"""
        Purpose:
            - To load the contents of a catheter table into the Brachy plan.
        Inputs:
            - pth_catheter_table_json := path to a json file having the info on the catheter table.
            here is the expected contents of the catheter table json:
            [
                {
                    "dwells":[
                        "angle":= angle of the IMBT shield
                        "position":{ := dwell position in the patient coordinate system
                            "x",
                            "y",
                            "z"
                        },
                        "relativePos":= dwell coordinate along the catheter from the reference point. increments of 5 mm
                        "rotation": { := rotation of the dwell position in the patient coordinate system
                            "x",
                            "y",
                            "z"
                        },
                        "time" := dwell time for this dwell position
                        "weight" := ratio of this dwell time over the sum of all dwell times in all catheters.
                        ...,
                    ],
                    "id":= the id of the caheter,
                    "points":[] := i do not know what this is. in all plans i have seen, it has been lefty empty
                }
            ]
        Outputs:
            - Void := will update the BrachyPlan.catheter_table attribute
        Dependencies:
            - json
        """
        # reset catheter table in case of a re-read
        self.catheter_table = None
        # load the json file
        with open(pth_catheter_table_json, "r") as json_file:
            catheter_table = json.load(json_file)

        self.catheter_table = catheter_table
        self._extract_dwell_numbers_times_coordinates_from_catheterTable()

    def _extract_dwell_numbers_times_coordinates_from_catheterTable(self):
        r"""
        Purpose:
            - To extract the dwell numbers, times, and coordinates from the catheter table
            and save them as class attributes.
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will update the BrachyPlan.dwell_numbers, BrachyPlan.dwell_times,
            and BrachyPlan.dwell_coordinates attributes
        """
        assert self.catheter_table is not None, "catheter table is not loaded"
        # reset the dwell_numbers, dwell times, coordinates, and num dwells
        (
            self.catheter_numbers,
            self.dwell_numbers,
            self.dwell_times,
            self.dwell_coordinates,
        ) = (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=np.float32),
            [],
        )
        self.num_catheters = None
        self.num_dwells = None

        # extract the attributes above from the catheter table
        dwell_counter = 1
        for catheter in self.catheter_table:
            self.catheter_numbers = np.append(self.catheter_numbers, catheter["id"])
            for dwell in catheter["dwells"]:
                self.dwell_numbers = np.append(self.dwell_numbers, dwell_counter)
                self.dwell_times = np.append(self.dwell_times, dwell["time"])
                self.dwell_coordinates.append(
                    {
                        "angle": dwell["angle"],
                        "position": dwell["position"],
                        "rotation": dwell["rotation"],
                        "relativePos": dwell["relativePos"],
                        "catheterId": catheter["id"],
                    }
                )
                dwell_counter += 1

        assert (
            len(self.catheter_numbers) - 1 == self.catheter_numbers[-1]
        ), "catheter numbers are not extracted correctly"
        self.num_catheters = len(self.catheter_numbers)

        assert (
            len(self.dwell_numbers) == self.dwell_numbers[-1]
        ), "dwell numbers are not extracted correctly"
        self.num_dwells = len(self.dwell_numbers)

    def _update_catheter_table_from_plan(self):
        r"""
        Purpose:
            - Assuming that the dwell times or coordinates have changed, we need to update
            the catheter_table attribute to match the plan.
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will update the BrachyPlan.catheter_table attribute
        """
        assert self.dwell_numbers.size != 0, "dwell numbers are not extracted"
        assert self.dwell_times.size != 0, "dwell times are not extracted"
        assert len(self.dwell_coordinates) != 0, "dwell coordinates are not extracted"
        assert self.num_dwells is not None, "number of dwells is not extracted"

        self.catheter_table = []

        for catheter_i in self.catheter_numbers:
            catheter = {}
            catheter["id"] = int(catheter_i)
            catheter["points"] = []
            catheter["dwells"] = []
            dwell = {}
            for dwell_i in self.dwell_numbers:
                if self.dwell_coordinates[dwell_i - 1]["catheterId"] != catheter_i:
                    continue
                dwell["angle"] = float(self.dwell_coordinates[dwell_i - 1]["angle"])
                dwell["position"] = list(
                    self.dwell_coordinates[dwell_i - 1]["position"].astype(np.float64)
                )
                dwell["relativePos"] = float(
                    self.dwell_coordinates[dwell_i - 1]["relativePos"]
                )
                dwell["rotation"] = list(
                    self.dwell_coordinates[dwell_i - 1]["rotation"].astype(np.float64)
                )
                dwell["time"] = float(self.dwell_times[dwell_i - 1].item())
                dwell["weight"] = float(
                    (self.dwell_times[dwell_i - 1] / np.sum(self.dwell_times)).item()
                )
                catheter["dwells"].append(deepcopy(dwell))

            self.catheter_table.append(deepcopy(catheter))

    def _update_dose_after_change_in_plan(self):
        r"""
        Purpose:
            - Assuming that the dwell times or coordinates have changed, we need to update
            the catheter_table attribute and the combined dose to match the plan.
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will update the BrachyPlan.catheter_table and BrachyPlan.combined_dose
            attributes
        """
        self._update_catheter_table_from_plan()
        self._calculate_combined_dose()

    def load_dose_rate_or_uncertainty_tensor(
        self,
        dir_dose_rate: str,
        type_dose_file: str = ".nrrd",
        load_dose_or_uncertainty: str = "dose",
        multi_processing: bool = False,
    ):
        r"""
        Purpose:
            - To load the dose rate tensor into the BrachyPlan object given a folder with
            patient's dose rate files and the catheter table loaded into the BrachyPlan object.
            In addition, combined dose is calculated as a linear combination of the dose rates
            and dwell times.
        Inputs:
            - dir_dose_rate :=  path to the directory containing the dose rate files. we assume
            that the name of the dose rate files end as "run_1.nrrd", "run_2.nrrd", etc.
            - type_dose_file := the type of dose rate file. The type could be ".nrrd" or ".3ddose"
            consult BrachyDose in dose_utils.py for more info on the dose rate file types.
            - load_dose_or_uncertainty := either "dose", "uncertainty", or "both"
            - multi_processing := if True, the dose rate files will be loaded in parallel. By default,
            we use 8 cores for parallel processing.
        Outputs:
            - Void := will update the BrachyPlan.dose_rate_tensor attribute
        Dependencies:
            - glob
            - BrachyDose
        """
        # make sure catheter table is loaded
        assert self.catheter_table is not None, "catheter table is not loaded"
        assert self.dwell_numbers.size != 0, "dwell numbers are not extracted"
        assert self.dwell_times.size != 0, "dwell times are not extracted"
        assert len(self.dwell_coordinates) != 0, "dwell coordinates are not extracted"
        assert self.num_dwells is not None, "number of dwells is not extracted"

        # here is the list of the dose rate files
        dose_rate_files = glob(os.path.join(dir_dose_rate, f"*{type_dose_file}"))

        dose_rate_files = [
            dosefile for dosefile in dose_rate_files if "combined" not in dosefile
        ]

        dose_rate_files.sort(
            key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1])
        )
        assert (
            len(dose_rate_files) == self.num_dwells
        ), "number of dose rate files does not match the number of dwell positions"

        test_dose_obj = BrachyDose(dose_rate_files[0])

        if load_dose_or_uncertainty not in ["dose", "uncertainty", "both"]:
            raise ValueError(
                "load_dose_or_uncertainty should be either 'dose', 'uncertainty', or 'both'"
            )

        # load the dose rate tensor
        if multi_processing:
            with Pool(8) as mp_pool:
                dose_or_uncertainty_list = np.array(
                    mp_pool.map(
                        partial(
                            _load_single_dose_or_uncertainty_to_dict,
                            load_dose_or_uncertainty=load_dose_or_uncertainty,
                        ),
                        dose_rate_files,
                    ),
                    dtype=np.float32,
                )

        else:
            dose_or_uncertainty_list = np.empty(len(dose_rate_files), dtype=object)
            for i, pth_dose_rate in tqdm(enumerate(dose_rate_files)):
                dose_or_uncertainty_list[i] = _load_single_dose_or_uncertainty_to_dict(
                    pth_dose_rate, load_dose_or_uncertainty
                )
            print(dose_or_uncertainty_list.shape)

        if load_dose_or_uncertainty == "both":
            self.dose_rate_tensor = np.array(
                dose_or_uncertainty_list[:, 0], dtype=np.float32
            )
            self.uncertainty_tensor = np.array(
                dose_or_uncertainty_list[:, 1], dtype=np.float32
            )
        elif load_dose_or_uncertainty == "dose":
            self.dose_rate_tensor = np.array(dose_or_uncertainty_list, dtype=np.float32)
        elif load_dose_or_uncertainty == "uncertainty":
            self.uncertainty_tensor = np.array(
                dose_or_uncertainty_list, dtype=np.float32
            )
        else:
            raise ValueError(
                "load_dose_or_uncertainty should be either 'dose', 'uncertainty', or 'both'"
            )

        del dose_or_uncertainty_list
        gc.collect()

        self.combined_dose = dose_with_empty_grid_like(test_dose_obj)

        if load_dose_or_uncertainty != "uncertainty":
            self._calculate_combined_dose()
        if load_dose_or_uncertainty != "dose":
            self._calculate_combined_uncertainty()

        if len(self.structure_list) != 0:
            for structure in self.structure_list:
                structure.mask = _resize_structure_mask(
                    structure.mask, self.combined_dose.grid.shape
                )

    def _calculate_combined_dose(self):
        """
        Purpose:
        - To calculate the combined dose by multiplying the dose rate tensor with the dwell times array.
        The result is stored in the combined_dose attribute.

        Raises:
            AssertionError: If the dose rate tensor or dwell times array is empty.
        """
        assert (
            self.dose_rate_tensor.size != 0
        ), "dose rate tensor is empty. Run load_dose_rate_or_uncertainty_tensor()"
        assert (
            self.dwell_times.size != 0
        ), "dwell times array is empty. Run _extract_dwell_numbers_times_coordinates_from_catheterTable()"

        # calculate the combined dose and store the result in the combined_dose attribute
        # this implementation is a little slow, and very very memory efficient
        for i in range(self.num_dwells):
            self.combined_dose.grid += self.dose_rate_tensor[i] * self.dwell_times[i]
            # this implementation is a bit faster, but very memory inefficient
            # self.combined_dose.grid = np.sum(
            #     self.dose_rate_tensor * self.dwell_times[:, np.newaxis, np.newaxis, np.newaxis],
            #     axis=0)

    def set_dvh_metric_goals(self, dvh_metric_goals: dict):
        r"""
        Purpose:
            - To set the dvh metric list of the BrachyPlan object.
        Inputs:
            - dvh_metric_goals := a list of dictionaries. each dictionary contains the keys:
            "structure_name", "clinical_goal", "observed_value", and "penalty_weight"
        Outputs:
            - Void := will update the BrachyPlan.dvh_metric_goals attribute
        """
        for dvh_metric in dvh_metric_goals:
            assert (
                "D" in dvh_metric
            ), "dvh metric name should start with D as we are only supporting dose metrics for now"
            assert (
                "cc" in dvh_metric or "%" in dvh_metric
            ), "dvh metric name should end with cc or '%' to signify the absolute or relative volume"
            assert (
                dvh_metric_goals[dvh_metric] is not None
            ), "for each dvh metric, the clinical threshold should be provided in Gy."

        self.dvh_metric_goals = dvh_metric_goals

    def create_structures(
        self,
        structure_mask_dict: dict = None,
        dir_structures_source: str = None,
        dose_cropped_by_body: bool = False,
    ):
        r"""
        Purpose:
            - To create a list of BrachyStructure objects given the path to the directory
            containing the structure masks. the list is stored in the BrachyPlan.structure_list attribute.
            Eeach BrachyStructure object will have attributes for the structure mask, the dose volume
            and uncertainty volume histograms, optimization attributes, and simulation attributes.

            The basic (mandatory) attributes are the structure name, mask and whether it is a target volume or not.
            If dvh metric goals are set, the BrachyStructure object will automatically update the DVH attributes
            in the BrachyStructure object.

        Inputes:
            - structure_mask_dict:dict := a dictionary with the structure name as key and the mask as value. This
            dictionary can be obtained from self.dicom_obj.structure_mask_dict.

            - dir_structures_source := path to the directory containing the structure masks.
            this could be dicom file (starting with RS) or nrrd files. If self.dicom_obj is not None,
            using this parameter will over-ride the previous structure objects.

        Outputs:
            - Void := will update the BrachyPlan.structure_list attribute
        Dependencies:
            - BrachyDicom
        """
        # contour names are assigned based on the keys in the structure mask dictionary
        if structure_mask_dict is None:
            # assert dir_structures_source is not None, "dir_structures_source is not provided"
            try:
                structure_mask_dict = _load_structure_mask(dir_structures_source)
            except Exception:
                raise ValueError(
                    "Either structure mask should be provided or dir_structure_source"
                ) from None

        # get the key corresponding to the body contour, which is used to squeeze the structure mask
        body_key = list(
            filter(lambda x: "body" in x.lower(), structure_mask_dict.keys())
        )[0]

        for structure_name in structure_mask_dict.keys():
            structure_obj = BrachyStructure()
            # get the name based on the structure mask dictionary key
            structure_obj.name = structure_name

            # get the mask from the structure mask dictionary
            structure_obj.mask = structure_mask_dict[structure_name]
            if dose_cropped_by_body:
                # obtain the range of body contour on each axis.
                # we assume that the body contour contains the word "body" in its name
                body_index_range = np.zeros([3, 2], dtype=int)
                for i in range(3):
                    body_index_range[i, :] = np.floor(
                        np.array(
                            [
                                np.argwhere(structure_mask_dict[body_key] == 1)[
                                    :, i
                                ].min(),
                                # off set of +1 is added to acount for python stopping before range end
                                np.argwhere(structure_mask_dict[body_key] == 1)[
                                    :, i
                                ].max()
                                + 1,
                            ]
                        )
                    ).astype(int)
                # apply body contour mask to the structure mask
                structure_obj.mask = structure_obj.mask[
                    body_index_range[0][0] : body_index_range[0][1],
                    body_index_range[1][0] : body_index_range[1][1],
                    body_index_range[2][0] : body_index_range[2][1],
                ]
            # resize the mask to match the dose grid if dose grid exists
            structure_obj.mask = (
                _resize_structure_mask(
                    structure_obj.mask, self.combined_dose.grid.shape
                )
                if self.combined_dose is not None
                else structure_obj.mask
            )

            # get the dvh metric goals if they are set
            if self.dvh_metric_goals is not None:
                try:
                    dvh_metric = list(
                        filter(
                            lambda x: x.split("(")[-1].split(")")[0].lower()
                            in structure_obj.name,
                            self.dvh_metric_goals.keys(),
                        )
                    )[0]
                except IndexError as e:
                    print(f"{structure_obj.name} is not in the dvh metric goals. {e}")
                    structure_obj.in_dvh = False
                    continue

                structure_obj.in_dvh = True
                structure_obj.dvh_metric_name = dvh_metric.split("(")[0]
                structure_obj.dvh_metric_clinical_goal = self.dvh_metric_goals[
                    dvh_metric
                ]

            # get the simulation parameters for that structure
            if self.simulation_setup is not None:
                raise NotImplementedError("to be implemented soon")

            if self.optimizer is not None:
                raise NotImplementedError("to be implemented soon")

            # add the structure object to the structure list
            self.structure_list.append(structure_obj)

    def load_applicator_list(
        self,
        pth_applicator_list_json: str,
        format: str = "WebApp",
    ):
        r"""
        Purpose:
            - To load the applicator list from a json file containing the applicator geometry.
        Inputs:
            - pth_applicator_list_json:str := path to the json file containing the applicator list with N applicators.
            The items inside this list have the attributes bellow. If any left empty, the default value will be used.
            these attributes could be changed later using the setter functions.

            if the format is WebApp, the attributes are:
                - "path": path to the applicator geometry file (.stl or .json).
                - "material": material of the applicator (str).
                - "density": density of the applicator (str).
                - "origin": origin of the applicator ([x,y,z]).
                - "rotation": rotation of the applicator ([w,x,y,z]).
                - "rotation_origin": origin of the rotation ([x,y,z]).
                - "coordinates": coordinates of the applicator ([x,y,z]).

            if the format is RapidBrachy, the attributes are:
                - "densities": list of densities of the applicator.
                - "filenames": list of filenames of the applicator.
                - "materials": list of materials of the applicator.
                - "points": list of points (x,y,z,x,y,z) describing the first and last dwell positions
                on the applicator in the frame of the applicator.
                - "shieldNormalx": normal of applicator in the x direction in the frame of CT.
                - "shieldNormaly": normal of applicator in the y direction in the frame of CT.
                - "shieldNormalz": normal of applicator in the z direction in the frame of CT.
                - "wRot": list of wRot of the applicator.
                - "x": list of x of the applicator.
                - "xRoti": list of xRot of the applicator i in [1, N].
                - "y": list of y of the applicator.
                - "yRoti": list of yRot of the applicator i in [1, N].
                - "z": list of z of the applicator.
                - "zRoti": list of zRot of the applicator i in [1, N].

            - format:str := the format of the applicator geometry file. options are "RapidBrachy" or "WebApp"
        Outputs:
            - Void := will update the BrachyPlan.applicator_list attribute
        """
        with open(pth_applicator_list_json, "r") as json_file:
            applicator_list = json.load(json_file)
        if format == "RapidBrachy":
            num_applicators = len(applicator_list["densities"])

            for i in range(num_applicators):

                j = i if i > 1 else ""
                shieldNormal = np.array(
                    [
                        (
                            applicator_list["shieldNormalx"]
                            if "shieldNormalx" in applicator_list
                            else 0
                        ),
                        (
                            applicator_list["shieldNormaly"]
                            if "shieldNormaly" in applicator_list
                            else 0
                        ),
                        (
                            applicator_list["shieldNormalz"]
                            if "shieldNormalz" in applicator_list
                            else 0
                        ),
                    ]
                )

                applicator_obj = BrachyApplicator(
                    pth_input_file=applicator_list["filenames"][i],
                    material=applicator_list["materials"][i],
                    density=applicator_list["densities"][i],
                    origin=self.patient_origin,
                    rotation=np.array(
                        [
                            applicator_list[f"wRot{j}"],
                            applicator_list[f"xRot{j}"],
                            applicator_list[f"yRot{j}"],
                            applicator_list[f"zRot{j}"],
                        ]
                    ),
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
                    normal=shieldNormal,
                    catheter_trajectory=applicator_list["points"],
                )

                self.applicator_list.append(applicator_obj)

        elif format == "WebApp":
            for applicator in applicator_list:

                applicator_obj = BrachyApplicator(
                    pth_input_file=applicator["path"] if "path" in applicator else None,
                    material=(
                        applicator["material"] if "material" in applicator else None
                    ),
                    density=applicator["density"] if "density" in applicator else None,
                    origin=applicator["origin"] if "origin" in applicator else None,
                    rotation=(
                        applicator["rotation"] if "rotation" in applicator else None
                    ),
                    rotation_origin=(
                        applicator["rotation_origin"]
                        if "rotation_origin" in applicator
                        else None
                    ),
                    coordinates=(
                        applicator["coordinates"]
                        if "coordinates" in applicator
                        else None
                    ),
                )
                self.applicator_list.append(applicator_obj)
        else:
            raise ValueError("format should be either 'RapidBrachy' or 'WebApp'")

    def _calculate_combined_uncertainty(self):
        r"""
        Purpose:
            - To calculate the combined uncertainty of the combined dose map.
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will update the BrachyPlan.combined_dose.uncertainty attribute
        """
        assert self.uncertainty_tensor is not None, "uncertainty tensor is not loaded"
        assert self.dwell_times is not None, "dwell times are not extracted"
        assert self.combined_dose is not None, "combined dose is not calculated yet"

        normalized_times = self.dwell_times / np.sum(self.dwell_times)

        # This implementation is a little slow, and very very memory efficient
        self.combined_dose.uncertainty = np.zeros_like(self.combined_dose.grid)
        for i in range(self.num_dwells):
            self.combined_dose.uncertainty += (
                self.uncertainty_tensor[i] * normalized_times[i]
            ) ** 2
        self.combined_dose.uncertainty = np.sqrt(self.combined_dose.uncertainty)

        # This implementation is a bit faster, but very memory inefficient
        # self.combined_dose.uncertainty = np.sqrt(
        #     np.sum(
        #         (self.uncertainty_tensor * normalized_times[:, np.newaxis, np.newaxis, np.newaxis])**2,
        #         axis=0))

    def calculate_dvh_metrics(self):
        r"""
        Purpose:
            - To get the observed value of the dvh metric for each structure in the BrachyPlan.
            the observed value is calculated from the combined dose map.
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will update the BrachyStructure.dvh_metric_observed attribute
        """
        assert self.structure_list is not None, "structure list is not created yet"
        self.dvh_metric_observed = {}
        for structure_obj in self.structure_list:
            structure_obj.get_dvh_metric(self.combined_dose)
            self.dvh_metric_observed[structure_obj.dvh_metric_name] = (
                structure_obj.dvh_metric_observed
            )

        return self.dvh_metric_observed

    def calculate_uncertainty_per_structure(self):
        r"""
        Purpose:
            - To calculate the uncertainty of each structure in the BrachyPlan.
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will update the BrachyStructure.uncertainty attribute
        """
        assert (
            self.combined_dose.uncertainty is not None
        ), "combined uncertainty is not calculated yet"
        assert self.structure_list is not None, "structure list is not created yet"
        for structure_obj in self.structure_list:
            # Apply structure mask to the uncertainty map
            masked_uncertainty = self.combined_dose.uncertainty * structure_obj.mask
            # isolate the uncertainty values that are in the mask
            flattened_uncertainty = masked_uncertainty[
                structure_obj.mask != 0
            ].flatten()
            # generate a histogram from the masked uncertainty
            histogram, bins_edges = np.histogram(
                flattened_uncertainty,
                bins=100,
                range=(0, flattened_uncertainty.max() + 0.1),
            )
            structure_obj.uvh = histogram * np.prod(self.combined_dose.voxel_size)
            structure_obj.uncertainty_mean = np.mean(flattened_uncertainty)
            structure_obj.uncertainty_std = np.std(flattened_uncertainty)
            structure_obj.uncertainty_max = np.max(flattened_uncertainty)
            structure_obj.uncertainty_min = np.min(flattened_uncertainty)

    def export_brachy_plan(
        self, export_format: str, dir_export: str, content_to_export: dict
    ):
        r"""
        Purpose:
            - To export the treatment plan file into a given export_format.
            The export_format can be either "RapidBrachy" or "WebApp".

        Inputs:
            - export_format := the export_format of the exported plan. options are:

                - "RapidBrachy":
                    - "run_#.3ddose" or "run_#.minidos" or "run_#.nrrd",
                    - "catheter_table.json"
                    - "dwell_#.plan",
                    - "run_#.mac",
                    - "ct.egsphant",
                    - "ApplicatorMaterials"
                    - "applicator_geometry.json",
                    - "structure_set.json"

                - "WebApp": Not implemented yet
                    - "run_#.nrrd",
                    - "dwell_#.json",
                    - "run_#.json",

            - dir_export := the directory to which the plan will be exported.
            - content_to_export := a dictionary with which the user specifies what parts
            of the plan to export. The keys are plan components, and the values are binary
            (True or False) except for "dose type", which can be either ".3ddose", ".minidos",
            or ".nrrd". The keys are:

                - "dose":bool,
                - "dose type":str := "nrrd", "minidos" or "3ddose",
                - "uncertainty", "dose rate maps",
                - "catheter_table", "plan", "mac", "egsphant",
                - "ApplicatorMaterials", applicator_geometry", "structure_set",

        Outputs:
            - Void := will export the available parts of a plan into the specified export_format.
        """
        assert os.path.exists(
            dir_export
        ), "export directory does not exist. please make the directory first"

        if export_format == "WebApp":

            raise NotImplementedError("export to WebApp is not implemented yet")

        elif export_format == "RapidBrachy":

            if content_to_export["dose"]:
                self._export_dose(
                    dir_export=dir_export,
                    with_uncertainty=content_to_export["uncertainty"],
                    dose_type=content_to_export["dose_type"],
                    dose_rate_maps=content_to_export["dose_rate_maps"],
                )
                print("Dose exported successfully")
            if content_to_export["catheter_table"]:
                # assumes file name is "catheter_table.json"
                self._export_catheter_table(dir_export)
                print("Catheter Table exported successfully")

            if content_to_export["plan"]:
                # assumes file name is "dwell_#.plan"
                self._export_plan_file(dir_export)
                print(".plan files were exported successfully")

            if content_to_export["mac"]:
                # assumes file name is "run_#.mac"
                self._export_mac_file(dir_export)
                print(".mac files were exported successfully")

            if content_to_export["egsphant"]:
                # assumes file name is "ct.egsphant"
                self._export_egsphant(dir_export)
                print("Egsphant file was exported successfully")

            if content_to_export["applicator_geometry"]:
                # assumes file name is "applicator_geometry.json"
                self._export_applicator_geometry(dir_export, export_format)
                print("applicator geometry file was exported successfully")

            if content_to_export["structure_set"]:
                # assumes file name is "structure_set.json"
                self._export_structure_set(dir_export)
                print("structure set file was exported successfully")

        else:
            raise ValueError("export_format should be either 'RapidBrachy' or 'WebApp'")

    def _export_dose(
        self,
        dir_export: str,
        with_uncertainty=False,
        dose_type=".minidos",
        dose_rate_maps=False,
    ):
        r"""
        Purpose:
            to export combined dose map with or without uncertainty in the provided export directory.
            exporting dose rate maps is optional.
        Inputs:
            - dir_export := the directory to which the dose map will be exported.
            - uncertainty := if True, the uncertainty map will be exported as well.
            - dose_type := the type of dose map to be exported. options are ".3ddose", ".minidos", or ".nrrd".
            - dose_rate_maps := if True, the dose rate maps will be exported as well.
        Outputs:
            - Void := will export the dose map into the specified export directory.
        Dependencies:
            - _export_single_dose_rate()
            - multiprocessing
        """
        assert self.combined_dose is not None, "combined dose is not calculated yet"
        # if uncertainty:
        self.combined_dose.write_brachydose_to_file(
            dir_export + "/combined" + dose_type
        )

        if dose_rate_maps:
            if cpu_count() < 4:
                for i in self.dwell_numbers:
                    _export_single_dose_rate(
                        self.dose_rate_tensor[i - 1],
                        i,
                        self.combined_dose,
                        dir_export,
                        dose_type,
                        self.uncertainty_tensor[i - 1],
                    )
            else:
                # prepare inputs to the parallel processing
                if with_uncertainty and self.uncertainty_tensor is not None:
                    print("Exporting dose rate maps with uncertainty")
                    giant_export_list = [
                        (dose_grid, dwell_number, uncertainty)
                        for dose_grid, dwell_number, uncertainty in zip(
                            self.dose_rate_tensor,
                            self.dwell_numbers,
                            self.uncertainty_tensor,
                        )
                    ]
                else:
                    print("Exporting dose rate maps without uncertainty")
                    giant_export_list = [
                        (dose_grid, dwell_number)
                        for dose_grid, dwell_number in zip(
                            self.dose_rate_tensor, self.dwell_numbers
                        )
                    ]
                with Pool(cpu_count() - 2) as mp_pool:
                    mp_pool.starmap(
                        partial(
                            _export_single_dose_rate,
                            doseObj_template=self.combined_dose,
                            dir_export=dir_export,
                            dose_type=dose_type,
                        ),
                        giant_export_list,
                    )

    def _export_catheter_table(self, dir_export: str):
        r"""
        Purpose:
            - to export catheter table of the plan into a file called catheter_table.json
            inside dir_export.
        Inputs:
            - dir_export := path to the directory where the export happens
        Outputs:
            - void := self.catheter_table is written to catheter_table.json
        Dependencies:
            - json
        """
        file_path = dir_export + "/catheter_table.json"
        with open(file_path, "w") as file:
            json.dump(self.catheter_table, file, indent=4)

    def _export_plan_file(self, dir_export: str):
        r"""
        Purpose:
            - To export dwell positions and their normalized times into ".plan" text files in the
            format required by RapidBrachy.
        Inputs:
            - dir_export := path to the directory where the export happens
        Outputs:
            - void := Two types of .plan files are written, one named combined.plan and the other
            named run_{dwellNumber}.plan. combined.plan contains info of all dwell positions and
            their normalized dwell time, and the run_{dwellNumber}.plan contains info of a single
            dwell position. The format of each .plan file is given in this example:
                "Treatment Plan
                56 Control Points
                Control Point
                weight = 0.00327228
                1 Dwell Position
                -10.2819,82.598,-1224.98,-0.0291444,-0.017922,0.999415,0,0,0,1,0,0,0
                Control Point ..."
        Dependencies:
            - None
        """
        total_dwell_time = np.sum(self.dwell_times)
        combined_plan = "Treatment Plan\n"
        combined_plan += f"{self.num_dwells} Control Points\n"

        for dwell_i in range(self.num_dwells):

            dwell_coordinates_str = np.array(
                list(self.dwell_coordinates[dwell_i]["position"].values())
                + list(self.dwell_coordinates[dwell_i]["rotation"].values())
                + [self.dwell_coordinates[dwell_i]["angle"]]
                + list(self.applicator_rotation_axis)
                + list(self.applicator_rotation_origin),
                dtype=np.float32,
            )
            dwell_coordinates_str = (
                ",".join(
                    [
                        str(int(coord)) if coord == int(coord) else format(coord, ".6f")
                        for coord in dwell_coordinates_str
                    ]
                )
                + "\n"
            )

            combined_plan += "Control Point\n"
            combined_plan += f"weight = {self.dwell_times[dwell_i]/total_dwell_time}\n"
            combined_plan += "1 Dwell Position\n"
            combined_plan += dwell_coordinates_str

            run_i_plan = "Treatment Plan\n"
            run_i_plan += "1 Control Points\n"
            run_i_plan += "Control Point\nweight = 1.0\n"
            run_i_plan += "1 Dwell Position\n"
            run_i_plan += dwell_coordinates_str
            with open(dir_export + f"/run_{dwell_i + 1}.plan", "w") as file:
                file.write(run_i_plan)

        with open(dir_export + "/combined.plan", "w") as file:
            file.write(combined_plan)

    def _export_mac_file(self, dir_export: str):
        r"""
        Purpose:
            - To export the simulation parameters of the plan into a macro files
            called combine.mac and run_{dwellNumber}.mac
        Inputs:
            - dir_export := path to the directory where the export happens
        Outputs:
            - void := Two types of .mac files are written, one named combined.mac and the other
            named run_{dwellNumber}.mac. combined.plan contains

            plan contains info of a single dwell position.

            The format of each .plan file is given in this example:
                /source_world/treatmentType HDR
                /source_world/switch MicroSelectronV2
                /source_world/coreMaterial G4_Ir
                /source_world/core/A 192
                /source_world/core/Z 77
                /sim/plan combined.plan
                /world/phantom ct.egsphant
                /parallel_world/ak_per_history 1.149000e-11
                /parallel_world/ref_ak 4.278729e+04
                /parallel_world/H 2.500000e+00
                /parallel_world/total_time 4.531841e+02
                /dose/format 3ddose
                /run/numberOfThreads 40
                /run/initialize
                /control/verbose 0
                /run/verbose 0
                /tracking/verbose 0
                /run/printProgress 1000000
                /sim/beamOn 10000000

        Dependencies:
            - simulation_utils
        """
        for dwell_i in range(self.num_dwells):
            sim_obj = deepcopy(self.combined_simulation_setup)
            sim_obj.pth_plan = f"dwell_{dwell_i + 1}.plan"
            sim_obj.total_time = 1
            with open(dir_export + f"/run_{dwell_i + 1}.mac", "w") as file:
                file.write(sim_obj.to_string())

        self.combined_simulation_setup.total_time = np.sum(self.dwell_times)
        with open(dir_export + "/combined.mac", "w") as file:
            file.write(self.combined_simulation_setup.to_string())

    def _export_egsphant(self, dir_export: str):
        r"""
        Purpose:
            - to export the egsphant file of the plan into dir_export
        Inputs:
            - dir_export := path to the directory where the export happens
        Outputs:
            - void := self.egsphant is written to ct.egsphant
        Dependencies:
            - BrachyEgsphant
        """
        file_path = dir_export + "/ct.egsphant"
        self.egsphant.write_to_ctegsphant(file_path)

    def _export_applicator_geometry(self, dir_export: str, format: str = "RapidBrachy"):
        r"""
        Purpose:
            - To export the applicator geometries either in the RapidBrachy Format (mac files and single json file)
            or in webapp format (json file).
        Inputs:
            - dir_export := path to the directory where the export happens
            - format := the format of the applicator geometry file. options are "RapidBrachy" or "WebApp"
        Outputs:
            - Void := will export the applicator geometries into the specified export directory.
        Dependencies:
            - None
        """
        # raise NotImplementedError("to be implemented soon")
        if format == "RapidBrachy":

            # initialize the fields of the json file:
            out_json = {
                "densities": [],
                "filenames": [],
                "materials": [],
                "points": [],
                "shieldNormalx": 0,
                "shieldNormaly": 0,
                "shieldNormalz": 0,
                "wRot": 0,
                "x": 0,
                "xRot": 0,
                "y": 0,
                "yRot": 0,
                "z": 0,
                "zRot": 0,
            }
            counter = 1
            for applicator in self.applicator_list:
                out_json["densities"].append(applicator.density)
                out_json["filenames"].append(applicator.pth_input_file)
                out_json["materials"].append(applicator.material)
                out_json["points"].append(applicator.catheter_trajectory)
                out_json["shieldNormalx"] = applicator.normal[0]
                out_json["shieldNormaly"] = applicator.normal[1]
                out_json["shieldNormalz"] = applicator.normal[2]
                
                subscript = "" if counter == 1 else counter
                out_json[f"wRot{subscript}"] = applicator.rotation[0]
                out_json[f"xRot{subscript}"] = applicator.rotation[1]
                out_json[f"yRot{subscript}"] = applicator.rotation[2]
                out_json[f"zRot{subscript}"] = applicator.rotation[3]
                
                out_json["x"] = applicator.coordinates[0]
                out_json["y"] = applicator.coordinates[1]
                out_json["z"] = applicator.coordinates[2]
                counter += 1
            
            with open(dir_export + "/applicator_geometry.json", "w") as file:
                json.dump(out_json, file, indent=4)
                
        elif format == "WebApp":
            out_json = [
                applicator.to_dict(format) for applicator in self.applicator_list
            ]
            with open(dir_export + "/applicator_geometry.json", "w") as file:
                json.dump(out_json, file, indent=4)
            
        else:
            raise ValueError("format should be either 'RapidBrachy' or 'WebApp'")

    def _export_structure_set(
        self, dir_export: str, export_format: str = "RapidBrachy"
    ):
        r"""
        Purpose:
            - to export the structure set of the plan into dir_export
        Inputs:
            - dir_export := path to the directory where the export happens
        Outputs:
            - void := self.structure_list is exported as a dictionary and
            written to structure_set.json
        Dependencies:
        """

        structure_set = []
        for structure in self.structure_list:
            structure_set.append(structure.to_dict(export_format))

        file_path = os.path.join(dir_export, "structure_set.json")
        with open(file_path, "w") as file:
            json.dump(structure_set, file, indent=4)

    def info(self):
        r"""
        Purpose:
            - to print the information of the plan
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will print the information of the plan
        Dependencies:
            - None
        """

        print("****BrachyPlan Information****")
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                print(f"{attr} := {value.shape}")
            elif isinstance(value, list):
                print(f"{attr} := {len(value)}")
            else:
                print(f"{attr} := {value}")


def _resize_structure_mask(structure_mask, target_shape):
    r"""
    Purpose:
        - To resize the structure mask to match the target shape.
    Inputs:
        - structure_mask:np.array := the structure mask to be resized.
        - target_shape:tuple := the target shape to which the structure mask will be resized.
    Outputs:
        - np.array := the resized structure mask
    """
    return ndimage.zoom(
        structure_mask, np.array(target_shape) / structure_mask.shape, order=0
    )


def _export_single_dose_rate(
    dose_grid: np.array,
    dwell_number: int,
    uncertainty: np.array = None,
    doseObj_template: BrachyDose = None,
    dir_export: str = None,
    dose_type: str = None,
):
    r"""
    Purpose:
        to write out a single dose rate map given the numpy grid for dose and uncertainty and
        a template dose object that has the same origin, voxel spacing and axis.
    Inputs:
        - dose_grid := the numpy array holding the dose rate maps
        - dwell_number:= the dwell number of the dose rate map
        - doseObj_template := a BrachyDose object that has the same origin, voxel spacing and axis
        - dir_export:= the directory to which the dose rate maps will be exported
        - dose_type := the type of dose rate map to be exported. options are ".3ddose", ".minidos", or ".nrrd"
        - uncertainty := the numpy array holding the uncertainty maps

    Output:
        - Void := dose file is written to dir_export+f"/run_{dwell_number}"+dose_type
    """
    doseObj = dose_with_empty_grid_like(doseObj_template)
    doseObj.grid = dose_grid
    if uncertainty is not None:
        doseObj.uncertainty = uncertainty

    doseObj.write_brachydose_to_file(dir_export + f"/run_{dwell_number}" + dose_type)


def _load_structure_mask(
    pth_structure_source: str,
    structure_name_list: list = None,
):
    """
    Load structure mask from different file formats. The acceptable formats are dicom, nrrd, and json.
    In the case of dicom files, providng a folder containing the dicom RS files is also acceptable.

    Inputs:
        pth_structure_source (str): The path to the structure source file.
        structure_name_list (list): A list of structure names to load.

    Returns:
        dict: A dictionary containing the structure masks.

    Raises:
        NotImplementedError: If the structure source type is not implemented yet.
        ValueError: If the structure source type is not recognized.
    """
    # if a folder is given, we assume that the structure source is dicom files
    if os.path.isdir(pth_structure_source):
        pth_structure_source = glob(os.path.join(pth_structure_source, "RS*.dcm"))[0]
        if pth_structure_source is None:
            raise ValueError(
                "No dicom structure file starting with RS, ending with .dcm is found in the directory"
            )

    structure_source_type = os.path.splitext(pth_structure_source)[1]

    if structure_source_type == ".dcm":
        print("loading structure set from dicom files")
        structure_mask_dict = BrachyDicom(
            os.path.dirname(pth_structure_source)
        ).structure_mask_dict

    elif structure_source_type == ".nrrd":
        print("loading structure set from nrrd file")
        raise NotImplementedError(
            "loading structure set from .nrrd file is not implemented yet"
        )

    elif structure_source_type == ".json":
        raise NotImplementedError(
            "loading structure set from .json file is not implemented yet"
        )

    else:
        raise ValueError("structure source type is not recognized")

    return structure_mask_dict


def dvh_metric(
    dose: np.array,
    num_bins: int,
    total_dose_max: float,
    threshold: float,
    voxel_volume: float,
    normalize_dose_by=None,
):
    r"""This function calculates the accumulative DVH given a dose matrix
    for a structure in the treatment plan.

    Inputs:
        - dose: a 1-D dose array, dtype = numpy matrix of floats
        - num_bins: a large number in general: we recommend 10 times
        the maximum dose for all structures.
        - total_dose_max: maximum of dose of the structure of interest
        - threshold: percent volume at which a certain dose is recieved,
        for example, for PTV D90%, threshold is 90.
        for urethra D0.1cc becomes 0.1 cc / total urethra volume * 100
        - voxel_volume: volume of a single voxel in cm^3
        - normalize_dose_by: if desired, the dose axis of the DVH can be normalized to the target dose.

    Dependencies
        1. scipy.interpolate.interp1d()
        2. np.histogram()
        3. np.cumsum()

    Outputs
        f(threshold): this is D90 or D1cc depending on the input threshold
        cum_dvh: this is the cumulative DVH after adding the new volum to the old one
    """

    histogram, bins_edges = np.histogram(
        dose, bins=num_bins, range=(0, total_dose_max + 0.1)
    )
    vol_hist = histogram * voxel_volume
    vol_hist = np.append(np.trim_zeros(vol_hist, trim="b"), 0)

    cum_dvh = np.cumsum(vol_hist[::-1])[::-1]
    normalized_cum_dvh = cum_dvh * 100 / cum_dvh[0]
    if normalize_dose_by is not None:
        dvh_dose_axis = bins_edges[: len(cum_dvh)] / normalize_dose_by
    else:
        dvh_dose_axis = bins_edges[: len(cum_dvh)]
    # for debugging{ let's plot the normalized dvh. nomralization is done both on dose and volume domains
    # dvh_plot = plt.plot(dvh_dose_axis, normalized_cum_dvh)
    # plt.show()
    # }
    f = interpolate.interp1d(normalized_cum_dvh, dvh_dose_axis, kind="linear")

    # in future, one could pass the DVH plot to be stored in the structure object.
    return f(threshold), normalized_cum_dvh


def _load_single_dose_or_uncertainty_to_dict(
    pth_dose_rate: str, load_dose_or_uncertainty: str = "both"
):
    r""" "
    Purpose:
        - To load a single dose rate file into the BrachyPlan object.
        this is to be used in the case of multiprocessing.
    Inputs:
        - pth_dose_rate := path to the dose rate file
        - load_dose_or_uncertainty := either "dose", "uncertainty", or "both"
    Outputs:
        - dose_or_uncert_map := the dose rate or uncertainty map of the dwell position
        specified by the index.
            If load_dose_or_uncertainty == "both", then dose_or_uncert_map[0] is dose and
            dose_or_uncert_map[1] is uncertainty.
    Dependencies:
        - BrachyDose()
    """
    dose_obj = BrachyDose(pth_dose_rate)
    if load_dose_or_uncertainty == "both":
        dose_or_uncert_map = np.zeros((2, *dose_obj.grid.shape), dtype=np.float32)
        dose_or_uncert_map[0] = dose_obj.grid
        dose_or_uncert_map[1] = dose_obj.uncertainty

    elif load_dose_or_uncertainty == "uncertainty":
        try:
            dose_or_uncert_map = np.zeros_like(
                BrachyDose(pth_dose_rate).grid, dtype=np.float32
            )
            dose_or_uncert_map = dose_obj.uncertainty
        except AttributeError:
            warnings.warn(
                f"uncertainty map is not loaded from {pth_dose_rate}. Moving on...",
                stacklevel=2,
            )

    elif load_dose_or_uncertainty == "dose":
        dose_or_uncert_map = np.zeros_like(
            BrachyDose(pth_dose_rate).grid, dtype=np.float32
        )
        dose_or_uncert_map = dose_obj.grid
    else:
        raise ValueError(
            "load_dose_or_uncertainty should be either 'dose', 'uncertainty', or 'both'"
        )

    return dose_or_uncert_map


def _type_nested_dict_list(data):

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                _type_nested_dict_list(value)
            else:
                print(f"{key}: {type(value)}")

    elif isinstance(data, list):
        for item in data:
            _type_nested_dict_list(item)
