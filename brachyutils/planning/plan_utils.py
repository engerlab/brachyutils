from itertools import chain
import re
import json
import os

import warnings
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import List, Literal, Union, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from opentps.core.data import ROIContour
from opentps.core.data.images import ROIMask

from tqdm import tqdm

from brachyutils.brachy_types import BrachyDose, DwellPosition

# from brachyutils.egsphant_utils import BrachyEgsphant
from brachyutils.geometry.applicator_utils import BrachyApplicator 
from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.geometry.catheter_utils.catheter_table import CatheterTable
from brachyutils.planning.structure_utils import BrachyStructure
from brachyutils.planning.simulation_utils import BrachySimulation
from brachyutils.planning.optimization.optim_configs import (
    Optimization_Config, Constraint_Config)
from pydantic import computed_field
from brachyutils.planning.plan_export_configs import (ExportConfig_BrachyPlan, ExportConfig_CatheterTable,
    ExportConfig_Egsphant, ExportConfig_PlanAndMac)
import pydicom
from glob import glob

pth_brachyutils = Path(__file__).parent.parent.parent.resolve()


class BrachyPlan:
    r"""
    ### Purpose:
    - This class holds the information regarding the brachytherapy treatment plan
    as well as all the functions to support the necessary plan operations.

    ### Attributes:

    #### Geometry and Structure Attributes:
    - phantom (BrachyPhantom): A BrachyPhantom object containing the patient geometry and structures.
    - structure_list (List[BrachyStructure]): A list of BrachyStructure objects containing the patient structures.
    - structure_dict (Dict[str, BrachyStructure]): A dictionary of the BrachyStructure objects with the
    phantom structure names as keys. This is automatically computed from the structure_list.
    - body_contour (ROIContour): The body contour of the patient.
    - phantom_origin (list): The origin of the phantom in the patient coordinate system.
    - organ_bounds (list): Min and max coordinates of the patient organs on each axis.
    - dvh_metric_goals (dict): Dictionary containing the DVH metric goals for the plan. The keys are
    the structure names from phantom. The names are of the format V_{#Gy|%}(organName),
    where # represents the numerical threshold and "|" is or. For example D95%(organName).
    
    - dvh_metrics_observed (dict): Dictionary containing the observed DVH metrics for the plan.
    - prescription_dose (float): The dose prescribed to the target volume.

    #### Catheter and Dwell Position Attributes:
    - catheter_table (CatheterTable): A catheter table object containing the catheter information.
    - num_catheters (int): The number of catheters in the plan.

    #### Applicator Attributes:
    - applicator_list (List[BrachyApplicator]): The list of all applicators in the plan.
    - applicator_rotation_axis (np.array): The rotation axis of applicators (default: [0, 0, 1]).
    - applicator_rotation_origin (np.array): The rotation origin of applicators (default: [0, 0, 0]).

    #### Dose Attributes:
    - combined_dose (BrachyDose): Sum of the dose rate maps weighted by the dwell times.

    #### Simulation and Optimization Attributes:
    - simulation_setup (BrachySimulation): A simulation setup object containing source info and simulation parameters.
    - optimization_config_dict (Dict[str, Optimization_Config]): Dictionary of optimization configurations for the plan.
    The keys are the structure names from phantom (usually loaded from DICOM RS). The structure name attribute
    could be a substring of the structure name in the phantom. For example, "CTV" in "CTV_BRACHY".
    - optimization_constraint_dict (Dict[str, Constraint_Config]): Dictionary of optimization constraints for the plan.
    see Constraint_Config for more details.
    """

    def __init__(
        self,
        #### for geometry definition:
        phantom: Union[Path, BrachyPhantom, dict] = None,
        #### for structure creation:
        prescription_dose: float | str | Path = None,
        #### for loading catheter table and/or applicators:
        catheter_table: Union[Path, CatheterTable, str] = None,
        applicator_pth_list: Union[Path, str, list] = None,
        #### for loading dose or uncertainty:
        combined_dose: Union[Path, str, BrachyDose] = None,
        dir_dose_rate: Path = None,
        #### for simulation setup:
        simulation_setup: dict | Path | str = None,
        #### for optimization setup:
        optimization_config_list:  List[Optimization_Config] | Path | str = None,
        **kwargs
    ):
        r"""
        ### Purpose:
        - To initialize the BrachyPlan object.

        ### Inputs:
        #### For geometry definition:
        - phantom: Path|BrachyPhantom|dict := the phantom object, the path to the phantom directory,
        or a dictionary containing the paths. A phantom object can include structures as well. See load_phantom() for more info.

        #### For Structure optimization and dosimetry
        - prescription_dose: float | str | Path = None := The dose that is prescribed to the target volume.
        This is used to calculate the DVH metrics. It can be a float or the path to a dicom file with prescription dose.

        #### For loading catheter table and applicators:
        - catheter_table: Path | CatheterTable := A catheter table object or the path to a json file containing the information of the catheter table.
        - applicator_pth_list := The list of applicator paths or the path to the json file containing the list. see load_applicator_list() for more info.

        #### for loading dose rates or uncertainty maps per dwell position:
        - combined_dose: Path|BrachyDose := the path to the combined dose file or a BrachyDose object.
        - dir_dose_rate:str := path to the directory containing the dose rate files for a patient.

        #### for simulation setup:
        - simulation_setup = None := dictionary containing the simulation setup,
        - load_uncertainty:bool := If True, the uncertainties of the dose rates are also loaded. 
        - multi_processing:bool :=  If True, 16 threads are used to load the dose rates simulatenously.
        - combined_dose_only:bool := If True, all the dose rates will be removed from memory after 
        combined dose is calculated.
        - dose_dtype:np.float32 := The floating point type to store the dose rates. 

        #### for optimization setup:
        - optimization_config_list: List[Optimization_Config] | Path | str := A list of
        Optimization_Config objects or the path to a json file containing the list.

        #### Keywords Arguments:
        - from_delivered_dwellpositions: bool = True := if True, will only load the dwell positions that
        had non-zero dwell times in the DICOM plan file. If False, will load all the dwell positions
        from the digitization points.
        - dwells_near_ptv: bool = True := if True, will remove the dwell positions that are outside PTV
        with a margine of 10 mm.
        - add_hotspots_to_phantom: bool = False := if True, will add hotspot structures to the phantom.
        this is good for debugging, but slows down the plan creation process.
        - one_hotspot_structure: bool: = True := if False, will create separate hotspot structures
        - applicator_format:str = "RapidBrachy" := the format of the applicator list 
        (default is "RapidBrachy"). See load_applicator_list() for more info. 
        - load_uncertainty: bool := If true, it will the uncertainty of the dose rates as well.
        - dvh_metric_goals: List[str] | Dict[str, float] | Path | str := A list of all DVH metric
        names or the dictionary containing the DVH metric names and their goals or the path to its
        json file. Look at set_dvh_metric_goals for guideline on the names of the DVH metrics.
        The phantom should be loaded with structures for the Brachy stuctures to be created. The names 
        are of the format V_{#Gy|%}(organName), where # represents the numerical threshold and "|" is or.
        For example D95%(organName).
        - strict_name_match: bool = True := If True, the name of the structure in the phantom and the DVH metric
        as well as the structure name in the optimization config should match perfectly. Otherwise, the name 
        of the structure in the DVH metric goals and optimization config can be a substring of the name of
        the structure in the phantom. For example, "CTV" in "CTV_BRACHY".
        - non_overlapping_structures := If true, all the structures that overlap with the target volume
        will be carved out of it. Use it wisely my friend!

        ### Outputs:
            - None := will initialize the BrachyPlan object
        """
        # declare the attributes
        # patient origin is used as a reference point for the catheter table,
        # the dwell coordinates, image origin, egsphant, and the dose objects.

        # phantom and geometry attributes
        self.phantom: BrachyPhantom = None
        self.dvh_metric_goals: dict = None
        self._structure_dict: Dict[BrachyStructure] = None
        self.structure_list: List[BrachyStructure] = []
        self.target_structure_names:List[str] = None
        self.body_contour: ROIContour = None
        self.phantom_origin: list = None  # np.array([0, 0, 0])  # x,y,z
        self.organ_bounds: list = None

        # catheter table attributes
        self.catheter_table: CatheterTable = None
        # applicator attributes
        self.applicator_list: List[BrachyApplicator] = []
        self.applicator_rotation_axis: np.array = np.array([0, 0, 1])  # x,y,z
        self.applicator_rotation_origin: float = np.array([0, 0, 0])  # x,y,z

        # simulation attributes
        self.simulation_setup: BrachySimulation = None

        # optimization attributes
        self.optimization_config_dict: Dict[str, Optimization_Config] = None
        self.optimization_constraint_dict: Dict[str, Constraint_Config] = None
 
        ## fill the attributes depending on the inputs to the constructor
        # load the prescription dose if the path is provided.
        if prescription_dose is not None:
            if isinstance(prescription_dose, Path) or isinstance(prescription_dose, str):
                prescription_dose = Path(prescription_dose)
                dicom_dose = pydicom.dcmread(prescription_dose)
                try:
                    prescription_dose = float(dicom_dose[0x3007, 0x1000].value)
                except KeyError:
                    raise KeyError(f"No prescription dose found at {prescription_dose}, please set it manually.")
        self.prescription_dose = prescription_dose

        # load the dicom plan if the path is provided
        if phantom is not None:
            if isinstance(phantom, BrachyPhantom):
                self.phantom = phantom
            elif (
                isinstance(phantom, Path)
                or isinstance(phantom, str)
                or isinstance(phantom, dict)
            ):
                self.load_phantom(phantom)
            else:
                raise ValueError("phantom should be a BrachyPhantom object or a path")
        # create structures based on the phantom structures and DVH metric goals
        if self.phantom is not None:
            if self.phantom.structure_set is not None:
                self.set_brachy_structure_list(
                    phantom=self.phantom,
                    non_overlapping_structures=kwargs.get("non_overlapping_structures", False)
                )

        if kwargs.get("dvh_metric_goals", None) is not None:
            dvh_metric_goals = kwargs.get("dvh_metric_goals")
            if self.prescription_dose is None:
                raise ValueError("prescription dose is not provided. Please provide it.")
            if isinstance(dvh_metric_goals, str) or isinstance(dvh_metric_goals, Path):
                dvh_metric_goals = Path(dvh_metric_goals)
                with open(dvh_metric_goals, "r") as json_file:
                    dvh_metric_goals = json.load(json_file)
            if isinstance(dvh_metric_goals, list):
                self.set_dvh_metric_goals(
                    dvh_metric_names=dvh_metric_goals,
                    strict_name_match=kwargs.get("strict_name_match", True))
            elif isinstance(dvh_metric_goals, dict):
                self.set_dvh_metric_goals(
                    dvh_metric_goals=dvh_metric_goals,
                    strict_name_match=kwargs.get("strict_name_match", True))
        # load the catheter table if the path is provided
        if catheter_table is not None:
            self.set_catheter_table(
                catheter_table=catheter_table,
                from_delivered_dwellpositions=kwargs.get("from_delivered_dwellpositions", False),
                dwells_near_ptv=kwargs.get("dwells_near_ptv", True),
                )
        # load the dose rate dict if the path is provided
        if dir_dose_rate is not None and combined_dose is None:
            self.catheter_table.load_dose_rates(
                dir_dose_rate=dir_dose_rate,
                load_uncertainty=kwargs.get("load_uncertainty", False),
                multi_processing=kwargs.get("multi_processing", True),
                combined_dose_only=kwargs.get("combined_dose_only", False),
                dose_dtype=kwargs.get("dose_dtype", np.float32),
                )
        elif dir_dose_rate is None and combined_dose is not None:
            self.catheter_table.set_combined_dose(combined_dose)
        elif dir_dose_rate is not None and combined_dose is not None:
            raise ValueError(
                "invalid input. Please provide either dir_dose_rate or combined_dose but not both"
            )
        
        # # load the simulation setup if the dictionary is provided
        if simulation_setup is not None:
            if isinstance(simulation_setup, dict):
                self.simulation_setup = BrachySimulation(
                    **simulation_setup
                )
            elif isinstance(simulation_setup, Path) or isinstance(
                simulation_setup, str
            ):
                # if json file, load the entire simulation dict from json file
                if str(simulation_setup).endswith(".json"):
                   self.simulation_setup = BrachySimulation(
                    pth_simulation_setup=simulation_setup
                )
                # if dicom plan file, load the source from the dicom file
                # and assuming the catheter table is loaded from the same dicom file,
                # provide the total time from the catheter table
                elif str(simulation_setup).endswith(".dcm"):
                    self.simulation_setup = BrachySimulation(
                        brachy_source=simulation_setup,
                        total_time=self.catheter_table.treatment_time if self.catheter_table else 0,
                        )

        # load the applicator list if the path is provided
        if applicator_pth_list is not None and applicator_format is not None:
            self.load_applicator_list(
                applicator_pth_list, kwargs.get("applicator_format", "RapidBrachy"))

        # # setup optimization
        if optimization_config_list is not None:
            self.setup_optimization(
                optimization_config_list,
                self.structure_list,
                add_hotspots_to_phantom=kwargs.get("add_hotspots_to_phantom", False),
                one_hotspot_structure=kwargs.get("one_hotspot_structure", True),
                strict_name_match=kwargs.get("strict_name_match", True)
            )

    @computed_field
    def combined_dose(self):
        return self.catheter_table.combined_dose

    @computed_field
    def structure_dict(self):
        if self._structure_dict is None:
            self._structure_dict = {structure.name: structure for structure in self.structure_list}
        return self._structure_dict

    def load_phantom(self, pth_phantom: Union[Path, dict]):
        r"""
        ### Purpose:
        - To load phantom from file path into Brachy Plan. Not that if a directory is provided,
        it should have only one phantom file.

        ### Inputs:
        - pth_phantom:str := The phantom path could be a directory of DICOM files
        or a directory of NRRD files. In addition, it could be the path to a json
        file containing paths to specific phantom files. Look at the inputs of BrachPhantom
        for more information on the expected keys of the json file.

        ### Outputs:
        - None := will update the BrachyPlan.phantom attribute
        """
        os.path.exists(pth_phantom), f"phantom path does not exist: {pth_phantom}"
        # initialize the inputs to the BrachyPhantom object
        dir_dicom = None
        pth_phantom_file = None
        pth_structures_file = None
        pth_egsphant_file = None
        # if the paths are provided as a dictionary
        if isinstance(pth_phantom, dict):
            phantom_config = pth_phantom
            for key in phantom_config:
                if key == "dir_dicom":
                    dir_dicom = phantom_config.get(key)
                elif key == "pth_phantom_file":
                    pth_phantom_file = phantom_config.get(key)
                elif key == "pth_structures_file":
                    pth_structures_file = phantom_config.get(key)
                elif key == "pth_egsphant_file":
                    pth_egsphant_file = phantom_config.get(key)
        # check if the pth_phantom is a directory or a json file
        elif os.path.isdir(pth_phantom):
            print("loading phantom from directory")
            # check if the directory contains dicom files or nrrd files
            file_list = glob(os.path.join(pth_phantom, "*.dcm"))
            if len(file_list) > 0:
                dir_dicom = pth_phantom
                pth_structures_file = list(filter(lambda x: "RS" in x, file_list))[0]
            else:
                file_list = glob(os.path.join(pth_phantom, "*.nrrd"))
                if len(file_list) > 0:
                    for file_name in file_list:
                        if file_name.endswith(".seg.nrrd"):
                            pth_structures_file = file_name
                        elif file_name.endswith(".egsphant.nrrd"):
                            pth_egsphant_file = file_name
                        elif file_name.endswith(".nrrd"):
                            pth_phantom_file = file_name
                    if pth_egsphant_file is None:
                        pth_egsphant_file = glob(
                            os.path.join(pth_phantom, "ct.egsphant.nrrd")
                        )
                        if len(pth_egsphant_file) == 0:
                            pth_egsphant_file = None
                else:
                    raise ValueError(
                        "invalid directory. Please provide a directory containing dicom or nrrd files"
                    )
        else:
            assert (
                os.path.splitext(pth_phantom)[1] == ".json"
            ), "invalid file format. Please provide a json file"
            with open(pth_phantom, "r") as json_file:
                phantom_config = json.load(json_file)
            for key in phantom_config:
                if key == "dir_dicom":
                    dir_dicom = phantom_config.get(key)
                elif key == "pth_phantom_file":
                    pth_phantom_file = phantom_config.get(key)
                elif key == "pth_structures_file":
                    pth_structures_file = phantom_config.get(key)
                elif key == "pth_egsphant_file":
                    pth_egsphant_file = phantom_config.get(key)

        # load the phantom
        self.phantom = BrachyPhantom(
            dir_dicom=dir_dicom,
            pth_phantom_file=pth_phantom_file,
            pth_structures_file=pth_structures_file,
            pth_egsphant_file=pth_egsphant_file,
        )
        self.phantom_origin = self.phantom.image_obj.origin

    def set_catheter_table(
        self,
        catheter_table: Union[Path, CatheterTable],
        from_delivered_dwellpositions: bool = True,
        dwells_near_ptv: bool = True,):
        r"""
        ### Purpose:
        - To set the catheter table of the plan and update the plan attributes accordingly.

        ### Inputs:
        - catheter_table: Path | CatheterTable := A catheter table object or the path to a
        json file containing the information of the catheter table.
        - from_delivered_dwellpositions: bool := Whether to load only the delivered dwell positions
        from the catheter table. Only applicable if the catheter table is loaded from a DICOM plan file. 
        If False, all the dwell positions from the digitization points will be loaded.
        - dwells_near_ptv: bool := Whether to remove dwells that are far from the PTV.

        ### Outputs:
        - None := will update the BrachyPlan.catheter_table attribute and all the related attributes
        such as dwell times, coordinates, etc.
        """
        if isinstance(catheter_table, (str, Path)):
            self.catheter_table = CatheterTable(
                catheters_dict=catheter_table,
                from_delivered_dwellpositions=from_delivered_dwellpositions,
                )
        elif isinstance(catheter_table, CatheterTable):
            if self.catheter_table is None:
                self.catheter_table = catheter_table
            else:
                self.catheter_table.merge(catheter_table)
        else:
            raise ValueError(
                "catheter_table should be a path or a CatheterTable object"
            )
        if dwells_near_ptv:
            for structure in self.structure_list:
                if structure.is_target:
                    if isinstance(structure.mask, ROIContour):
                        mask = structure.mask.getBinaryMask(
                            origin=self.phantom.image_obj.origin,
                            gridSize=self.phantom.image_obj.gridSize,
                            spacing=self.phantom.image_obj.spacing,
                        )
                    else:
                        mask = structure.mask
                    self.catheter_table.remove_outside_mask(
                        mask=mask,
                        margin_mm=5.0,
                    )

    def set_dvh_metric_goals(
        self,
        dvh_metric_names: List[str] | Path = None,
        dvh_metric_goals: dict | Path = None,
        strict_name_match: bool = True
        ) -> None:
        r"""
        ### Purpose:
        - To set the dvh metric list of the BrachyPlan object and each of the BrachyStructures.
        You can provide either the dvh_metric_names or the dvh_metric_goals.

        ### Inputs:
        - dvh_metric_names: List[str] := a list containing the DVH metrics for this structure. The names 
        are of the format V_{#Gy|%}(organName), where # represents the numerical threshold and "|" is or.
        For example D95%(organName).
        - dvh_metric_goals:Dict[str, float] := a dictionary of DVH metrics and their clinical goals.
        The keys should be following the same convention as for dvh_metric_names.
        - strict_name_match: bool := If True, the name of the structure in the phantom and the DVH metric
        goals should mask perfectly. Otherwise, the name of the structure in the DVH metric goals can be
        a substring of the name of the structure in the phantom. For example, "CTV" in "CTV_BRACHY".

        ### Outputs:
        - None := will update the BrachyPlan.dvh_metric_goals attribute as well as 
        BrachyStructure.dvh_metric_names and BrachyStructure.dvh_metric_goals.
        The keys of the BrachyPlan.dvh_metric_goals are:
        {
            BrachyStructure.name: {
                "dvh_metric_names: [list of the names for that structure],
                "dvh_metric_goals: {if applicable}
            }    
        }
        """
        if dvh_metric_names is not None and dvh_metric_goals is not None:
            raise ValueError("Please provide either dvh_metric_names or dvh_metric_goals, not both") 

        if len(self.structure_list) == 0:
            raise ValueError("The plan structure set is empty, please run set_brachy_structure_list")

        if isinstance(dvh_metric_names, Path):
            with open(dvh_metric_names, "r") as json_file:
                dvh_metric_names = json.load(json_file)

        if dvh_metric_goals is not None:
            if isinstance(dvh_metric_goals, Path):
                with open(dvh_metric_goals, "r") as json_file:
                    dvh_metric_goals = json.load(json_file)
            dvh_metric_names = list(dvh_metric_goals.keys())

        self.dvh_metric_goals = {}
        # let's match the structure names in the DVH with the structure names
        # in the BrachyPlan.
        for brachy_structure in self.structure_list:
            # get all the dvh metrics for this structure
            structure_dvh_metrics_names = []
            for dvh_name in dvh_metric_names:
                structure_name_in_dvh = dvh_name.split("(")[-1].split(")")[0]
                if strict_name_match:
                    if brachy_structure.name == structure_name_in_dvh:
                        structure_dvh_metrics_names.append(dvh_name)
                    else:
                        continue
                else:
                    if structure_name_in_dvh in brachy_structure.name:
                        structure_dvh_metrics_names.append(dvh_name)
                    else:
                        continue
            if len(structure_dvh_metrics_names) == 0:
                continue
            self.dvh_metric_goals[brachy_structure.name] = {
                "dvh_metric_names": structure_dvh_metrics_names}
            brachy_structure.set_dvh_metric_names(
                self.dvh_metric_goals.get(
                    brachy_structure.name).get(
                        "dvh_metric_names")
            )
            if dvh_metric_goals is not None:
                structure_dvh_metric_goals = {}
                for dvh_name in structure_dvh_metrics_names:
                    structure_dvh_metric_goals[
                        dvh_name] = dvh_metric_goals[dvh_name]
                self.dvh_metric_goals[
                    brachy_structure.name][
                        "dvh_metric_goals"] = structure_dvh_metric_goals 
                brachy_structure.set_dvh_metric_goals(
                    self.dvh_metric_goals.get(brachy_structure.name).get("dvh_metric_goals")
                )

    def set_brachy_structure_list(
        self,
        phantom: BrachyPhantom,
        mask_type: Union[ROIContour, ROIMask] = ROIMask,
        non_overlapping_structures:bool = False
        )->None:
        r"""
        ### Purpose:
        - To create a list of BrachyStructure objects from the structures in the phantom.
        Each BrachyStructure object will have attributes for the structure
        contour, the DVH and uncertainty volume histograms, optimization attributes,
        and simulation attributes. Here, we only set the mask. We also ensure that there is
        no overlap between the mask of the target structure and the OARs, priority is given
        to OAR.

        ### Inputes:
        - phantom := the phantom with its structures fully loaded.
        - dvh_metric_goals := the dvh metric goals dictionary
        - mask_type: ROIContour | ROIMask := Phantom masks will be converted to this type when being
        stored in BrachySturucture.
        - non_overlapping_structures := If true, all the structures that overlap with the target volume
        will be carved out of it. Use it wisely my friend!
        ### Outputs:
        - None := will update the BrachyPlan.structure_list attribute
        """
        self.structure_list = []
        if phantom.cached_structure_masks is not None:
            structure_masks = deepcopy(phantom.cached_structure_masks)
        else:
            structure_masks: dict = phantom.get_structure_mask(
                phantom.structure_names, mask_type,
            )

        for structure_name in structure_masks.keys():
            structure_obj = BrachyStructure(
                name=structure_name,
                mask=structure_masks[structure_name],
                is_target=True if (
                    "ctv" in structure_name.lower()
                    or "ptv" in structure_name.lower())  else False,
            )
            self.structure_list.append(structure_obj)
        self.body_contour = phantom.get_structure_mask(
            ["body"],
            mask_type=ROIContour,
            strict_name_match=False,).get("body", None)

        self.target_structure_names = [
            structure.name for structure 
            in self.structure_list if structure.is_target]

        if non_overlapping_structures:
            # If there is an OAR that goes through the PTV, cut it out of PTV.
            # this is because each voxel can have planning role only.
            # in prostate, the urethra goes through the PTV.
            for target_structure in self.target_structure_names:
                for structure in self.structure_list:
                    if structure.is_target:
                        continue
                    if "body" in structure.name.lower():
                        continue 
                    # find intersection between this structure and target_structure
                    overlap = (
                        self.structure_dict.get(target_structure).mask.imageArray 
                        &  structure.mask.imageArray)
                    if np.any(overlap):
                        self.structure_dict.get(target_structure).mask.imageArray = (
                            self.structure_dict.get(target_structure).mask.imageArray 
                            & (np.ma.ones_like(overlap)^overlap))
   
    def get_dvh_metrics(
        self,
        combined_dose: BrachyDose=None,
        prescription_dose: float = None,
        return_percentage: bool = True,
        ) -> dict:
        r"""
        ### Purpose:
        - To get the observed value of the dvh metric for each structure in the BrachyPlan.
        the observed value is calculated from the combined dose map.

        ### Inputs:
        - self := the BrachyPlan object

        ### Outputs:
        - dvh_metrics_observed: a dictionary mapping every DVH metric to its observed value.
        """
        assert self.structure_list is not None, "structure list is not created yet"
        assert self.prescription_dose is not None, "prescription dose is not set"
        assert self.dvh_metric_goals is not None, "DVH metrics are not set, please run set_dvh_metric_goals()"
        if combined_dose is None:
            combined_dose = self.combined_dose
        if prescription_dose is None:
            prescription_dose = self.prescription_dose
        dvh_metrics_observed = {}
        for structure_obj in self.structure_list:
            if "hotspot_estimator" in structure_obj.name.lower():
                continue
            if not any(structure_obj.dvh_metric_names):
                continue
            observed_metrics = structure_obj.get_dvh_metric(
                combined_dose,
                prescription_dose,
                return_percentage,
                self.body_contour,
                )
            dvh_metrics_observed.update(observed_metrics)
        return dvh_metrics_observed

    def export_dvh_metrics(self, output_pth: Union[str, Path]):
        r"""
        ### Purpose:
        - To export the dvh metrics of the BrachyPlan to a json file.

        ### Inputs:
        - output_pth := path to the output json file

        ### Outputs:
        - None := will export the dvh metrics to a json file
        """
        assert self.dvh_metrics_observed is not None, "dvh metrics are not calculated yet"
        assert output_pth.endswith(".json"), "output path should be a json file"
        with open(output_pth, "w") as json_file:
            json.dump(self.dvh_metrics_observed, json_file, indent=4)
    
    def export_dvh_metric_goals(self, output_pth: Union[str, Path]):
        r"""
        ### Purpose:
        - To export the dvh metric goals of the BrachyPlan to a json file.

        ### Inputs:
        - output_pth := path to the output json file

        ### Outputs:
        - None := will export the dvh metric goals to a json file
        """
        assert self.dvh_metric_goals is not None, "dvh_metric_goals object is not created yet"
        assert output_pth.endswith(".json"), "output path should be a json file"
        with open(output_pth, "w") as json_file:
            json.dump(self.dvh_metric_goals, json_file, indent=4)

    def calculate_uncertainty_per_structure(self):
        r"""
        ### Purpose:
        - To calculate the uncertainty of each structure in the BrachyPlan.

        ### Inputs:
        - self := the BrachyPlan object

        ### Outputs:
        - None := will update the BrachyStructure.uncertainty attribute
        """
        assert (
            self.combined_dose.uncertainty_image is not None
        ), "combined uncertainty is not calculated yet"
        assert self.structure_list is not None, "structure list is not created yet"
        from opentps.core.processing.imageProcessing.resampler3D import (
            resampleImage3DOnImage3D,
        )

        for structure_obj in self.structure_list:
            # resample the uncertainty image on the structure
            structure_mask = structure_obj.mask.getBinaryMask(
                origin=self.combined_dose.uncertainty_image.origin,
                spacing=self.combined_dose.uncertainty_image.spacing,
                gridSize=self.combined_dose.uncertainty_image.gridSize
            )
            masked_uncertainty = self.combined_dose.uncertainty_image.imageArray * structure_mask.imageArray

            # isolate the uncertainty values that are in the mask
            flattened_uncertainty = masked_uncertainty.flatten()
            # generate a histogram from the masked uncertainty
            histogram, bins_edges = np.histogram(
                flattened_uncertainty,
                bins=100,
                range=(0, flattened_uncertainty.max() + 0.1),
            )
            structure_obj.uvh = histogram * np.prod(self.combined_dose.dose_image.spacing)
            structure_obj.uncertainty_mean = np.mean(flattened_uncertainty)
            structure_obj.uncertainty_std = np.std(flattened_uncertainty)
            structure_obj.uncertainty_max = np.max(flattened_uncertainty)
            structure_obj.uncertainty_min = np.min(flattened_uncertainty)

    def export_brachy_plan(
        self,
        content_to_export: ExportConfig_BrachyPlan | dict,
        ):
        r"""
        ### Purpose:
        - To export the brachytherapy treatment plan and its components to files based on the
        provided export configuration. Supports exporting dose, catheter tables, plan files,
        simulation macros, egsphant files, applicator geometries, and structure sets.

        ### Inputs:
        - content_to_export := ExportConfig_BrachyPlan object or dictionary containing export
        configuration. If a dictionary is provided, it will be converted to an ExportConfig_BrachyPlan object.
        
        The configuration object/dictionary should contain:
            - dir_export (Path): Directory where exported files will be written.
            - export_config_dose (ExportConfig_Dose|None): Dose export configuration.
            - export_config_cathetertable (ExportConfig_CatheterTable|None): Catheter table export configuration.
            - export_config_plan_and_mac (ExportConfig_PlanAndMac|None): Plan and Mac file export configuration.
            - export_config_egsphant (ExportConfig_Egsphant|None): Egsphant file export configuration.
            - applicator_geometry (bool): Whether to export applicator geometry.
            - structure_set (bool): Whether to export structure set.

        ### Outputs:
        - None := Exported files are written to the directory specified in content_to_export.dir_export.
        The function conditionally exports the following file types based on configuration:
            - Dose files (combined dose and optionally dose rate maps)
            - Catheter table (.json or .mrk.json)
            - Plan files (.plan files for each dwell position)
            - Macro files (.mac simulation files)
            - Egsphant phantom file (ct.egsphant)
            - Applicator stl files
            - Structure set file (structure_set.json)

        ### Dependencies:
        - ExportConfig_BrachyPlan
        - export_dose()
        - export_catheter_table()
        - export_plan_files()
        - export_mac_files()
        - _export_egsphant()
        - _export_structure_set()

        ### Example of content_to_export dict:
        ```python
        EXPORT_CONFIG = {
            "dir_export": "/path/to/export/directory",
            "export_config_dose": {
                "name_combined": "combined",
                "write_dose_rate_maps": False,
                "multi_processing": True
            },
            "export_config_cathetertable": {
                "name": "catheter_table",
                "remove_text": True,
                "one_markup_per_catheter": False
            },
            "export_config_egsphant": {
                "name": "egsphant",
                "assign_material_from_ct": False,
                "strict_name_match": False
            },
            "export_config_plan_and_mac": {
                "combined_only": True,
                "name_combined": "combined",
                "body_mesh_name": "BODY"
            },
            "applicator_geometry": False,
            "structure_set": False
        }
        ```
        """
        if isinstance(content_to_export, dict):
            content_to_export = ExportConfig_BrachyPlan(**content_to_export)
        dir_export = content_to_export.dir_export
        dir_export.mkdir(parents=True, exist_ok=True)

        if content_to_export.export_config_dose:
            self.catheter_table.export_dose(content_to_export.export_config_dose)

        if content_to_export.export_config_cathetertable:
            self.export_catheter_table(
                export_config_cathetertable=content_to_export.export_config_cathetertable,
                catheter_table=self.catheter_table,
            )

        if content_to_export.export_config_egsphant:
            self._export_egsphant(
                export_config_egsphant=content_to_export.export_config_egsphant
            )

        if content_to_export.export_config_plan_and_mac:
            self.export_plan_files(
                export_config_plan_and_mac=content_to_export.export_config_plan_and_mac,
                catheter_table=self.catheter_table,
                )
            self.export_mac_files(
                export_config_plan_and_mac=content_to_export.export_config_plan_and_mac,
                catheter_table=self.catheter_table
                )


        if content_to_export.applicator_geometry:
            for applicator in self.applicator_list:
                applicator.to_stl(dir_export / f"{applicator.name}.stl")

        if content_to_export.structure_set:
            self._export_structure_set(
                str(dir_export), content_to_export.get("materials_table", None)
            )

    def export_catheter_table(
        self,
        export_config_cathetertable: ExportConfig_CatheterTable,
        catheter_table: CatheterTable,
        ):
        r"""
        ### Purpose:
        - to export the catheter table to a given directory in mrk.json or .json format.

        ### Inputs:
        - export_config_cathetertable: The catheter table export configuration. Look at ExportConfig_CatheterTable for more info
        - catheter_table: The catheter table to export.

        ### Outputs:
        - None := will export the catheter table into the specified export directory.
        """
        if export_config_cathetertable.file_extension == "mrk.json":
            catheter_table.write_to_slicer_markup(
                pth_mrk_json=export_config_cathetertable.pth_catheter_table,
                remove_text=export_config_cathetertable.remove_text,
                one_markup_per_catheter=export_config_cathetertable.one_markup_per_catheter,
            )
        elif export_config_cathetertable.file_extension == ".json":
            catheter_table.write_to_json(
                pth_json=export_config_cathetertable.pth_catheter_table
            )

    def export_plan_files(
        self,
        export_config_plan_and_mac:ExportConfig_PlanAndMac,
        catheter_table:CatheterTable,
        ):
        r"""
        ### Purpose:
        - To export dwell positions and their normalized times into ".plan" text files in the
        format required by RapidBrachy.

        ### Inputs:
        - export_config_plan_and_mac = The export configuration for the plan files. see ExportConfig_PlanAndMac
        - catheter_table:= The catheter table with the dwells.

        ### Outputs:
        - None := Two types of .plan files are written, one named combined.plan and the other
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

        ### Dependencies:
            - None
        """
        total_dwell_time = catheter_table.treatment_time
        num_dwells = catheter_table.num_dwell_positions
        combined_plan = "Treatment Plan\n"
        combined_plan += f"{num_dwells} Control Points\n"

        for cat in catheter_table:
            for dwell in cat.dwells:
                if not dwell.gen_dose_rate:
                    continue
                dwell_coordinates_str = np.array(
                    list(dwell.position)
                    + list(dwell.rotation)
                    + [dwell.angle]
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

                catheter_idx = cat.index
                dwell_idx = dwell.index
                combined_plan += "Control Point\n"
                combined_plan += f"weight = {dwell.time/total_dwell_time}\n"
                combined_plan += f"1 Dwell Position - Catheter {catheter_idx + 1}\n"
                combined_plan += dwell_coordinates_str

                run_i_plan = "Treatment Plan\n"
                run_i_plan += "1 Control Points\n"
                run_i_plan += "Control Point\nweight = 1.0\n"
                run_i_plan += "1 Dwell Position\n"
                run_i_plan += dwell_coordinates_str
                # Not dealing with shield angle for now but the new convention for filename is
                # xxx_catheter#_dwell#_shieldangle.plan
                shield_angle = dwell.angle
                if not export_config_plan_and_mac.combined_only:
                    order = f"{catheter_idx + 1}_{dwell_idx + 1}_{shield_angle}"
                    with open(export_config_plan_and_mac.dir_export / f"dwell_{order}.plan", "w") as file:
                        file.write(run_i_plan)
    
        if export_config_plan_and_mac.combined_only:
            with open(export_config_plan_and_mac.pth_plan_combined, "w") as file:
                file.write(combined_plan)

    def export_mac_files(
        self,
        export_config_plan_and_mac: ExportConfig_PlanAndMac,
        catheter_table:CatheterTable,
        ):
        r"""
        ### Purpose:
        - To export the simulation parameters of the plan into a macro files
        and run_{catheterNumber}_{dwellNumber}_{shieldAngle}.mac

        ### Inputs:
        - export_config_plan_and_mac:= The export configuration for macro files.
        - catheter_table:= The catheter table with the dwells.

        ### Outputs:
        - None := Two types of .mac files are written, one named combined.mac and the other
        named run_{catheterNumber}_{dwellNumber}_{shieldAngle}.mac. combined.plan contains

        plan contains info of a single dwell position.

        The format of each .mac file is given in this example:
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
        ### Dependencies:
        - simulation_utils
        """
        sim_obj = deepcopy(self.simulation_setup)
        sim_obj.total_time = catheter_table.treatment_time
        sim_obj.pth_plan = export_config_plan_and_mac.pth_plan_combined.name
        sim_obj.pth_phantom = export_config_plan_and_mac.pth_phantom
        sim_obj.applicator_list = self.applicator_list

        if export_config_plan_and_mac.auto_mvm:
            #check if we need it - if the dimensions of the image are sufficiently small
            if any(self.phantom.egsphant_obj.material_image.gridSizeInWorldUnit < 400): #if our phantom is smaller than 40 cm in any direction
                for structure in self.phantom.structure_names:
                    if structure.lower() == "body" or structure.lower() == "external":
                        export_config_plan_and_mac.body_mesh_name = structure
                    #material already defaults to soft tissue

        if export_config_plan_and_mac.body_mesh_name is not None:
            sim_obj.pth_body_stl = export_config_plan_and_mac.pth_body_stl.name
            sim_obj.body_material = export_config_plan_and_mac.body_mesh_material
            body_mask = self.phantom.get_structure_mask([export_config_plan_and_mac.body_mesh_name], mask_type = ROIMask, strict_name_match=False)[export_config_plan_and_mac.body_mesh_name]

            from brachyutils.geometry.phantom_utils import mask_to_stl

            mask_to_stl(
                roi_mask=body_mask,
                pth_output=export_config_plan_and_mac.pth_body_stl
            )


        if export_config_plan_and_mac.combined_only:
            with open(export_config_plan_and_mac.pth_mac_combined, "w") as file:
                file.write(sim_obj.to_string())

        else:
            for cat in catheter_table:
                for dwell in cat.dwells:
                    catheter_idx = cat.index
                    dwell_idx = dwell.index
                    # Not dealing with shield angle for now but the new convention for filename is
                    # xxx_catheter#_dwell#_shieldangle.plan
                    shield_angle = dwell.angle
                    order = f"{catheter_idx + 1}_{dwell_idx + 1}_{shield_angle}"
                    sim_obj.pth_plan = f"dwell_{order}.plan"
                    sim_obj.total_time = 1

                    with open(export_config_plan_and_mac.dir_export / f"run_{order}.mac", "w") as file:
                        file.write(sim_obj.to_string())
        print(".mac files were exported successfully")

    def _export_egsphant(
        self,
        export_config_egsphant: ExportConfig_Egsphant,
        ):
        r"""
        ### Purpose:
        - to export the egsphant file of the plan into dir_export

        ### Inputs:
        - export_config_egsphant:= The export configuration for egsphant file. see ExportConfig_Egsphant

        ### Outputs:
        - None := egsphant file is generated from phantom and is written to ct.egsphant

        ### Dependencies:
        - BrachyEgsphant
        """
        self.phantom.write_to_egsphant(
            pth_output=export_config_egsphant.pth_egsphant,
            material_dict=export_config_egsphant.material_dict,
            assign_material_from_ct=export_config_egsphant.assign_material_from_ct,
            crop_by_contour=export_config_egsphant.crop_by_contour,
            marginInMM=export_config_egsphant.marginInMM,
            resampled_spacing=export_config_egsphant.resampled_spacing,
            resampled_origin=export_config_egsphant.resampled_origin,
            background_material=export_config_egsphant.background_material,
            strict_name_match=export_config_egsphant.strict_name_match
        )

        print("Egsphant file was exported successfully")

    def _export_structure_set(
        self,
        dir_export: str,
        materials_table: Union[dict, Path] = None,
    ):
        r"""
        ### Purpose:
        - to export the structure set of the plan into dir_export

        ### Inputs:
        - dir_export := path to the directory where the export happens
        - material_table: dict | Path := the dictionary of the materials. if Path, the path to the material file.
        The dictionary contains the name of the elements for each voxel,
        and the following keys: [
            "density" := the density of the material in g/cm^3,
            "HU_limit" := the lower HU limit threshold of the material,
            "structure_name := {optional} the name of the structure in the dicom file that represents the material,"
        ]

        ### Outputs:
        - None := self.structure_list is exported as a dictionary and
        written to structure_set.json

        ### Dependencies:
        """
        raise NotImplementedError("now that you are here, finish this function thank you!") #no thanks, Jonathan.
        structure_set = []
        for structure in self.structure_list:
            structure_set.append(structure.to_dict())

            if materials_table is not None:
                from brachyutils.geometry.egsphant_utils import _load_material_dict

                material_dict = _load_material_dict(materials_table)
                for material in material_dict:
                    if (
                        material_dict[material].get("structure_name", "")
                        == structure.name
                    ):
                        structure_set[-1]["density"] = material_dict[material][
                            "density"
                        ]

        file_path = os.path.join(dir_export, "structure_set.json")
        with open(file_path, "w") as file:
            json.dump(structure_set, file, indent=4)
        print("structure set file was exported successfully")

    def info(self):
        r"""
        ### Purpose:
        - to print the information of the plan

        ### Inputs:
        - self := the BrachyPlan object

        ### Outputs:
        - None := will print the information of the plan

        ### Dependencies:
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

    def setup_optimization(
        self, 
        optimization_config_list:List[Optimization_Config] | Path | str,
        structure_list:List[BrachyStructure],
        add_hotspots_to_phantom:bool=False,
        one_hotspot_structure:bool=True,
        strict_name_match:bool = True,
        ):
        r"""
        ### Purpose:
        - Given the optimization config list either as a list or in a json file, put each
        optimization config inside the BrachyStructures. Also, create the hotspot estimator
        structure if needed.

        ### Inputs:
        - self := the BrachyPlan object
        - optimization_config_list := a list of Optimization_Config objects or a path to a json file
        that contains the list of optimization config objects. Look at Optimization_Config 
        for more info on the fields of the optimization config object.
        - structure_list := a list of BrachyStructure objects that represent the structures in the plan.
        - add_hotspots_to_phantom := whether to add the hotspot estimator structures to the phantom.
        - strict_name_match := whether to match the structure names exactly or as substrings.

        ### Outputs:
        - None := The structure objects in self.structure_list will be updated
        with the optimization config objects, and the hotspot estimator 
        structures will be created and added to self.structure_list if needed.
        """
        self._reset_optimization()
        self.optimization_config_dict = defaultdict(Optimization_Config)
        self.optimization_constraint_dict = defaultdict(Constraint_Config)

        if isinstance(optimization_config_list, (Path, str)):
            optimization_config_list = Path(optimization_config_list).resolve()
            if str(optimization_config_list).endswith(".json"):
                with open(optimization_config_list, "r") as json_file:
                    optimization_config_list = json.load(json_file)
            else:
                raise ValueError("optimization_config_list can be a json file or a list of Optimization_Config objects")
        target_structure_names = [
            structure.name.lower() for structure in self.structure_list
            if structure.is_target
            ]
        for config in optimization_config_list:
            if config.penalty_weight_hotspot != 0:
                hotspot_structure_name = config.structure_name.lower()
                structure_matched = any(
                    hotspot_structure_name
                    in name.lower() for name in target_structure_names)
                if not structure_matched:
                    raise ValueError(
                        "penalty_weight_hotspot can only be set for PTV or CTV structures"
                    )
                self._create_hotspot_structures(
                    target_optim_config=config,
                    add_hotspots_to_phantom=add_hotspots_to_phantom,
                    one_hotspot_structure=one_hotspot_structure)
            for struc in structure_list:
                if strict_name_match:
                    structure_matched = config.structure_name.lower() == struc.name.lower()
                else:
                    structure_matched = config.structure_name.lower() in struc.name.lower()
                if structure_matched:
                    assert config.is_target == struc.is_target, f"The target structure in plan and optimization \
config do not match for structure {struc.name}"
                    struc.set_optimization_config(config)
                    self.optimization_config_dict[struc.name] = config
                    # check if the structure is a target and catheter
                    # recommendation is not needed
                    if config.is_target and not (config.catheter_recommendaion):
                        # set constraints on the catheters
                        for catheter in self.catheter_table:
                            self.optimization_constraint_dict[catheter.name_id] = Constraint_Config(
                                constraint_type="bound",
                                variable_type="catheter",
                                equal=1,
                                variable_name_ids=[catheter.name_id]
                            )
                    break

    def _create_hotspot_structures(
        self,
        target_optim_config: Optimization_Config,
        add_hotspots_to_phantom:bool=False,
        one_hotspot_structure:bool=True
        ):
        r"""
        ### Purpose:
        - to create structures where hotspots are likely to occur inside the ptv or ctv.
        These structures are created as spheres with radius of dwell step size centered in 
        between two dwell positions that are within a step size distance from each other.
        There could be only one hotspot structure per each dwell pair.

        ### Inputs:
        - self := the BrachyPlan object
        - config := the optimization config object that contains the parameters for the hotspot structure

        ### Outputs:
        - None := hot spot structures are appended to the self.structure_list
        """
        step_size = self.catheter_table.step_size
        # identify unique dwell pairs that are withi n the step size distance
        dwell_pairs = []
        def distance(pos1, pos2):
            return np.linalg.norm(pos1 - pos2)
        def center(pos1, pos2):
            return (pos1 + pos2) / 2
        
        all_dwells:List[DwellPosition] = self.catheter_table.all_dwells

        for i in range(len(all_dwells)):
            for j in range(i + 1, len(all_dwells)):
                current_distance = distance(
                    all_dwells[i].position,
                    all_dwells[j].position) 
                if current_distance <= step_size:
                    dwell_pairs.append(
                        {
                            "dwell_pair": (
                                {
                                    "dwell": all_dwells[i].name_id,
                                },
                                {
                                    "dwell": all_dwells[j].name_id,
                                }),
                            "center": center(
                                all_dwells[i].position,
                                all_dwells[j].position
                            ),
                            "radius": step_size,
                            "distance": current_distance,
                            "inter-catheter": True if (
                                all_dwells[i].catheter_index
                                != all_dwells[j].catheter_index
                                ) else False
                        }
                    )
        # create hotspot structures masks for each dwell pair
        hotspot_mask_list = []
        
        if all_dwells[0].dose_rate is None:
            reference_image = self.phantom.image_obj
        else:
            reference_image = all_dwells[0].dose_rate.dose_image

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    _gen_hotspot_mask,
                    dwell_pair,
                    reference_image.gridSize,
                    reference_image.origin,
                    reference_image.spacing,
                ): dwell_pair for dwell_pair in dwell_pairs
            }
            for action in tqdm(
                as_completed(futures),
                desc="Generating hotspot estimator volumes",
                total=len(dwell_pairs)
            ):
                try:
                    hotspot_mask_list.append(action.result())
                except:
                    raise ValueError("failed building hotspot volumes")
                    
        if one_hotspot_structure:
            mask_union = np.zeros_like(
                hotspot_mask_list[0].mask.imageArray, dtype=bool
            )
            for mask in hotspot_mask_list:
                mask_union = np.logical_or(mask_union, mask.mask.imageArray)

            hotspot_config = Optimization_Config(
                structure_name="hotspot_estimator_combined",
                is_target=False,
                spacing_mm=target_optim_config.spacing_mm,
                dose_voxel_goal=target_optim_config.dose_voxel_goal*target_optim_config.hotspot_threshold,
                penalty_weight_linear=target_optim_config.penalty_weight_hotspot
            )
            hotspot_mask_list = [
                BrachyStructure(
                    name="hotspot_estimator_combined",
                    mask=ROIMask(
                        name="hotspot_estimator_combined",
                        imageArray=mask_union,
                        origin=self.phantom.image_obj.origin,
                        spacing=self.phantom.image_obj.spacing,
                    ),
                    is_target=False,
                    in_dvh=False,
                    optimization_config=hotspot_config
                )
            ]

        for mask in hotspot_mask_list:
            self.structure_list.append(mask)
            if add_hotspots_to_phantom:
                self.phantom.set_structure_set(
                    mask_dict={mask.name: mask.mask},
                    mask_colors={mask.name:[251, 159, 255]}
                    )

    def _reset_optimization(self):
        r"""
        ### Purpose:
        - to reset the optimization configurations of all structures in the plan.

        ### Inputs:
        - self := the BrachyPlan object

        ### Outputs:
        - None := optimization_config attribute of all structures in the plan is set to None
        """
        for structure in self.structure_list:
            if structure.name.startswith("hotspot_estimator_"):
                self.structure_list.remove(structure)
                self.phantom.remove_structure(structure.name)
                continue
            structure.optimization_config = None

    def get_dose_rate_matrices_for_catheter(
        self,
        catheter_index: int
    ) -> Dict[str, BrachyDose]:
        r"""
        ### Purpose:
        - to get the dose rate matrices for all dwell positions in a given catheter.
        this function assumes that dose rate dictionary matches the index+1 convension.

        ### Inputs:
        - catheter_index := the index of the catheter in the catheter table

        ### Outputs:
        - Dict[BrachyDose]: A dictionary containing the dose rates for the speicific catheter.
        the keys are in the format catheter_{index+1}_dwell_{index+1}
        """
        catheter = self.catheter_table[catheter_index]
        dose_rates_catheter = defaultdict(BrachyDose)
        for dwell in catheter.dwells:
            dose_rates_catheter[
                f"dwell_{dwell.name_id}"
                ] = dwell.dose_rate
        return dose_rates_catheter

def _gen_hotspot_mask(
    dwellpair: dict,
    gridSize: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    ):
    r"""
    ### Purpose:
    - to create structures where hotspots are likely to occur inside the ptv or ctv.
    These structures are created as spheres with radius of dwell step size centered in 
    between two dwell positions that are within a step size distance from each other.
    There could be only one hotspot structure per each dwell pair.

    ### Inputs:
    - dwellpair := dictionary containing the dwell pair information
        The information includes:
            - dwell_pair := a tuple of two dictionaries, each containing the dwell name_id and other
            information of the dwell position such as its position and dose rate if available.
            - center := the center of the hotspot structure as a tuple of three floats (x, y, z)
            - radius := the radius of the hotspot structure as a float
    - gridSize := the size of the grid as a tuple of three integers (x, y, z)
    - origin := the origin of the grid as a tuple of three floats (x, y, z)
    - spacing := the spacing of the grid as a tuple of three floats (x, y, z)

    ### Outputs:
    - None := hot spot structures are appended to the self.structure_list
    """
    from brachyutils.geometry.phantom_utils import generate_sphere_mask
        
    dwell_mask = generate_sphere_mask(
        center=dwellpair["center"],
        radius=dwellpair["radius"],
        gridSize=gridSize,
        origin=origin,
        spacing=spacing,
        name=(
            f"hotspot_estimator_dwell_{(dwellpair['dwell_pair'])[0]['dwell']}"
            + f"/dwell_{(dwellpair['dwell_pair'])[1]['dwell']}"
            ),
    )
    return BrachyStructure(
        name=dwell_mask.name,
        mask=dwell_mask,
        is_target=False,
        in_dvh=False,
    )

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

def load_dicom_to_plan(
    dir_dicom: Path | str,
    load_dicom_dose: bool = False,
    load_dicom_source: bool = True,
    load_dicom_catheter_table: bool = True,
    load_dicom_prescription_dose: bool = True,
    **kwargs) -> BrachyPlan:
    r"""
    ### Purpose:
    - To load all the contents of a dicom directory into a BrachyPlan object.

    ### Inputs:
    - dir_dicom := the path to the dicom directory.
    - load_dicom_dose := if True, the dose dicom file will be loaded.
    - load_dicom_source := if True, the source dicom file will be loaded.
    - load_dicom_catheter_table := if True, the catheter table dicom file will be loaded.
    - load_dicom_prescription_dose := if True, the prescription dose will be loaded from the dicom file.
    If False, the prescription dose will be set to None.
    - **kwargs := additional arguments to be passed to the BrachyPlan constructor

    ### Outputs:
    - BrachyPlan := the BrachyPlan object with all the contents of the dicom directory
    """
    dir_dicom = Path(dir_dicom)
    dose_dcm = list(dir_dicom.glob("[Rr][Dd]*.dcm"))
    plan_dcm = list(chain(
        dir_dicom.glob("[Rr][Pp]*.dcm"),
        dir_dicom.glob("[Pp][Ll]*.dcm"),
        ))

    if load_dicom_dose:
        if len(dose_dcm) != 1:
            raise FileNotFoundError("There should be exactly one dose dicom file that starts with RD or rd in the directory")
        combined_dose = kwargs.get("combined_dose", dose_dcm[0])
    else:
        combined_dose = None

    if load_dicom_source:
        if len(plan_dcm) != 1:
            raise FileNotFoundError("There should be exactly one source dicom file that starts with RP or PL in the directory")
        brachy_source = plan_dcm[0]
    else:
        brachy_source = None

    if load_dicom_catheter_table:
        if len(plan_dcm) != 1:
            raise FileNotFoundError("There should be exactly one catheter table dicom file that starts with RP or PL in the directory")
        catheter_table = kwargs.get("catheter_table", plan_dcm[0])
    else:
        catheter_table = None

    if load_dicom_prescription_dose:
        if len(plan_dcm) != 1:
            raise FileNotFoundError("There should be exactly one prescription dose dicom file that starts with RP or PL in the directory")
        prescription_dose = plan_dcm[0]
    else:
        prescription_dose = kwargs.pop("prescription_dose", None)

    simulation_setup = kwargs.pop("simulation_setup", None)
    new_sim_setup = deepcopy(simulation_setup) # this is to avoid memory reference issues during forloops
    if new_sim_setup is None:
        new_sim_setup = plan_dcm[0]
    if isinstance(new_sim_setup, dict):
        if new_sim_setup.get("brachy_source") is None:
            new_sim_setup["brachy_source"] = brachy_source

    return BrachyPlan(
        phantom=dir_dicom,
        catheter_table=catheter_table,
        combined_dose=combined_dose,
        simulation_setup=new_sim_setup,
        prescription_dose=prescription_dose,
        **kwargs
        )
