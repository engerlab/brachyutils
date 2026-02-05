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

from brachyutils.dose.dose_utils import BrachyDose

# from brachyutils.egsphant_utils import BrachyEgsphant
from brachyutils.geometry.applicator_utils import BrachyApplicator 
from brachyutils.geometry.phantom_utils import BrachyPhantom
from brachyutils.geometry.catheter_utils.catheter_table import Catheter, CatheterTable
from brachyutils.planning.structure_utils import BrachyStructure
from brachyutils.planning.simulation_utils import BrachySimulation
# from brachyutils.types import Optimization_Config
from brachyutils.planning.optimization.optim_utils import Optimization_Config

from pydantic import BaseModel, ConfigDict, Field, model_validator, computed_field

class ExportConfig_Dose(BaseModel):
    """
    Configuration for exporting dose data from the plan.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True  # Enables auto-docs from Field desc [web:48]
    )
    dir_export: str | Path = Field(None, description="Directory where dose files are exported.")
    name_combined: str = Field("combined", description="File name for combined dose output.")
    file_extension: Literal[".seq.nrrd", ".3ddose"] = Field(
        ".seq.nrrd", description="Allowed file extensions for dose files."
    )
    write_dose_rate_maps: bool = Field(
        False, description="Whether to write individual dose rate maps to files."
    )
    multi_processing: bool = Field(
        True, description="Enable multiprocessing for export (yes/no toggle)."
    )
    @computed_field
    def pth_combined(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.name_combined+self.file_extension)

class ExportConfig_PlanFile(BaseModel):
    """
    Configuration for exporting .plan files from the plan.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True  # Enables auto-docs from Field desc [web:48]
    )
    dir_export: str | Path = Field(None, description="Directory where the plan files are exported.")
    combined_only:bool = Field(True, description="If true, only combined plan is written. \
Per dwell position plan is generated.")
    name_combined:str = Field("combined", description="The name of the file for combined plan")
    file_extension: Literal[".plan"] = Field(".plan", description="File extension for plan files.")
    @computed_field
    def pth_combined(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.name_combined+self.file_extension)

class ExportConfig_MacFile(BaseModel):
    """
    Configuration for exporting .mac files from the plan.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_attribute_docstrings=True  # Enables auto-docs from Field desc [web:48]
    )
    dir_export: str | Path = Field(None, description="Directory where Mac files are exported.")
    combined_only:bool = Field(True, description="If true, only combined mac is written. \
Per dwell position plan is generated.")
    name_combined:str = Field("combined", description="The name of the file for combined mac")
    file_extension: Literal[".mac"] = Field(".mac", description="File extension for mac files.")
    body_name_stl: str = Field("BODY", description="Name of the body structure to be saved as a separate STL.")
    @computed_field
    def pth_combined(self)->Path:
        return self.dir_export/(self.name_combined+self.file_extension)
    @computed_field    
    def pth_body_stl(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.body_name_stl+".stl")

class ExportConfig_Egsphant(BaseModel):
    r"""
    The Export info needed for exporting Egsphant files.
    If using Monte Carlo simulations from RapidBrachyMC, It is recommended that
    the user crop the egsphant to a small region around the relevant anatomy and
    use provide the body_name_stl to save the body structure as a separate STL file. 
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_export: str | Path = Field(None, description="Directory where Egsphant file is exported.")
    name: str = Field("egsphant", description="File name for Egsphant output.")
    file_extension: Literal[".seq.nrrd", ".egsphant"] = Field(
        ".seq.nrrd", 
        description="Allowed file extensions for Egsphant files.")
    material_dict: dict | Path = Field(
        Path("admin/constants/structure_materials_prostate.json"),
        description="Dictionary of material names and their properties.")
    assign_material_from_ct: bool = Field(False, description="Whether to assign materials from CT data or based on contours.")
    crop_by_contour: str = Field(None, description="Name of the contour to crop by.")
    resampled_spacing: List[float] = Field(None, description="Spacing for resampling the phantom.")
    resampled_origin: List[float] = Field(None, description="Origin for resampling the phantom.")
    background_material: str = Field(None, description="Material name for background.")
    strict_name_match: bool = Field(True, description="Whether to enforce strict name matching for materials.")
    body_name_stl: str = Field(None, description="Name of the body structure to be saved as a separate STL.")
    @computed_field
    def pth_egsphant(self)->Path:
        return self.dir_export/(self.name+self.file_extension)
    @computed_field
    def pth_body_stl(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.body_name_stl+".stl")

class ExportConfig_CatheterTable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_export: str | Path = Field(None, description="Directory where catheter table is exported.")
    name: str = Field("catheter_table", description="File name for catheter table output.")
    file_extension: Literal[".json", ".mrk.json"] = Field(
        ".mrk.json", description="File extension for catheter table export.)")
    remove_text: bool = Field(True, description="Text to remove from dwell names.")
    one_markup_per_catheter: bool = Field(False, description="Whether to create one markup per catheter.")
    @computed_field
    def pth_catheter_table(self)->Path:
        self.dir_export = Path(self.dir_export)
        return self.dir_export/(self.name+self.file_extension)

# TODO: in future, add these export configs if neeeded
# class ExportConfig_Applicator(BaseModel):
#     model_config = ConfigDict(arbitrary_types_allowed=True)
# class ExportConfig_BrachyStructure(BaseModel):
#     model_config = ConfigDict(arbitrary_types_allowed=True)

class ExportConfig_BrachyPlan(BaseModel):
    r"""
    ### Purpose:
    - Configuration for exporting various components of a brachytherapy treatment plan.
    The components are catheter table, dose, egsphant, plan file, and mac file.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dir_export:str | Path = Field(..., description="Base directory where all plan components are exported.")
    export_config_dose: ExportConfig_Dose | bool = Field(False, description="Configuration for exporting dose data.")
    export_config_cathetertable: ExportConfig_CatheterTable | bool = Field(False, description="Configuration for exporting catheter table.")
    export_config_egsphant: ExportConfig_Egsphant | bool = Field(False, description="Configuration for exporting egsphant file.")
    export_config_planfile: ExportConfig_PlanFile | bool = Field(False, description="Configuration for exporting plan file.")
    export_config_macfile: ExportConfig_MacFile | bool = Field(False, description="Configuration for exporting mac file.")
    # TODO: in future, add these export configs if neeeded
    # export_config_applicator: ExportConfig_Applicator = None
    # export_config_phantom: ExportConfig_BrachyStructure = None
    applicator_geometry: bool = Field(False, description="Whether to export applicator geometry into a stl file.")
    structure_set: bool = Field(False, description="Whether to export structure set info into a json file.")

    @model_validator(mode="before")
    def validate_inputs(data):
        for k, v in data.items():
            if k.startswith("export_config_") and isinstance(v, bool):
                data[k] = {} if v else False
        return data

    @model_validator(mode="after")
    def validate_config(self):
        # make sure that the paths of dir exports are 
        # set correctly for all the inner attributes
        for _, value in self:                
            if isinstance(value, BaseModel):
                if value.dir_export is None:
                    value.dir_export = self.dir_export
        return self

class BrachyPlan:
    r"""
    ### Purpose:
    - This class holds the information regarding the brachytherapy treatment plan
    as well as all the functions to support the necessary plan operations.

    ### Attributes:
    #### Geometry and Structure Attributes:
    - phantom (BrachyPhantom): A BrachyPhantom object containing the patient geometry and structures.
    - structure_list (List[BrachyStructure]): A list of BrachyStructure objects containing the patient structures.
    - body_contour (ROIContour): The body contour of the patient.
    - phantom_origin (list): The origin of the phantom in the patient coordinate system.
    - organ_bounds (list): Min and max coordinates of the patient organs on each axis.
    - dvh_metric_goals (dict): Dictionary containing the DVH metric goals for the plan.
    - dvh_metrics_observed (dict): Dictionary containing the observed DVH metrics for the plan.
    - prescription_dose (float): The dose prescribed to the target volume.

    #### Catheter and Dwell Position Attributes:
    - catheter_table (CatheterTable): A catheter table object containing the catheter information.
    - num_catheters (int): The number of catheters in the plan.
    - catheter_numbers (list): The catheter ID numbers for each catheter in the catheter table.
    - num_dwells (int): The total number of dwell positions along all catheters in the plan.
    - dwell_numbers (list): The dwell number ID of each dwell position in the plan.
    - dwell_times (List[float]): The dwell time for each dwell position in the plan.
    - dwell_coordinates (List[list]): The coordinates of each dwell position in patient coordinates.

    #### Applicator Attributes:
    - applicator_list (List[BrachyApplicator]): The list of all applicators in the plan.
    - applicator_rotation_axis (np.array): The rotation axis of applicators (default: [0, 0, 1]).
    - applicator_rotation_origin (np.array): The rotation origin of applicators (default: [0, 0, 0]).

    #### Dose Attributes:
    - dose_rate_dict (defaultdict[BrachyDose]): Dictionary holding 3D dose rate maps for each dwell position.
    - combined_dose (BrachyDose): Sum of the dose rate maps weighted by the dwell times.

    #### Simulation and Optimization Attributes:
    - simulation_setup (BrachySimulation): A simulation setup object containing source info and simulation parameters.
    - optimization_config_list (List[Optimization_Config]): List of optimization configurations for the plan.
    """

    def __init__(
        self,
        #### for geometry definition:
        phantom: Union[Path, BrachyPhantom, dict] = None,
        #### for structure creation:
        dvh_metric_goals: Union[dict, Path] = None,
        prescription_dose: float = None,
        strict_name_match: bool = True,
        #### for loading catheter table and/or applicators:
        catheter_table: Union[Path, CatheterTable, str] = None,
        from_delivered_dwellpositions: bool = False,
        applicator_pth_list: Union[Path, str, list] = None,
        applicator_format: Literal["RapidBrachy", "WebApp"] = None,
        #### for loading dose or uncertainty:
        combined_dose: Union[Path, str, BrachyDose] = None,
        dir_dose_rate: Path = None,
        load_uncertainty:bool=False,
        multi_processing: bool = False,
        combined_dose_only: bool = False,
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
        - dvh_metric_goals:dict|Path := Dictionary containing the DVH metric goals or the path to its json file. Look at BrachyStructure for more info.
        The phantom should be loaded with structures for the Brachy stuctures to be created.
        - prescription_dose: float = None := The dose that is prescribed to the target volume. This is used to calculate the DVH metrics. 

        #### for loading catheter table and applicators:
        - catheter_table: Path | CatheterTable := A catheter table object or the path to a json file containing the information of the catheter table.
        from_delivered_dwellpositions: bool = True := If true, only the subset of dwell positions that had
        none zero dwell times in the DICOM plan file will be loaded. If false, all the dwell positions
        from the digitization points will be loaded.
        - applicator_pth_list := The list of applicator paths or the path to the json file containing the list. see load_applicator_list() for more info.
        - applicator_format:str = "RapidBrachy" := the format of the applicator list (default is "RapidBrachy"). See load_applicator_list() for more info.

        #### for loading dose rates or uncertainty maps per dwell position:
        - combined_dose: Path|BrachyDose := the path to the combined dose file or a BrachyDose object.
        - dir_dose_rate:str := path to the directory containing the dose rate files for a patient.
        - type_dose_file:str = ".nrrd" := the type of dose file to load (default is ".nrrd").
        - load_dose_or_uncertainty:str = "dose" := specify whether to load "dose" or "uncertainty" or "both" (default is "dose").
        - multi_processing:bool = False := flag to enable multi-processing for loading dose or uncertainty (default is False).
        - combined_dose_only:bool = False := flag to keep only the combined dose in memory after loading (default is False).

        #### Keywords Arguments:
        - dwells_near_ptv: bool = True := if True, will remove the dwell positions that are outside PTV
        with a margine of 10 mm.
        - add_hotspots_to_phantom: bool = False := if True, will add hotspot structures to the phantom.
        this is good for debugging, but slows down the plan creation process.
        - one_hotspot_structure: bool: = True := if False, will create separate hotspot structures 
        XXX simplify the constructor inputs by using only kwargs for optional inputs?

        #### for simulation setup:
        - simulation_setup = None := dictionary containing the simulation setup,

        ### Outputs:
            - Void := will initialize the BrachyPlan object
        """
        # declare the attributes
        # patient origin is used as a reference point for the catheter table,
        # the dwell coordinates, image origin, egsphant, and the dose objects.
        # XXX: figure out how to sort out patient origin to match all above.

        # phantom and geometry attributes
        self.phantom: BrachyPhantom = None
        self.dvh_metric_goals: dict = None
        self.dvh_metrics_observed: dict = None
        self.structure_list: List[BrachyStructure] = []
        self.body_contour: ROIContour = None
        self.phantom_origin: list = None  # np.array([0, 0, 0])  # x,y,z
        self.organ_bounds: list = None

        # catheter table attributes
        self.catheter_table: CatheterTable = None
        self.num_catheters: int = None
        self.catheter_numbers:list = np.array([], dtype=int)  # shape: (num_catheters, 1)
        self.num_dwells: int = None
        self.dwell_numbers: list = np.array([], dtype=int)  # shape: (num_dwells, 1)
        self.dwell_times: List[float] = np.array([], dtype=np.float32)  # shape: (num_dwells, 1)
        self.dwell_coordinates: List[list] = []  # shape: (num_dwells, 3)

        # applicator attributes
        self.applicator_list: List[BrachyApplicator] = []
        # XXX: figure out if the two below are dwell or applicator attributes?
        # they are dwell attributes that are impacted by applicator rotation. for now, leave them be.
        self.applicator_rotation_axis: np.array = np.array([0, 0, 1])  # x,y,z
        self.applicator_rotation_origin: float = np.array([0, 0, 0])  # x,y,z

        # dose attributes
        self.dose_rate_dict = defaultdict(BrachyDose)
        self.combined_dose: BrachyDose = None

        # simulation attributes
        self.simulation_setup: BrachySimulation = None

        ## fill the attributes depending on the inputs to the constructor
        # set the dvh metric goals if provided
        self.prescription_dose = prescription_dose
        if dvh_metric_goals is not None:
            if self.prescription_dose is None:
                raise ValueError("prescription dose is not provided. Please provide it.")
            self.set_dvh_metric_goals(dvh_metric_goals)

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
        if self.phantom is not None and self.dvh_metric_goals is not None:
            self.create_brachy_structure_set(
                phantom=self.phantom,
                dvh_metric_goals=self.dvh_metric_goals,
                strict_name_match=strict_name_match,
            )

        # load the catheter table if the path is provided
        if catheter_table is not None:
            if isinstance(catheter_table, (str, Path)):
                self.catheter_table = CatheterTable(
                    catheter_list=catheter_table,
                    from_delivered_dwellpositions=from_delivered_dwellpositions,
                    )
            elif isinstance(catheter_table, CatheterTable):
                self.catheter_table = catheter_table
            else:
                raise ValueError(
                    "catheter_table should be a path or a CatheterTable object"
                )
            if kwargs.get("dwells_near_ptv", True):
                for structure in self.structure_list:
                    if structure.is_target:
                        if isinstance(structure.mask, ROIContour):
                            mask = structure.mask.getBinaryMask(
                                origin=self.phantom.image_obj.origin,
                                gridSize=self.phantom.image_obj.gridSize,
                                spacing=self.phantom.image_obj.spacing,
                            )
                        self.catheter_table.remove_outside_mask(
                            mask=mask,
                            margin_mm=5.0,
                        )

            self.update_plan_from_catheter_table()

        # load the dose rate dict if the path is provided
        if dir_dose_rate is not None and combined_dose is None:
            self.load_dose_rate_dict(
                dir_dose_rate=dir_dose_rate,
                load_uncertainty=load_uncertainty,
                multi_processing=multi_processing,
                combined_dose_only=combined_dose_only,
            )
        elif dir_dose_rate is None and combined_dose is not None:
            if isinstance(combined_dose, BrachyDose):
                self.combined_dose = combined_dose
            elif isinstance(combined_dose, Path) or isinstance(combined_dose, str):
                self.combined_dose = BrachyDose(Path(combined_dose))
        elif dir_dose_rate is not None and combined_dose is not None:
            raise ValueError(
                "invalid input. Please provide either dir_dose_rate or combined_dose but not both"
            )
        else:
            warnings.warn("no dose rate is loaded", stacklevel=2)

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
                        total_time=np.sum(self.dwell_times))

        # load the applicator list if the path is provided
        if applicator_pth_list is not None and applicator_format is not None:
            self.load_applicator_list(applicator_pth_list, applicator_format)

        # # setup optimization
        if optimization_config_list is not None:
            self.optimization_config_list = optimization_config_list
            self.setup_optimization(
                self.optimization_config_list,
                self.structure_list,
                add_hotspots_to_phantom=kwargs.get("add_hotspots_to_phantom", False),
                one_hotspot_structure=kwargs.get("one_hotspot_structure", True),
            )

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
        - Void := will update the BrachyPlan.phantom attribute
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

    def update_plan_from_catheter_table(self):
        r"""
        ### Purpose:
        - To extract the dwell numbers, times, and coordinates from the catheter table
        and save them as class attributes.
        ### Inputs:
        - self := the BrachyPlan object
        ### Outputs:
        - Void := will update the self.dwell_numbers, self.dwell_times,
        and self.dwell_coordinates attributes
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
        for catheter in self.catheter_table.catheter_list:
            self.catheter_numbers = np.append(self.catheter_numbers, catheter.index)
            for dwell in catheter.dwells:
                self.dwell_numbers = np.append(self.dwell_numbers, dwell_counter)
                self.dwell_times = np.append(self.dwell_times, dwell.time)
                self.dwell_coordinates.append(
                    {
                        "angle": dwell.angle,
                        "position": dwell.position,
                        "rotation": dwell.rotation,
                        "relativePos": dwell.relativePos,
                        "catheter_index": catheter.index,
                        "dwell_index": dwell.index,
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
        if any(self.dose_rate_dict):
            self._calculate_combined_dose()

    def _update_catheter_table_from_plan(self):
        r"""
        ### Purpose:
        - Assuming that the dwell times or coordinates have changed, we need to update
        the catheter_table attribute to match the plan.
        ### Inputs:
        - self := the BrachyPlan object
        ### Outputs:
        - Void := will update the self.catheter_table attribute
        """
        raise ValueError("This function is deprecated and never used anyways")
        assert self.dwell_numbers.size != 0, "dwell numbers are not extracted"
        assert self.dwell_times.size != 0, "dwell times are not extracted"
        assert len(self.dwell_coordinates) != 0, "dwell coordinates are not extracted"
        assert self.num_dwells is not None, "number of dwells is not extracted"

        new_catheter_table = []

        for catheter_i in self.catheter_numbers:
            catheter = {}
            catheter["index"] = int(catheter_i)
            catheter["points"] = []
            catheter["dwells"] = []
            dwell = {}
            for dwell_i in self.dwell_numbers:
                if self.dwell_coordinates[dwell_i - 1]["catheter_index"] != catheter_i:
                    continue
                dwell["index"] = int(dwell_i)
                dwell["angle"] = float(self.dwell_coordinates[dwell_i - 1]["angle"])
                dwell["position"] = {
                    "x": float(self.dwell_coordinates[dwell_i - 1]["position"][0]),
                    "y": float(self.dwell_coordinates[dwell_i - 1]["position"][1]),
                    "z": float(self.dwell_coordinates[dwell_i - 1]["position"][2]),
                }
                dwell["relativePos"] = float(
                    self.dwell_coordinates[dwell_i - 1]["relativePos"]
                )
                dwell["rotation"] = {
                    "x": float(self.dwell_coordinates[dwell_i - 1]["rotation"][0]),
                    "y": float(self.dwell_coordinates[dwell_i - 1]["rotation"][1]),
                    "z": float(self.dwell_coordinates[dwell_i - 1]["rotation"][2]),
                }
                dwell["time"] = float(self.dwell_times[dwell_i - 1].item())
                dwell["weight"] = float(
                    (self.dwell_times[dwell_i - 1] / np.sum(self.dwell_times)).item()
                )
                catheter["dwells"].append(deepcopy(dwell))

            new_catheter_table.append(deepcopy(catheter))
        self.catheter_table = CatheterTable(catheter_list=new_catheter_table)

    def _update_dose_after_change_in_plan(self):
        r"""
        ### Purpose:
        - Assuming that the dwell times or coordinates have changed, we need to update
        the catheter_table attribute and the combined dose to match the plan.
        ### Inputs:
        - self := the BrachyPlan object
        ### Outputs:
        - Void := will update the BrachyPlan.catheter_table and BrachyPlan.combined_dose
        attributes
        """
        raise ValueError("This function is also deprecated")
        self._update_catheter_table_from_plan()
        self._calculate_combined_dose()

    def load_dose_rate_dict(
        self,
        dir_dose_rate: str| Path,
        load_uncertainty:bool=False,
        multi_processing: bool = False,
        combined_dose_only: bool = False,
    ):
        r"""
        ### Purpose:
        - To load the dose rates into the BrachyPlan object given a folder with
        patient's dose rate files and the catheter table loaded into the BrachyPlan object.
        In addition, combined dose is calculated as a linear combination of the dose rates
        and dwell times.
        ### Inputs:
        - `dir_dose_rate` :=  path to the directory containing the dose rate files. we assume
        that the name of the dose rate files end as "run_X_X_X.seq.nrrd", "run_X_X_X.seq.nrrd", etc.
        where the X corresponds to the catheter index+1, dwell index+1, and angle in increasing order.
        - `load_uncertainty`:= If true, uncertainty is loaded from the dose file, else it'll be set to 1. 
        - `multi_processing` := if True, the dose rate files will be loaded in parallel. By default,
        we use 8 cores for parallel processing.
        - `combined_dose_only`:bool = False := flag to keep only the combined dose in memory after loading.
        ### Outputs:
        - Void := will update the BrachyPlan.dose_rate_dict attribute
        """
        # make sure catheter table is loaded
        assert self.catheter_table is not None, "catheter table is not loaded"
        # assert self.dwell_numbers.size != 0, "dwell numbers are not extracted"
        # assert self.dwell_times.size != 0, "dwell times are not extracted"
        # assert len(self.dwell_coordinates) != 0, "dwell coordinates are not extracted"
        # assert self.num_dwells is not None, "number of dwells is not extracted"

        pth_dose_rate = Path(dir_dose_rate).resolve()
        if not pth_dose_rate.exists():
            raise ValueError(f"directory of dose rates does not exist: {pth_dose_rate}")
        dose_rate_files = list(pth_dose_rate.glob("run_*.seq.nrrd"))

        new_dose_rate_files = []
        # load file if they have not been loaded since modification
        for pth in dose_rate_files:
            if not self.dose_rate_dict.get(pth.name, None):
                new_dose_rate_files.append(pth)
            elif Path.stat().st_mtime != self.dose_rate_dict.get(pth.name).modification_time:
                new_dose_rate_files.append(pth)
            else:
                continue

        if multi_processing:       
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(_load_single_dose_rate, pth, load_uncertainty): pth
                    for pth in new_dose_rate_files
                    }
                for action in tqdm(
                    as_completed(futures),
                    desc="Loading dose rate maps",
                    total=len(new_dose_rate_files)):
                    try:
                        dose_rate = action.result()
                        self.dose_rate_dict[dose_rate.path.name] = dose_rate
                    except:
                        failed_path = futures[action]
                        raise ValueError(f"Failed loading f{failed_path}")
        else:
            for pth in tqdm(
                new_dose_rate_files,
                desc="Loading dose rate maps",
                total=len(new_dose_rate_files)):
                dose_rate = _load_single_dose_rate(
                    pth_dose_rate=pth,
                    load_uncertainty=load_uncertainty)
                self.dose_rate_dict[dose_rate.path.name] = dose_rate

        # now sort dose rates according to increasing catheter name and shield numbers
        # Fastest and most compact version
        sorted_items = sorted(
            self.dose_rate_dict.items(),
            key=lambda f: tuple(map(int, f[0].removeprefix('run_').removesuffix('.seq.nrrd').split('_')))
        )
        self.dose_rate_dict = defaultdict(BrachyDose, sorted_items)

        self._calculate_combined_dose()
        if load_uncertainty:
            self._calculate_combined_uncertainty()
        if combined_dose_only:
            del self.dose_rate_dict

    def _calculate_combined_dose(self):
        """
        ### Purpose:
        - To calculate the combined dose by multiplying the dose rates with the dwell times.
        The result is stored in the combined_dose attribute.
        We require strict name matching between the dwell names and dose rate names!
        ### Inputs:
        - None: but it needs the following attributes to be filled:
        - self.dose_rate_dict
        - self.catheter_table
        ### Raises:
            AssertionError: If the dose rate tensor or dwell times array is empty.
        """
        if not any(self.dose_rate_dict):
            raise ValueError("dose rate tensor is empty. Run load_dose_rate_dict()")

        self.combined_dose = BrachyDose.dose_with_empty_grid_like(
            list(self.dose_rate_dict.values())[0])
        for catheter in self.catheter_table:
            for dwell in catheter.dwells:
                self.combined_dose.dose_image.imageArray += self.dose_rate_dict.get(
                    f"run_{catheter.index+1}_{dwell.index+1}_{int(dwell.angle)}.seq.nrrd"
                    ).dose_image.imageArray * dwell.time

    def set_dvh_metric_goals(self, dvh_metric_goals: Union[dict, Path]):
        r"""
        ### Purpose:
        - To set the dvh metric list of the BrachyPlan object.
        ### Inputs:
        - dvh_metric_goals := a list of dictionaries. each dictionary contains the keys:
        "structure_name", "clinical_goal", "observed_value", and "penalty_weight"
        ### Outputs:
            - Void := will update the BrachyPlan.dvh_metric_goals attribute
        """
        if isinstance(dvh_metric_goals, Path):
            with open(dvh_metric_goals, "r") as json_file:
                dvh_metric_goals = json.load(json_file)
        self.dvh_metric_goals = dvh_metric_goals

    def create_brachy_structure_set(
        self,
        phantom: BrachyPhantom,
        dvh_metric_goals: dict,
        mask_type: Union[ROIContour, ROIMask] = ROIContour,
        strict_name_match: bool = True,
    ):
        r"""
        ### Purpose:
        - To create a list of BrachyStructure objects from the structures in the phantom and
        the DVH metric goals. Each BrachyStructure object will have attributes for the structure
        contour, the DVH and uncertainty volume histograms, optimization attributes, and simulation attributes.
        ### Inputes:
        - self.phantom := the phantom with its structures fully loaded.
        - self.dvh_metric_goals := the dvh metric goals dictionary
        ### Outputs:
        - Void := will update the BrachyPlan.structure_list attribute
        ### Dependencies:
        - BrachyDicom
        """
        self.structure_list = []
        structure_names_in_dvh = list(set([ #list of the structure names
            x.split("(")[-1].split(")")[0] for x in dvh_metric_goals.keys()
        ]))
        #separate dvh metric goals into separate dictionaries by structure
        dvh_metric_goals_by_structure = {}
        for structure_name in structure_names_in_dvh:
            dvh_metric_goals_per_struct = {
                key: value
                for key, value in dvh_metric_goals.items()
                if structure_name in key
            }
            dvh_metric_goals_by_structure[structure_name] = dvh_metric_goals_per_struct
        if phantom.cached_structure_masks is not None:
            structure_masks = deepcopy(phantom.cached_structure_masks)
            for k in list(structure_masks.keys()):
                if k not in structure_names_in_dvh:
                    structure_masks.pop(k)
        else:
            structure_masks: dict = phantom.get_structure_mask(
                structure_names_in_dvh, mask_type, strict_name_match=strict_name_match
            )

        for structure_name in structure_masks.keys():
            structure_obj = BrachyStructure(
                name=structure_name,
                mask=structure_masks[structure_name],
                is_target=True if (
                    "ctv" in structure_name.lower()
                    or "ptv" in structure_name.lower())  else False,
                in_dvh=True,
                dvh_metric_goals=dvh_metric_goals_by_structure[structure_name],
            )
            self.structure_list.append(structure_obj)
        if phantom.cached_structure_masks is not None:
            body_key = None
            for k in phantom.cached_structure_masks.keys():
                if k.lower() == "body":
                    body_key = k
            self.body_contour = phantom.cached_structure_masks.get(body_key, None)
        else:
            self.body_contour = phantom.get_structure_mask(
                ["body"], ROIContour, strict_name_match=False
            ).get("body", None)

    def load_applicator_list(
        self,
        applicator_list_pth: Union[list, Path, str],
        format: str = "WebApp",
    ):
        r"""
        ### Purpose:
        - To load the applicator list from a json file containing the applicator geometry.
        ### Inputs:
        - applicator_list_pth:str := path to the json file containing the applicator list with N applicators.
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
        ### Outputs:
            - Void := will update the BrachyPlan.applicator_list attribute
        """
        if isinstance(applicator_list_pth, Path) or isinstance(
            applicator_list_pth, str
        ):
            with open(applicator_list_pth, "r") as json_file:
                applicator_list = json.load(json_file)
        if format == "RapidBrachy":
            num_applicators = len(applicator_list["densities"])

            for i in range(num_applicators):

                j = i + 1 if i >= 1 else ""
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
                    normal=shieldNormal,
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
                    normal=applicator["normal"] if "normal" in applicator else None,
                    catheter_trajectory=(
                        applicator["catheter_trajectory"]
                        if "catheter_trajectory" in applicator
                        else None
                    ),
                )
                self.applicator_list.append(applicator_obj)
        else:
            raise ValueError("format should be either 'RapidBrachy' or 'WebApp'")

    def _calculate_combined_uncertainty(self):
        r"""
        ### Purpose:
        - To calculate the combined uncertainty of the combined dose map based on the
        dose rate dictionary and dwell times.
        We require strict name matching between the dwell names and the name of dose rate files
        ### Inputs:
        - self := the BrachyPlan object
        ### Outputs:
        - Void := will update the BrachyPlan.combined_dose.uncertainty attribute
        """
        assert self.combined_dose is not None, "combined dose is not calculated yet"
        if not any(self.dose_rate_dict):
            raise ValueError("dose rate tensor is empty. Run load_dose_rate_dict()")

        treatment_time = self.catheter_table.treatment_time
        for catheter in self.catheter_table:
            for dwell in catheter.dwells:
                self.combined_dose.uncertainty_image.imageArray += (self.dose_rate_dict.get(
                    f"run_{catheter.index+1}_{dwell.index+1}_{int(dwell.angle)}.seq.nrrd"
                    ).uncertainty_image.imageArray * (dwell.time/treatment_time)**2)
        self.combined_dose.uncertainty_image.imageArray = np.sqrt(
            self.combined_dose.uncertainty_image.imageArray)

    def get_dvh_metrics(
        self,
        combined_dose: BrachyDose=None,
        prescription_dose: float = None,
        return_percentage: bool = True,
        ):
        r"""
        ### Purpose:
        - To get the observed value of the dvh metric for each structure in the BrachyPlan.
        the observed value is calculated from the combined dose map.
        ### Inputs:
        - self := the BrachyPlan object
        ### Outputs:
        - Void := will update the BrachyStructure.dvh_metrics_observed attribute
        """
        assert self.structure_list is not None, "structure list is not created yet"
        assert self.prescription_dose is not None, "prescription dose is not set"
        if combined_dose is None:
            combined_dose = self.combined_dose
        if prescription_dose is None:
            prescription_dose = self.prescription_dose
        self.dvh_metrics_observed = {}
        for structure_obj in self.structure_list:
            if "hotspot_estimator" in structure_obj.name.lower():
                continue
            observed_metrics = structure_obj.get_dvh_metric(
                combined_dose,
                prescription_dose,
                return_percentage,
                self.body_contour,
                )
            self.dvh_metrics_observed.update(observed_metrics)
        return self.dvh_metrics_observed

    def export_dvh_metrics(self, output_pth: Union[str, Path]):
        r"""
        ### Purpose:
        - To export the dvh metrics of the BrachyPlan to a json file.
        ### Inputs:
        - output_pth := path to the output json file
        ### Outputs:
        - Void := will export the dvh metrics to a json file
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
        - Void := will export the dvh metric goals to a json file
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
        - Void := will update the BrachyStructure.uncertainty attribute
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
            - export_config_planfile (ExportConfig_PlanFile|None): Plan file export configuration.
            - export_config_macfile (ExportConfig_MacFile|None): Macro file export configuration.
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
            - Applicator geometry files (applicator_geometry.json and .mac files)
            - Structure set file (structure_set.json)

        ### Dependencies:
        - ExportConfig_BrachyPlan
        - export_dose()
        - export_catheter_table()
        - export_plan_files()
        - export_mac_files()
        - _export_egsphant()
        - _export_applicator_geometry()
        - _export_structure_set()
        """
        if isinstance(content_to_export, dict):
            content_to_export = ExportConfig_BrachyPlan(**content_to_export)
        dir_export = content_to_export.dir_export
        dir_export.mkdir(parents=True, exist_ok=True)

        if content_to_export.export_config_dose:
            self.export_dose(content_to_export.export_config_dose)

        if content_to_export.export_config_cathetertable:
            self.export_catheter_table(
                export_config_cathetertable=content_to_export.export_config_cathetertable,
                catheter_table=self.catheter_table,
            )

        if content_to_export.export_config_planfile:
            self.export_plan_files(
                export_config_planfile=content_to_export.export_config_planfile,
                catheter_table=self.catheter_table,
                )

        if content_to_export.export_config_macfile:
            self.export_mac_files(
                export_config_macfile=content_to_export.export_config_macfile,
                catheter_table=self.catheter_table
                )

        if content_to_export.export_config_egsphant:
            self._export_egsphant(
                export_config_egsphant=content_to_export.export_config_egsphant
            )

        if content_to_export.applicator_geometry:
            self._export_applicator_geometry(str(content_to_export.dir_export))

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
            catheter_table.write_json(
                export_config_cathetertable.dir_export
            )

    def export_dose(
        self,
        export_config_dose: ExportConfig_Dose
    ):
        r"""
        ### Purpose:
        - to export combined dose map and if needed the dose rate maps to a given directory.
        exporting dose rate maps is optional.
        ### Inputs:
        - export_config_dose: The dose export configuration. Look at ExportConfig_Dose for more info 
        ### Outputs:
        - None := will export the dose map into the specified export directory.
        ### Dependencies:
        - _write_single_dose_rate()
        - multiprocessing
        """
        assert self.combined_dose is not None, "combined dose is not calculated yet"
        dir_export = Path(export_config_dose.dir_export)
        # write combined dose
        self.combined_dose.write_brachydose_to_file(
            export_config_dose.pth_combined
        )

        if export_config_dose.write_dose_rate_maps:
            if export_config_dose.multi_processing:
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {
                        executor.submit(_write_single_dose_rate, self.dose_rate_dict.get(dose_rate_name), dir_export, export_config_dose.file_extension):
                            dose_rate_name for dose_rate_name in self.dose_rate_dict
                        }
                    for action in tqdm(as_completed(futures), desc="Writing dose rate maps"):
                        try:
                            action.result()
                        except:
                            failed_path = futures[action]
                            raise ValueError(f"Failed writing {failed_path}")
            else:
                for dose_rate in tqdm(self.dose_rate_dict, desc="Writing dose rate maps"):
                    _write_single_dose_rate(
                        dose_rate=self.dose_rate_dict.get(dose_rate),
                        dir_export=dir_export,
                        dose_extension=export_config_dose.file_extension)
        print(f"Dose exported to {dir_export}")

    def export_plan_files(
        self,
        export_config_planfile:ExportConfig_PlanFile,
        catheter_table:CatheterTable,
        ):
        r"""
        ### Purpose:
        - To export dwell positions and their normalized times into ".plan" text files in the
        format required by RapidBrachy.
        ### Inputs:
        - export_config_planfile:= The export configuration for the plan files. see ExportConfig_PlanFile
        - catheter_table:= The catheter table with the dwells.
        ### Outputs:
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
        ### Dependencies:
            - None
        """
        # total_dwell_time = np.sum(self.dwell_times)
        total_dwell_time = catheter_table.treatment_time
        num_dwells = catheter_table.num_dwell_positions
        combined_plan = "Treatment Plan\n"
        combined_plan += f"{num_dwells} Control Points\n"

        for cat in catheter_table:
            for dwell in cat.dwells:
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
                shield_angle = 0
                if not export_config_planfile.combined_only:
                    with open(export_config_planfile.dir_export + f"/dwell_{catheter_idx + 1}_{dwell_idx + 1}_{shield_angle}.plan", "w") as file:
                        file.write(run_i_plan)

        with open(export_config_planfile.pth_combined, "w") as file:
            file.write(combined_plan)
        print(".plan files were exported successfully")

    def export_mac_files(
        self,
        export_config_macfile: ExportConfig_MacFile,
        catheter_table:CatheterTable,
        ):
        r"""
        ### Purpose:
        - To export the simulation parameters of the plan into a macro files
        and run_{catheterNumber}_{dwellNumber}_{shieldAngle}.mac
        ### Inputs:
        - export_config_macfile:= The export configuration for macro files.
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

        with open(export_config_macfile.pth_combined, "w") as file:
            file.write(sim_obj.to_string())

        if not export_config_macfile.combined_only:
            for cat in catheter_table:
                for dwell in cat.dwells:
                    catheter_idx = cat.index
                    dwell_idx = dwell.index
                    # Not dealing with shield angle for now but the new convention for filename is
                    # xxx_catheter#_dwell#_shieldangle.plan
                    shield_angle = 0
                    sim_obj = deepcopy(self.simulation_setup)
                    order = f"{catheter_idx + 1}_{dwell_idx + 1}_{shield_angle}"
                    sim_obj.pth_plan = f"dwell_{order}.plan"
                    sim_obj.total_time = 1

                    with open(export_config_macfile.dir_export 
                            + f"/run_{order}.mac", "w") as file:
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
            resampled_spacing=export_config_egsphant.resampled_spacing,
            resampled_origin=export_config_egsphant.resampled_origin,
            background_material=export_config_egsphant.background_material,
            strict_name_match=export_config_egsphant.strict_name_match
        )
        if export_config_egsphant.body_name_stl is not None:
            body_mask = self.body_contour.getBinaryMask(
                origin=self.phantom.origin,
                spacing=self.phantom.spacing,
                gridSize=self.phantom.gridSize
            )
            self.phantom.mask_to_stl(
                roi_mask=self.body_contour,
                mask=body_mask,
                pth_output=export_config_egsphant.pth_body_stl
            )

        print("Egsphant file was exported successfully")

    def _export_applicator_geometry(
        self, dir_export: str, export_format: str = "RapidBrachy"
    ):
        r"""
        ### Purpose:
        - To export the applicator geometries either in the RapidBrachy Format (mac files and single json file)
        or in webapp format (json file).
        ### Inputs:
        - dir_export := path to the directory where the export happens
        - format := the format of the applicator geometry file. options are "RapidBrachy" or "WebApp"
        ### Outputs:
        - Void := will export the applicator geometries into the specified export directory.
        ### Dependencies:
        - None
        """
        if export_format == "RapidBrachy":

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
            counter = 0
            for applicator in self.applicator_list:

                out_json["densities"].append(applicator.density)
                out_json["filenames"].append(applicator.path)
                out_json["materials"].append(applicator.material)
                out_json["points"].append(
                    applicator.catheter_trajectory.flatten().tolist()
                )
                out_json["shieldNormalx"] = float(applicator.normal[0])
                out_json["shieldNormaly"] = float(applicator.normal[1])
                out_json["shieldNormalz"] = float(applicator.normal[2])

                subscript = counter + 1 if counter >= 1 else ""
                out_json[f"wRot{subscript}"] = float(applicator.rotation[0])
                out_json[f"xRot{subscript}"] = float(applicator.rotation[1])
                out_json[f"yRot{subscript}"] = float(applicator.rotation[2])
                out_json[f"zRot{subscript}"] = float(applicator.rotation[3])

                out_json["x"] = float(applicator.coordinates[0])
                out_json["y"] = float(applicator.coordinates[1])
                out_json["z"] = float(applicator.coordinates[2])
                counter += 1

            with open(dir_export + "/applicator_geometry.json", "w") as file:
                json.dump(out_json, file, indent=4)

        elif export_format == "WebApp":
            out_json = [
                applicator.to_dict(format) for applicator in self.applicator_list
            ]
            with open(dir_export + "/applicator_geometry.json", "w") as file:
                json.dump(out_json, file, indent=4)

        else:
            raise ValueError("format should be either 'RapidBrachy' or 'WebApp'")

        # export the mac files for each applicator
        for applicator in self.applicator_list:
            applicator.to_mac(os.path.join(dir_export, f"{applicator.name}.mac"))
        print("applicator geometry file was exported successfully")

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
        - void := self.structure_list is exported as a dictionary and
        written to structure_set.json
        ### Dependencies:
        """
        raise NotImplementedError("now that you are here, finish this function thank you!")
        structure_set = []
        for structure in self.structure_list:
            structure_set.append(structure.to_dict(export_format))

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
        - Void := will print the information of the plan
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
        one_hotspot_structure:bool=True
        ):
        r"""
        ### Purpose:
        - Given the optimization config list either as a list or in a json file, put each
        optimization config inside the BrachyStructures. Also, create the hotspot estimator
        structure if needed.
        """
        self._reset_optimization()
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
                if config.structure_name.lower() not in target_structure_names:
                    raise ValueError(
                        "penalty_weight_hotspot can only be set for PTV or CTV structures"
                    )
                self._create_hotspot_structures(
                    target_optim_config=config,
                    add_hotspots_to_phantom=add_hotspots_to_phantom,
                    one_hotspot_structure=one_hotspot_structure)
            for struc in structure_list:
                if config.structure_name.lower() == struc.name.lower():
                    assert config.is_target == struc.is_target, f"The target structure in plan and optimization \
config do not match for structure {struc.name}"
                    struc.set_optimization_config(config)
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
        self.update_plan_from_catheter_table()
        step_size = self.catheter_table.step_size
        # identify unique dwell pairs that are withi n the step size distance
        dwell_pairs = []
        def distance(pos1, pos2):
            return np.linalg.norm(pos1 - pos2)
        def center(pos1, pos2):
            return (pos1 + pos2) / 2

        for i in range(len(self.dwell_coordinates)):
            for j in range(i + 1, len(self.dwell_coordinates)):
                current_distance = distance(
                    np.array(self.dwell_coordinates[i]["position"]),
                    np.array(self.dwell_coordinates[j]["position"])) 
                if current_distance <= step_size:
                    dwell_pairs.append(
                        {
                            "dwell_pair": (
                                {
                                    "catheter":self.dwell_coordinates[i]["catheter_index"]+1,
                                    "dwell": self.dwell_coordinates[i]["dwell_index"]+1
                                },
                                {
                                    "catheter":self.dwell_coordinates[j]["catheter_index"]+1,
                                    "dwell": self.dwell_coordinates[j]["dwell_index"]+1
                                }),
                            "center": center(
                                np.array(self.dwell_coordinates[i]["position"]),
                                np.array(self.dwell_coordinates[j]["position"])
                            ),
                            "radius": step_size,
                            "distance": current_distance,
                            "inter-catheter": True if (
                                self.dwell_coordinates[i]["catheter_index"] 
                                != self.dwell_coordinates[j]["catheter_index"]
                                ) else False
                        }
                    )
        # create hotspot structures masks for each dwell pair
        hotspot_mask_list = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    _gen_hotspot_mask,
                    dwell_pair,
                    self.phantom.image_obj.gridSize,
                    self.phantom.image_obj.origin,
                    self.phantom.image_obj.spacing,
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
        TODO: get rid of +1 when moving towards catheter generation from digi points
        TODO: Consider adding angle to the name later when IMBT is involved.
        """
        dose_rates_catheter = defaultdict(BrachyDose)
        
        for name, dose_rate in self.dose_rate_dict.items():
            cath_num = name.split("_")[1]
            if catheter_index+1 == int(cath_num):
                dwell_num = name.split("_")[2]
                dose_rates_catheter[f"catheter_{cath_num}_dwell_{dwell_num}"] = dose_rate
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
        {}
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
            f"hotspot_estimator_catheter_{(dwellpair['dwell_pair'])[0]['catheter']}_dwell_{(dwellpair['dwell_pair'])[0]['dwell']}"
            + f"/catheter_{(dwellpair['dwell_pair'])[1]['catheter']}_dwell_{(dwellpair['dwell_pair'])[1]['dwell']}"
            ),
    )
    return BrachyStructure(
        name=dwell_mask.name,
        mask=dwell_mask,
        is_target=False,
        in_dvh=False,
    )

def _write_single_dose_rate(
    dose_rate:BrachyDose,
    dir_export: str | Path = None,
    dose_extension: str = None,
    file_name: str = None,
    ):
    r"""
    ### Purpose:
    to write out a single dose rate map and uncertainty to a directory.
    ### Inputs:
    - dose_rate:= The BrachyDose object for the dose rate data.
    - dir_export:= the directory to which the dose rate maps will be exported
    - file_name:= The name of the file inside dir_export. Following the RapidBrachy standard, it should be
    "run_{catheter.index+1}_{dwell.index+1}_{angle}.seq.nrrd". if none, dose_rate.path.name is used.
    - dose_extension := the type of dose rate map to be exported. options are ".3ddose", ".minidos", or ".nrrd"
    ### Output:
    - Void := dose file is written to dir_export+f"/{file_name}.{dose_type}
    """
    if file_name is None:
        file_name = dose_rate.path.name.split(".")[0]
    if dose_extension is None:
        dose_extension = ".seq.nrrd"
    dir_export = Path(dir_export)
    pth_out = dir_export/(file_name+dose_extension)
    dose_rate.write_brachydose_to_file(pth_dose_file=pth_out)

def _load_single_dose_rate(
    pth_dose_rate:Path,
    load_uncertainty=False
    )->BrachyDose:
        return BrachyDose(pth_dose_file=pth_dose_rate, load_uncertainty=load_uncertainty)

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
    load_dicom_plan: bool = True,
    **kwargs) -> BrachyPlan:
    r"""
    ### Purpose:
    - To load all the contents of a dicom directory into a BrachyPlan object.
    ### Inputs:
    - dir_dicom := the path to the dicom directory
    - load_dicom_dose := if True, the dose dicom file will be loaded
    - load_dicom_plan := if True, the plan dicom file will be loaded
    - **kwargs := additional arguments to be passed to the BrachyPlan constructor
    ### Outputs:
    - BrachyPlan := the BrachyPlan object with all the contents of the dicom directory
    """
    all_dicom_files = list(Path(dir_dicom).glob("*.dcm"))
    if len(all_dicom_files) == 0:
        raise FileNotFoundError("No dicom files found in the directory")
    # structure_dcm = [dcm for dcm in all_dicom_files if "RS" in dcm.name or "rs" in dcm.name]
    dose_dcm = []
    plan_dcm = []
    if load_dicom_dose:
        dose_dcm = [dcm for dcm in all_dicom_files if str(dcm.name).lower().startswith("rd")]
    if load_dicom_plan:
        plan_dcm = [
            dcm for dcm in all_dicom_files if
            (
                "rp" in str(dcm.name).lower()
                or "pl" in str(dcm.name).lower()
            )
        ]

    # structure_dcm = structure_dcm[0] if len(structure_dcm) > 0 else None
    dose_dcm = dose_dcm[0] if len(dose_dcm) > 0 else None
    plan_dcm = plan_dcm[0] if len(plan_dcm) > 0 else None
    simulation_setup = kwargs.pop("simulation_setup", None)

    new_sim_setup = deepcopy(simulation_setup) # this is to avoid memory reference issues during forloops
    if new_sim_setup is None:
        new_sim_setup = plan_dcm
    if isinstance(new_sim_setup, dict):
        if new_sim_setup.get("brachy_source") is None:
            new_sim_setup["brachy_source"] = plan_dcm
    combined_dose = kwargs.pop("combined_dose", dose_dcm)
    return BrachyPlan(
        phantom=dir_dicom,
        catheter_table=plan_dcm,
        combined_dose=combined_dose,
        simulation_setup=new_sim_setup,
        **kwargs
    )
