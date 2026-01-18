import gc
import json
import os

# import re
import warnings
from copy import deepcopy
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Literal, Union, Dict, Tuple

import numpy as np
from opentps.core.data import DVH, ROIContour
from opentps.core.data.images import ROIMask

from scipy import interpolate, ndimage

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

class BrachyPlan:
    r"""
    ### Purpose:
    - This class holds the information regarding the brachytherapy treatment plan
    as well as all the functions to support the necessary plan operations.

    ### Attributes:
    - phantom:= A BrachyPhantom object containing the patient geometry and structures.
    - dvh_metric_goals:= A dictionary containing the DVH metric goals for the plan.
    - dvh_metrics_observed:= A dictionary containing the observed DVH metrics for the plan.
    - structure_list:= A list of BrachyStructure objects containing the patient structures.
    - phantom_origin:= The origin of the phantom in the patient coordinate system.
    - organ_bounds:= A dictionary containing the min and max coordinates of the patient organs on each axis. 
    - catheter_table:= A catheter table object containing the catheter information.
    - num_catheters:= The number of catheters in the plan.
    - catheter_numbers:= The catheter id numbers for each catheter in the catheter table.
    - num_dwells:= The total number of dwell positions along all catheters in the plan.
    - dwell_numbers:= The dwell number id of each dwell position in the plan.
    - dwell_times:= The dwell time for each dwell position in the plan.
    - dwell_coordinates:= The coordinate of each dwell position in patient coordinates?
    - applicator_list:= The list of all the applicators in the plan.
    - applicator_rotation_axis:= The rotation axis of each applicator
    - applicator_rotation_origin:= The rotation origin of each applicator.
    - dose_rate_tensor:= a tensor holding 3D dose rate maps for each dwell position.
    - combined_dose:= sum of the dose rate maps weighted by the dwell times.
    - uncertainty_tensor:= sqaure root of the sum of the squares of the uncertainty maps weighted by the 
    dwell times normalized to the treatment time.
    - simulation_setup:= A simulation setup object containing the source info as well as simulation parameters.
    - prescription_dose:= The dose that is prescribed to the target volume.

    ### Functions:
    - update_plan_from_catheter_table()
    - _update_catheter_table_from_plan()
    - _update_dose_after_change_in_plan()
    - load_dose_rate_or_uncertainty_tensor()
    - _calculate_combined_dose()
    - set_dvh_metric_goals()
    - create_brachy_structure_set()
    - get_dvh_metrics()
    - _calculate_combined_uncertainty()
    - calculate_uncertainty_per_structure()
    - export_brachy_plan ()
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
        type_dose_file: Literal[".nrrd", ".3ddose"] = ".nrrd",
        load_dose_or_uncertainty: Literal["dose", "uncertainty", "both"] = "dose",
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
        self.dose_rate_dict = np.array(
            [], dtype=np.float32
        )  # shape: (num_dwells, z, y, x)
        self.combined_dose: BrachyDose = None
        self.uncertainty_tensor = np.array(
            [], dtype=np.float32
        )  # shape: (num_dwells, z, y, x)

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

        # load the dose rate tensor if the path is provided
        if dir_dose_rate is not None and combined_dose is None:
            self.load_dose_rate_or_uncertainty_tensor(
                dir_dose_rate=dir_dose_rate,
                type_dose_file=type_dose_file,
                load_dose_or_uncertainty=load_dose_or_uncertainty,
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
        if self.dose_rate_dict.any():
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
        self._update_catheter_table_from_plan()
        self._calculate_combined_dose()

    def load_dose_rate_or_uncertainty_tensor(
        self,
        dir_dose_rate: str,
        type_dose_file: Literal[".nrrd", ".3ddose"] = ".nrrd",
        load_dose_or_uncertainty: Literal["dose", "uncertainty", "both"] = "dose",
        multi_processing: bool = False,
        combined_dose_only: bool = False,
    ):
        r"""
        ### Purpose:
        - To load the dose rate tensor into the BrachyPlan object given a folder with
        patient's dose rate files and the catheter table loaded into the BrachyPlan object.
        In addition, combined dose is calculated as a linear combination of the dose rates
        and dwell times.
        ### Inputs:
        - dir_dose_rate :=  path to the directory containing the dose rate files. we assume
        that the name of the dose rate files end as "run_1.nrrd", "run_2.nrrd", etc.
        - type_dose_file := the type of dose rate file. The type could be ".nrrd" or ".3ddose"
        consult BrachyDose in dose_utils.py for more info on the dose rate file types.
        - load_dose_or_uncertainty := either "dose", "uncertainty", or "both"
        - multi_processing := if True, the dose rate files will be loaded in parallel. By default,
        we use 8 cores for parallel processing.
        - combined_dose_only:bool = False := flag to keep only the combined dose in memory after loading (default is False).
        ### Outputs:
        - Void := will update the BrachyPlan.dose_rate_dict attribute
        ### Dependencies:
        - glob
        - BrachyDose
        """
        # make sure catheter table is loaded
        assert self.catheter_table is not None, "catheter table is not loaded"
        assert self.dwell_numbers.size != 0, "dwell numbers are not extracted"
        assert self.dwell_times.size != 0, "dwell times are not extracted"
        assert len(self.dwell_coordinates) != 0, "dwell coordinates are not extracted"
        assert self.num_dwells is not None, "number of dwells is not extracted"


        def get_dwell_order(dose_rate_path):
            file_name = os.path.basename(dose_rate_path)
            return get_dwell_order_from_file_name(file_name)

        def get_dwell_order_from_file_name(file_name):
            """
            Files should have this format:
            run_{catheter#}_{Dwell#incatheter}_{shieldangle}.seq.nrrd
            Assuming that there are less than 10000 dwell positions per catheter
            We order based on 10000 * catheter# + Dwell#incatheter
            """
            x = file_name.split(".")[0][4:]
            catheter_nb, dwell_nb, shield_angle = x.split("_")
            return 10000 * int(catheter_nb) + int(dwell_nb)
        
        # here is the list of the dose rate files
        if isinstance(dir_dose_rate, str) or isinstance(dir_dose_rate, Path):
            dose_rate_files = glob(os.path.join(dir_dose_rate, f"run*{type_dose_file}"))
            dose_rate_files = [
                dosefile for dosefile in dose_rate_files if "combined" not in dosefile
            ]
            dose_rate_files.sort(
                key=lambda x: get_dwell_order(x)
            )

        else:
            assert isinstance(dir_dose_rate, dict) and isinstance(dir_dose_rate[list(dir_dose_rate.keys())[0]][0], np.ndarray), (
                "Expected a folder with dose rate files saved or a dictionary of tuples (numpy arrays, header info)."
            )
            dose_rate_files = dir_dose_rate
            sorted_dict = dict(sorted(dose_rate_files.items(), key=lambda item: get_dwell_order_from_file_name(item[0])))
            dose_rate_files = [x for x in sorted_dict.values()]

        

        assert (
            len(dose_rate_files) == self.num_dwells
        ), ("number of dose rate files does not match the number of dwell positions"
            f" in the catheter table. Expected {self.num_dwells} but found {len(dose_rate_files)} at {os.path.join(dir_dose_rate, f"run*{type_dose_file}")}"
        )

        test_dose_obj = BrachyDose(dose_rate_files[0])

        if load_dose_or_uncertainty not in ["dose", "uncertainty", "both"]:
            raise ValueError(
                "load_dose_or_uncertainty should be either 'dose', 'uncertainty', or 'both'"
            )

        # load the dose rate tensor
        if multi_processing:
            with Pool(processes=16 if cpu_count()>8 else 4) as pool:
                func = partial(
                    _load_single_dose_or_uncertainty_to_dict,
                    load_dose_or_uncertainty=load_dose_or_uncertainty,
                )
                dose_or_uncertainty_list = list(
                    tqdm(
                        pool.imap(func, dose_rate_files),
                        total=len(dose_rate_files),
                        desc="Loading dose rates...",
                    )
                )    

        else:
            # dose_or_uncertainty_list = np.empty(len(dose_rate_files), dtype=object)
            dose_or_uncertainty_list = [None] * len(dose_rate_files)
            for i, pth_dose_rate in tqdm(enumerate(dose_rate_files), total=len(dose_rate_files), desc="Loading dose rates..."):
                dose_or_uncertainty_list[i] = _load_single_dose_or_uncertainty_to_dict(
                    pth_dose_rate, load_dose_or_uncertainty
                )
            # print(dose_or_uncertainty_list.shape)

        if load_dose_or_uncertainty == "both":
            self.dose_rate_dict = np.array(
                dose_or_uncertainty_list, dtype=np.float32
            )[0, :]
            self.uncertainty_tensor = np.array(
                dose_or_uncertainty_list, dtype=np.float32
            )[1, :]
        elif load_dose_or_uncertainty == "dose":
            self.dose_rate_dict = np.array(dose_or_uncertainty_list, dtype=np.float32)
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

        self.combined_dose = BrachyDose.dose_with_empty_grid_like(test_dose_obj)

        if load_dose_or_uncertainty != "uncertainty":
            self._calculate_combined_dose()
        if load_dose_or_uncertainty != "dose":
            self._calculate_combined_uncertainty()
        # free up memory
        if combined_dose_only:
            self.dose_rate_dict = None
            self.uncertainty_tensor = None

        # if len(self.structure_list) != 0:
        #     for structure in self.structure_list:
        #         structure.mask = _resize_structure_mask(
        #             structure.mask, self.combined_dose.grid.shape
        #         )

    def _calculate_combined_dose(self):
        """
        ### Purpose:
        - To calculate the combined dose by multiplying the dose rate tensor with the dwell times array.
        The result is stored in the combined_dose attribute.

        ### Raises:
            AssertionError: If the dose rate tensor or dwell times array is empty.
        """
        assert (
            self.dose_rate_dict.size != 0
        ), "dose rate tensor is empty. Run load_dose_rate_or_uncertainty_tensor()"
        assert (
            self.dwell_times.size != 0
        ), "dwell times array is empty. Run update_plan_from_catheter_table()"

        # calculate the combined dose and store the result in the combined_dose attribute
        temp_dose_array = np.zeros_like(self.dose_rate_dict[0])
        for i in range(self.num_dwells):
            temp_dose_array += self.dose_rate_dict[i] * self.dwell_times[i]

        self.combined_dose.set_dose_array(temp_dose_array)

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
        - To calculate the combined uncertainty of the combined dose map.
        ### Inputs:
        - self := the BrachyPlan object
        ### Outputs:
        - Void := will update the BrachyPlan.combined_dose.uncertainty attribute
        """
        assert self.uncertainty_tensor is not None, "uncertainty tensor is not loaded"
        assert self.dwell_times is not None, "dwell times are not extracted"
        assert self.combined_dose is not None, "combined dose is not calculated yet"

        normalized_times = self.dwell_times / np.sum(self.dwell_times)

        uncertainty = np.zeros_like(self.combined_dose.get_dose_array())
        for i in range(self.num_dwells):
            uncertainty += (self.uncertainty_tensor[i] * normalized_times[i]) ** 2
        uncertainty = np.sqrt(uncertainty)
        self.combined_dose.set_uncertainty_array(uncertainty)

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
        dir_export: str | Path,
        content_to_export: Dict[str, bool | str] = None,
        export_format: str = "RapidBrachy"
    ):
        r"""
        ### Purpose:
        - To export the treatment plan file into a given export_format.
        The export_format can be either "RapidBrachy" or "WebApp".
        ### Inputs:
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
            - "dose_type":str := "nrrd", "minidos" or "3ddose",
            - "uncertainty", "dose rate maps",
            - "catheter_table", "plan", "mac", "egsphant",
            - "ApplicatorMaterials", applicator_geometry", "structure_set",
        ### Outputs:
            - Void := will export the available parts of a plan into the specified export_format.
        """
        dir_export = Path(dir_export)
        dir_export.mkdir(parents=True, exist_ok=True)
        if export_format == "WebApp":

            raise NotImplementedError("export to WebApp is not implemented yet")

        elif export_format == "RapidBrachy":

            if content_to_export.get("dose", False):
                self._export_dose(
                    dir_export=str(dir_export),
                    with_uncertainty=content_to_export.get("uncertainty", False),
                    dose_type=content_to_export.get("dose_type", ".seq.nrrd"),
                    dose_rate_maps=content_to_export.get("dose_rate_maps", False),
                )
                print("Dose exported successfully")
            if content_to_export.get("catheter_table", False):
                # assumes file name is "catheter_table.json"
                self._export_catheter_table(str(dir_export))
                print("Catheter Table exported successfully")

            if content_to_export.get("plan", False):
                # assumes file name is "dwell_#.plan"
                self._export_plan_file(
                    dir_export=str(dir_export),
                    combined_only=content_to_export.get("combined_only", True)
                    )
                print(".plan files were exported successfully")

            if content_to_export.get("mac", False):
                # assumes file name is "run_#.mac"
                self._export_dwell_mac_file(
                    dir_export=str(dir_export),
                    combined_only=content_to_export.get("combined_only", True)
                    )
                print(".mac files were exported successfully")

            if content_to_export.get("egsphant", False):
                # assumes file name is "ct.egsphant"
                self._export_egsphant(
                    dir_export=str(dir_export),
                    material_dict=content_to_export.get("materials_table", None),
                    assign_material_from_ct=content_to_export.get("assign_material_from_ct", True),
                    crop_by_contour=content_to_export.get("crop_by_contour", None),
                    strict_name_match=content_to_export.get("strict_name_match", True),
                    resampled_spacing=content_to_export.get("resampled_spacing", None),
                    resampled_origin=content_to_export.get("resampled_origin", None),
                    background_material=content_to_export.get("background_material", "Air"),
                )
                print("Egsphant file was exported successfully")

            if content_to_export.get("applicator_geometry", False):
                # assumes file name is "applicator_geometry.json"
                self._export_applicator_geometry(str(dir_export), export_format)
                print("applicator geometry file was exported successfully")

            if content_to_export.get("structure_set", False):
                # assumes file name is "structure_set.json"
                self._export_structure_set(
                    str(dir_export), content_to_export.get("materials_table", None)
                )
                print("structure set file was exported successfully")

        else:
            raise ValueError("export_format should be either 'RapidBrachy' or 'WebApp'")

    def _export_dose(
        self,
        dir_export: str,
        with_uncertainty=False,
        dose_type=".seq.nrrd",
        dose_rate_maps=False,
    ):
        r"""
        ### Purpose:
        - to export combined dose map with or without uncertainty in the provided export directory.
        exporting dose rate maps is optional.
        ### Inputs:
        - dir_export := the directory to which the dose map will be exported.
        - uncertainty := if True, the uncertainty map will be exported as well.
        - dose_type := the type of dose map to be exported. options are ".3ddose", ".minidos", or ".nrrd".
        - dose_rate_maps := if True, the dose rate maps will be exported as well.
        ### Outputs:
        - Void := will export the dose map into the specified export directory.
        ### Dependencies:
        - _export_single_dose_rate()
        - multiprocessing
        """
        assert self.combined_dose is not None, "combined dose is not calculated yet"
        # if uncertainty:
        self.combined_dose.write_brachydose_to_file(
            Path(dir_export) / f"combined{dose_type}"
        )

        if dose_rate_maps:
            if cpu_count() < 4:
                for i in self.dwell_numbers:
                    _export_single_dose_rate(
                        self.dose_rate_dict[i - 1],
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
                            self.dose_rate_dict,
                            self.dwell_numbers,
                            self.uncertainty_tensor,
                        )
                    ]
                else:
                    print("Exporting dose rate maps without uncertainty")
                    giant_export_list = [
                        (dose_grid, dwell_number)
                        for dose_grid, dwell_number in zip(
                            self.dose_rate_dict, self.dwell_numbers
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
        ### Purpose:
        - to export catheter table of the plan into a file called catheter_table.json
        inside dir_export.
        ### Inputs:
        - dir_export := path to the directory where the export happens
        ### Outputs:
        - void := self.catheter_table is written to catheter_table.json
        ### Dependencies:
        - json
        """
        file_path = dir_export + "/catheter_table.json"
        with open(file_path, "w") as file:
            json.dump(self.catheter_table.to_dict(), file, indent=4)

    def _export_plan_file(
        self,
        dir_export: str,
        combined_only:bool=True):
        r"""
        ### Purpose:
        - To export dwell positions and their normalized times into ".plan" text files in the
        format required by RapidBrachy.
        ### Inputs:
        - dir_export := path to the directory where the export happens
        - combined_only := if True, only the combined.plan file will be exported. if False,
        the individual dwell position files will also be exported.
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
        total_dwell_time = np.sum(self.dwell_times)
        combined_plan = "Treatment Plan\n"
        combined_plan += f"{self.num_dwells} Control Points\n"

        for dwell_i in range(self.num_dwells):
            dwell_coordinates_str = np.array(
                list(self.dwell_coordinates[dwell_i]["position"])
                + list(self.dwell_coordinates[dwell_i]["rotation"])
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

            catheter_idx = self.dwell_coordinates[dwell_i]["catheter_index"]
            dwell_idx = self.dwell_coordinates[dwell_i]["dwell_index"]
            combined_plan += "Control Point\n"
            combined_plan += f"weight = {self.dwell_times[dwell_i]/total_dwell_time}\n"
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
            if not combined_only:
                with open(dir_export + f"/dwell_{catheter_idx + 1}_{dwell_idx + 1}_{shield_angle}.plan", "w") as file:
                    file.write(run_i_plan)

        with open(dir_export + "/combined.plan", "w") as file:
            file.write(combined_plan)

    def _export_dwell_mac_file(
        self,
        dir_export: str,
        combined_only: bool = True
    ):
        r"""
        ### Purpose:
        - To export the simulation parameters of the plan into a macro files
        called combine.mac and run_{catheterNumber}_{dwellNumber}_{shieldAngle}.mac
        ### Inputs:
        - dir_export := path to the directory where the export happens
        - combined_only: bool:= if True, only the combined.mac file will be exported. if False,
        the individual dwell position files will also be exported.
        ### Outputs:
        - void := Two types of .mac files are written, one named combined.mac and the other
        named run_{catheterNumber}_{dwellNumber}_{shieldAngle}.mac. combined.plan contains

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
        ### Dependencies:
        - simulation_utils
        """
        for dwell_i in range(self.num_dwells):

            catheter_idx = self.dwell_coordinates[dwell_i]["catheter_index"]
            dwell_idx = self.dwell_coordinates[dwell_i]["dwell_index"]
            # Not dealing with shield angle for now but the new convention for filename is
            # xxx_catheter#_dwell#_shieldangle.plan
            shield_angle = 0
            sim_obj = deepcopy(self.simulation_setup)
            sim_obj.pth_plan = f"dwell_{catheter_idx + 1}_{dwell_idx + 1}_{shield_angle}.plan"
            sim_obj.total_time = 1
            if not combined_only:
                with open(dir_export + f"/run_{catheter_idx + 1}_{dwell_idx + 1}_{shield_angle}.mac", "w") as file:
                    file.write(sim_obj.to_string())

        self.simulation_setup.total_time = np.sum(self.dwell_times)
        with open(dir_export + "/combined.mac", "w") as file:
            file.write(self.simulation_setup.to_string())

    def _export_egsphant(
        self,
        dir_export: Union[str, Path],
        material_dict: Union[dict, Path],
        assign_material_from_ct: bool,
        crop_by_contour: str = None,
        resampled_spacing: List[float] = None,
        resampled_origin: List[float] = None,
        background_material: str = None,
        strict_name_match: bool = True
    ):
        r"""
        ### Purpose:
        - to export the egsphant file of the plan into dir_export
        ### Inputs:
        - dir_export := path to the directory where the export happens
        - material_dict: dict | Path := the dictionary of the materials. if Path, the path to the material file.
        The dictionary contains the name of the elements for each voxel,
        and the following keys: [
            "density" := the density of the material in g/cm^3,
            "HU_limit" := the lower HU limit threshold of the material,
            "structure_name := {optional} the name of the structure in the dicom file that represents the material,"
        ]
        - assign_material_from_ct := if True, the material names will be assigned from the ct.egsphant file.
        ### Outputs:
        - void := egsphant file is generated from phantom and is written to ct.egsphant
        ### Dependencies:
        - BrachyEgsphant
        """
        file_path = dir_export + "/ct.egsphant"
        # if isinstance(material_dict, Path):
        #     with open(material_dict, "r") as json_file:
        #         material_dict = json.load(json_file)

        self.phantom.write_to_egsphant(
            pth_output=Path(file_path),
            material_dict=material_dict,
            assign_material_from_ct=assign_material_from_ct,
            crop_by_contour=crop_by_contour,
            resampled_spacing=resampled_spacing,
            resampled_origin=resampled_origin,
            background_material=background_material,
            strict_name_match=strict_name_match
        )

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

    def _export_structure_set(
        self,
        dir_export: str,
        materials_table: Union[dict, Path] = None,
        export_format: str = "RapidBrachy",
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
        with Pool(processes=8) as pool:
            partial_func = partial(
                _gen_hotspot_mask,
                gridSize=self.phantom.image_obj.gridSize,
                origin=self.phantom.image_obj.origin,
                spacing=self.phantom.image_obj.spacing,
            )
            hotspot_mask_list = list(
                tqdm(pool.imap(partial_func,dwell_pairs),
                total=len(dwell_pairs),
                desc="Generating hotspot structures")
            )
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
    ) -> List[List[np.ndarray]]:
        r"""
        ### Purpose:
        - to get the dose rate matrices for all dwell positions in a given catheter.
        this function assumes that the dose rate tensor is already sorted from small to 
        large catheter and dwell indices.
        ### Inputs:
        - catheter_index := the index of the catheter in the catheter table
        ### Outputs:
        - dose_rate_matrices := a dictionary mapping catheter index to the list of dose rate matrices
        from the dwell positions in that catheter.
        """
        start_doserate_index = 0
        end_doserate_index = 0
        for cat in self.catheter_table:
            num_dwells_in_catheter = len(cat.dwells)
            if cat.index < catheter_index:
                start_doserate_index += num_dwells_in_catheter
            elif cat.index == catheter_index:
                end_doserate_index = start_doserate_index + num_dwells_in_catheter
                break
        dose_rate_indicices = np.arange(
            start=start_doserate_index,
            stop=end_doserate_index,
            step=1)
        return self.dose_rate_dict[dose_rate_indicices]

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

def _export_single_dose_rate(
    dose_grid: np.array,
    dwell_number: int,
    uncertainty: np.array = None,
    doseObj_template: BrachyDose = None,
    dir_export: str = None,
    dose_type: str = None,
):
    r"""
    ### Purpose:
    to write out a single dose rate map given the numpy grid for dose and uncertainty and
    a template dose object that has the same origin, voxel spacing and axis.
    ### Inputs:
    - dose_grid := the numpy array holding the dose rate maps
    - dwell_number:= the dwell number of the dose rate map
    - doseObj_template := a BrachyDose object that has the same origin, voxel spacing and axis
    - dir_export:= the directory to which the dose rate maps will be exported
    - dose_type := the type of dose rate map to be exported. options are ".3ddose", ".minidos", or ".nrrd"
    - uncertainty := the numpy array holding the uncertainty maps
    ### Output:
    - Void := dose file is written to dir_export+f"/run_{dwell_number}"+dose_type
    """
    raise Exception("Bug found here. file name should match the new standard")
    doseObj = BrachyDose.dose_with_empty_grid_like(doseObj_template)
    doseObj.set_dose_array(dose_grid)
    if uncertainty is not None:
        doseObj.set_uncertainty_array(uncertainty)

    doseObj.write_brachydose_to_file(dir_export + f"/run_{dwell_number}" + dose_type)

def _load_single_dose_or_uncertainty_to_dict(
    pth_dose_rate: str, load_dose_or_uncertainty: str = "both"
):
    r""" "
    ### Purpose:
    - To load a single dose rate file into the BrachyPlan object.
    this is to be used in the case of multiprocessing.
    ### Inputs:
    - pth_dose_rate := path to the dose rate file
    - load_dose_or_uncertainty := either "dose", "uncertainty", or "both"
    ### Outputs:
    - dose_or_uncert_map := the dose rate or uncertainty map of the dwell position
    specified by the index.
        If load_dose_or_uncertainty == "both", then dose_or_uncert_map[0] is dose and
        dose_or_uncert_map[1] is uncertainty.
    ### Dependencies:
    - BrachyDose()
    """
    # print("loading dose or uncertainty from:", pth_dose_rate)
    dose_obj = BrachyDose(pth_dose_rate)
    if load_dose_or_uncertainty == "both":
        dose_or_uncert_map = np.zeros(
            (2, *dose_obj.get_dose_array().shape), dtype=np.float32
        )
        dose_or_uncert_map[0] = dose_obj.get_dose_array()
        dose_or_uncert_map[1] = dose_obj.get_uncertainty_array()

    elif load_dose_or_uncertainty == "uncertainty":
        try:
            dose_or_uncert_map = np.zeros_like(
                dose_obj.get_dose_array(), dtype=np.float32
            )
            dose_or_uncert_map = dose_obj.get_uncertainty_array()
        except AttributeError:
            warnings.warn(
                f"uncertainty map is not loaded from {pth_dose_rate}. Moving on...",
                stacklevel=2,
            )

    elif load_dose_or_uncertainty == "dose":
        dose_or_uncert_map = np.zeros_like(dose_obj.get_dose_array(), dtype=np.float32)
        dose_or_uncert_map = dose_obj.get_dose_array()
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
