import os
from glob import glob
import json
import numpy as np
import gc
# from typing import Optional
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager, cpu_count
from functools import partial
import time

from dose_utils import BrachyDose, dose_with_empty_grid_like
from dicom_utils import get_strcuture_mask_from_dicom
from egsphant_utils import BrachyEgsphant 

from scipy import ndimage
from copy import deepcopy


import re
from scipy import interpolate

class BrachyStructure:
    r"""
    Purpose:
        - this class holds the information regarding a structure inside a brachytherapy 
        treatment plan. 
        
    Attributes:
        - name:str
        - name_in_gurobiModel:str
        - bound_coordinates:list
        - penalty_weight:float
        - dvh_metric_name:str
        - dvh_metric_clinical_goal:float
        - dvh_metric_observed:float
    """
    
    def __init__(self):
        self.name:str = None
        self.mask:np.array = None # shape: (z, y, x)
        self.target_volume:bool = None

        # dose volume histogram
        self.in_dvh:bool = None
        self.dvh_metric_name:str = None
        self.dvh_metric_clinical_goal:float = None
        self.dvh_metric_observed:float = None
        self.normalized_cummulative_dvh:np.array = None
        
        # uncertainty volume histogram
        self.uvh:np.array  = None
        self.uncertainty_mean:float = None
        self.uncertainty_std:float = None
        self.uncertainty_max:float = None
        self.uncertainty_min:float = None

        # optimization attributes
        self.name_in_gurobiModel:str = None
        self.bound_coordinates_in_gurobiModel:list = None
        self.penalty_weight_linear:float = None
        self.penalty_weight_quadratic:float = None
        self.penalty_weight_uniformity:float = None
        self.dose_limit:float = None
        self.max_dose:float = 500
        self.min_dose:float = 0
        
        # simulation attributes
        self.density:float = None # 0
        self.density_mode:str = None # ""
        self.material:str = None # "CT Material"

    def get_dvh_metric(self, combined_dose:BrachyDose):
        assert self.mask is not None, "mask is not loaded"
        assert self.dvh_metric_name is not None, "dvh metric name is not set"
        assert self.dvh_metric_clinical_goal is not None, "dvh metric clinical goal is not set"

        num_bins = int(combined_dose.grid.max()*10) + 1
        total_dose_max = combined_dose.grid.max()

        structure_dose = combined_dose.grid * self.mask
        structure_dose = structure_dose[structure_dose != 0].flatten()
        voxel_volume = np.prod(combined_dose.vox_size)
        num_voxels_in_structure = np.sum(self.mask)

        if "%" in self.dvh_metric_name:
            histogram_limit = float(*re.findall('-?\d+\.?\d*', self.dvh_metric_name))
        elif "cc" in self.dvh_metric_name:
            histogram_limit = float(*re.findall('-?\d+\.?\d*', self.dvh_metric_name))\
                /(voxel_volume*num_voxels_in_structure)*100
        else:
            raise ValueError("invalid name for DVH metric name. \
                The metric should have percent sign (%) or cc.")

        self.dvh_metric_observed, self.normalized_cummulative_dvh = \
            dvh_metric(structure_dose, num_bins, total_dose_max, histogram_limit, voxel_volume)

    def to_dict(self, export_format:str):
        r"""
        Purpose:
            - To export the BrachyStructure object into a dictionary of a certain format.
        Inputs:
            - export_format := the export_format of the exported plan. an example is:
                - "RapidBrachyExport":{
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
        elif export_format == "RapidBrachyExport":
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
                "uniformity_weight": self.penal_weight_uniformity}
    
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
        - extract_dwell_numbers_times_coordinates_from_catheterTable()
        - load_dose_rate_or_uncertainty_tensor()
        - set_dvh_metric_goals()
        - create_structures()
        - calculate_DVH_metrics()
        - calculate_combined_uncertainty()
        - calculate_uncertainty_per_structure()
    """
    
    def __init__(
            self, 
            # for loading catheter table:
            pth_catheterTable_json:str=None,
            
            # for loading dose or uncertainty:
            dir_dose_rate:str=None,
            type_dose_file:str=".nrrd",
            load_dose_or_uncertainty:str="dose",
            multi_processing:bool=False,
            
            # for structure creation:
            dvh_metric_goals:dict=None,
            dir_structure_source:str=None,
            dose_cropped_by_body:bool=True,
            
            # for simulation setup:
            dir_egsphant:str=None,
            dir_applicator_geometry:str=None,
            dir_applicator_materials:str=None,
            ):
        r"""
        Purpose:
            - To initialize the BrachyPlan object.
        Inputs:
            # for loading catheter table:
            - pth_catheterTable_json:str := path to a json file containing the information of the catheter table.
            # for loading dose or uncertainty:
            - dir_dose_rate:str := path to the directory containing the dose rate files for a patient.
            - type_dose_file:str = ".nrrd" := the type of dose file to load (default is ".nrrd").
            - load_dose_or_uncertainty:str = "dose" := specify whether to load "dose" or "uncertainty" or "both" (default is "dose").
            - multi_processing:bool = False := flag to enable multi-processing for loading dose or uncertainty (default is False).
            # for structure creation:
            - dvh_metric_goals:dict = None := dictionary containing the DVH metric goals (default is None).
            - dir_structure_source:str = None := path to the directory containing the structures (default is None).
            - dose_cropped_by_body:bool = True := flag to indicate whether the dose is cropped by body (default is True).
        Outputs:
            - Void := will initialize the BrachyPlan object
        Dependencies:
            -  
        """
        # catheter table attributes
        self.catheter_table = None
        self.num_catheters = None
        self.catheter_numbers = np.array([], dtype=int) #shape: (num_catheters, 1)
        self.num_dwells = None
        self.dwell_numbers = np.array([], dtype=int) #shape: (num_dwells, 1)
        self.dwell_times = np.array([], dtype=np.float32) #shape: (num_dwells, 1)
        self.dwell_coordinates = [] #shape: (num_dwells, 3) 

        # dose attributes
        self.dose_rate_tensor = np.array([], dtype=np.float32) #shape: (num_dwells, z, y, x)
        self.combined_dose = None
        self.uncertainty_tensor = np.array([], dtype=np.float32) #shape: (num_dwells, z, y, x)

        # sturctures attributes
        # self.organ_bounds = None
        self.dvh_metric_goals = None
        self.structure_list = []

        # imaging attributes [for future]
        # self.ct_image = None
        # self.mr_image = None
        # self.ultrasound_image = None
        
        # simulation attributes
        self.egsphant = None
        self.applicator_geometry = None
        self.applicator_materials = None

        # load the catheter table if the path is provided
        if pth_catheterTable_json is not None:
            self.load_catheterTable_json(pth_catheterTable_json)

        if dir_dose_rate is not None:
            self.load_dose_rate_or_uncertainty_tensor(
                dir_dose_rate, 
                type_dose_file=type_dose_file, 
                load_dose_or_uncertainty=load_dose_or_uncertainty, 
                multi_processing=multi_processing)

        if dir_structure_source is not None and dvh_metric_goals is not None:
            self.set_dvh_metric_goals(dvh_metric_goals)
            self.create_structures(dir_structure_source, dose_cropped_by_body)
            
        if dir_egsphant is not None:
            self.egsphant = BrachyEgsphant(dir_egsphant)
            
        if dir_applicator_geometry is not None or dir_applicator_materials is not None:
            raise NotImplementedError("to be implemented soon")
        

    def load_catheterTable_json(
        self, 
        pth_catheterTable_json:str):
        r"""
        Purpose:
            - To load the contents of a catheter table into the Brachy plan.
        Inputs:
            - pth_catheterTable_json := path to a json file having the info on the catheter table. 
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
        with open(pth_catheterTable_json, 'r') as json_file:
            catheter_table = json.load(json_file)
       
        self.catheter_table = catheter_table
        self.extract_dwell_numbers_times_coordinates_from_catheterTable()
    
    def extract_dwell_numbers_times_coordinates_from_catheterTable(self):
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
        self.dwell_numbers, self.dwell_times, self.dwell_coordinates = \
             np.array([], dtype=int),  np.array([], dtype=np.float32), []
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
                        "catheterId": catheter["id"]
                    })
                dwell_counter += 1
        
        assert len(self.catheter_numbers)-1 == self.catheter_numbers[-1], "catheter numbers are not extracted correctly"
        self.num_catheters = len(self.catheter_numbers)
        
        assert len(self.dwell_numbers) == self.dwell_numbers[-1], "dwell numbers are not extracted correctly"
        self.num_dwells = len(self.dwell_numbers)

    def update_catheter_table_from_plan(self):
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
        assert len(self.dwell_coordinates) !=0, "dwell coordinates are not extracted"
        assert self.num_dwells is not None, "number of dwells is not extracted"
        
        self.catheter_table = []
        
        for catheter_i in self.catheter_numbers:
            catheter = {}
            catheter["id"] = catheter_i
            catheter["points"] = []
            catheter["dwells"] = []
            dwell = {}
            for dwell_i in self.dwell_numbers:
                if self.dwell_coordinates[dwell_i-1]["catheterId"] != catheter_i:
                    continue
                dwell["angle"] = self.dwell_coordinates[dwell_i-1]["angle"]
                dwell["position"] = self.dwell_coordinates[dwell_i-1]["position"]
                dwell["relativePos"] = self.dwell_coordinates[dwell_i-1]["relativePos"]
                dwell["rotation"] = self.dwell_coordinates[dwell_i-1]["rotation"]
                dwell["time"] = self.dwell_times[dwell_i-1]
                dwell["weight"] = self.dwell_times[dwell_i-1] / np.sum(self.dwell_times)
                catheter["dwells"].append(deepcopy(dwell))   
               
            self.catheter_table.append(deepcopy(catheter)) 
        
    def update_after_change_in_plan(self):
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
        self.update_catheter_table_from_plan()
        self.calculate_combined_dose()
    
    def load_dose_rate_or_uncertainty_tensor(
        self, 
        dir_dose_rate:str,
        type_dose_file:str=".nrrd",
        load_dose_or_uncertainty:str="dose",
        multi_processing:bool=False):
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
        assert len(self.dwell_coordinates) !=0, "dwell coordinates are not extracted"
        assert self.num_dwells is not None, "number of dwells is not extracted"
        
        # here is the list of the dose rate files
        dose_rate_files = glob(os.path.join(dir_dose_rate, f"*{type_dose_file}"))
        
        dose_rate_files = [dosefile for dosefile in dose_rate_files if "combined" not in dosefile]
        
        dose_rate_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))
        assert len(dose_rate_files) == self.num_dwells, "number of dose rate files does not match the number of dwell positions"

        test_dose_obj = BrachyDose(dose_rate_files[0])
        
        if load_dose_or_uncertainty not in ["dose", "uncertainty", "both"]:
            raise ValueError("load_dose_or_uncertainty should be either 'dose', 'uncertainty', or 'both'")
                 
        # load the dose rate tensor 
        if multi_processing:
            with Pool(8) as mp_pool:
                dose_or_uncertainty_list = np.array(mp_pool.map(
                        partial(_load_single_dose_or_uncertainty_to_dict, load_dose_or_uncertainty=load_dose_or_uncertainty),
                        dose_rate_files), dtype=np.float32)  
                
        else:  
            dose_or_uncertainty_list = np.empty(len(dose_rate_files), dtype=object)
            for i, pth_dose_rate in tqdm(enumerate(dose_rate_files)):     
                dose_or_uncertainty_list[i] = _load_single_dose_or_uncertainty_to_dict(pth_dose_rate, load_dose_or_uncertainty)
        
        if load_dose_or_uncertainty == "both":
            self.dose_rate_tensor = np.array(dose_or_uncertainty_list[:, 0], dtype=np.float32)
            self.uncertainty_tensor = np.array(dose_or_uncertainty_list[:, 1], dtype=np.float32) 
        elif load_dose_or_uncertainty == "dose":
            self.dose_rate_tensor = np.array(dose_or_uncertainty_list, dtype=np.float32)
        elif load_dose_or_uncertainty == "uncertainty":
            self.uncertainty_tensor = np.array(dose_or_uncertainty_list, dtype=np.float32)
        else:
            raise ValueError("load_dose_or_uncertainty should be either 'dose', 'uncertainty', or 'both'")
        
        del dose_or_uncertainty_list
        gc.collect()
        
        self.combined_dose = dose_with_empty_grid_like(test_dose_obj)
        # for debugging{
        # assert np.array_equal(
        #     np.concatenate(self.combined_dose.voxel_edges), 
        #     np.concatenate(test_dose_obj.voxel_edges)), \
        #     "voxel edges of combined dose map and dwell dose rate map do not match"

        # assert self.combined_dose.is_not_empty(), "combined dose is empty"
        # }
        if load_dose_or_uncertainty != "uncertainty":
            self.calculate_combined_dose()
        if load_dose_or_uncertainty != "dose":
            self.calculate_combined_uncertainty()
        
    def calculate_combined_dose(self):
            """
            Purpose:
            - To calculate the combined dose by multiplying the dose rate tensor with the dwell times array.
            The result is stored in the combined_dose attribute.
    
            Raises:
                AssertionError: If the dose rate tensor or dwell times array is empty.
            """
            assert self.dose_rate_tensor.size != 0, "dose rate tensor is empty. Run load_dose_rate_or_uncertainty_tensor()"
            assert self.dwell_times.size != 0, "dwell times array is empty. Run extract_dwell_numbers_times_coordinates_from_catheterTable()"
            
            # calculate the combined dose and store the result in the combined_dose attribute 
                # this implementation is a little slow, and very very memory efficient
            for i in range(self.num_dwells):
                self.combined_dose.grid += self.dose_rate_tensor[i] * self.dwell_times[i]
                # this implementation is a bit faster, but very memory inefficient
                # self.combined_dose.grid = np.sum(
                #     self.dose_rate_tensor * self.dwell_times[:, np.newaxis, np.newaxis, np.newaxis],
                #     axis=0)
    
    def set_dvh_metric_goals(self, dvh_metric_goals:dict):
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
            assert "D" in dvh_metric, "dvh metric name should start with D as we are only supporting dose metrics for now"
            assert "cc" in dvh_metric or "%" in dvh_metric, "dvh metric name should end with cc or '%' to signify the absolute or relative volume"
            assert dvh_metric_goals[dvh_metric] != None, "for each dvh metric, the clinical threshold should be provided in Gy."

        self.dvh_metric_goals = dvh_metric_goals
        
    def create_structures(
        self,
        dir_structures_source:str, 
        dose_cropped_by_body:bool=True):
        r"""
        Purpose: 
            - To create a list of BrachyStructure objects given the path to the directory
            containing the structure masks. the list is stored in the BrachyPlan.structure_list attribute.
        Inputes:
            - dir_structures_source := path to the directory containing the structure masks. 
            this could be dicom files or nrrd files.
            - size_uncropped_dose_grid := the size of the uncropped dose grid. this is needed to
            match the size of the structure mask to the size of the dose grid.
        Outputs:
            - Void := will update the BrachyPlan.structure_list attribute
        Dependencies:
            - get_strcuture_mask_from_dicom
        """
        assert self.dvh_metric_goals is not None, "dvh metric goals are not set, run set_dvh_metric_goals()"

        structure_name_list = ['body']
        for dvh_metric in self.dvh_metric_goals:
            structure_obj = BrachyStructure()
            structure_obj.name = dvh_metric.split("(")[-1].split(")")[0]
            structure_name_list.append(structure_obj.name)
            structure_obj.dvh_metric_name = dvh_metric.split("(")[0]
            structure_obj.dvh_metric_clinical_goal = self.dvh_metric_goals[dvh_metric]
            self.structure_list.append(structure_obj)

        # load the structure mask
        structure_mask_dict = load_structure_mask(dir_structures_source, structure_name_list)

        # get the index extent of body contour on each axis
        if dose_cropped_by_body:
            body_index_range = np.zeros([3, 2], dtype=int)
            for i in range(3):
                body_index_range [i, :] = np.floor(np.array([
                    np.argwhere(structure_mask_dict["body"]==1)[:, i].min(), 
                    # off set of +1 is added to acount for python stopping before range end
                    np.argwhere(structure_mask_dict["body"]==1)[:, i].max()+1])).astype(int)
                        
        for structure in self.structure_list:
            mask = structure_mask_dict[structure.name]
            # apply body contour mask to the structure mask
            if dose_cropped_by_body:
                mask = mask[
                    body_index_range[0][0]:body_index_range[0][1], 
                    body_index_range[1][0]:body_index_range[1][1], 
                    body_index_range[2][0]:body_index_range[2][1]]
            
            structure.mask = ndimage.zoom(
                mask, 
                np.array(self.combined_dose.grid.shape)/mask.shape, 
                order=0)
            
            # print(structure.mask.shape)
            # print(size_uncropped_dose_grid)
            # print(self.combined_dose.grid.shape)
    
    def calculate_combined_uncertainty(self):
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
            self.combined_dose.uncertainty += (self.uncertainty_tensor[i] * normalized_times[i])**2
        self.combined_dose.uncertainty = np.sqrt(self.combined_dose.uncertainty)
        
        # This implementation is a bit faster, but very memory inefficient
        # self.combined_dose.uncertainty = np.sqrt(
        #     np.sum(
        #         (self.uncertainty_tensor * normalized_times[:, np.newaxis, np.newaxis, np.newaxis])**2,
        #         axis=0))
                
    def calculate_DVH_metrics(self):
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
        for structure_obj in self.structure_list:
            structure_obj.get_dvh_metric(self.combined_dose)

    def calculate_uncertainty_per_structure(self):
        r"""
        Purpose:
            - To calculate the uncertainty of each structure in the BrachyPlan.
        Inputs:
            - self := the BrachyPlan object
        Outputs:
            - Void := will update the BrachyStructure.uncertainty attribute
        """
        assert self.combined_dose.uncertainty is not None, "combined uncertainty is not calculated yet"
        assert self.structure_list is not None, "structure list is not created yet"
        for structure_obj in self.structure_list:
            # Apply structure mask to the uncertainty map
            masked_uncertainty = self.combined_dose.uncertainty * structure_obj.mask
            # isolate the uncertainty values that are in the mask
            flattened_uncertainty = masked_uncertainty[structure_obj.mask != 0].flatten()
            # generate a histogram from the masked uncertainty
            histogram, bins_edges = np.histogram(
                flattened_uncertainty, 
                bins=100, 
                range=(0, flattened_uncertainty.max()+0.1))
            structure_obj.uvh = histogram * np.prod(self.combined_dose.vox_size)
            structure_obj.uncertainty_mean = np.mean(flattened_uncertainty)
            structure_obj.uncertainty_std = np.std(flattened_uncertainty)
            structure_obj.uncertainty_max = np.max(flattened_uncertainty)
            structure_obj.uncertainty_min = np.min(flattened_uncertainty)
    
    def export_plan(
        self, 
        export_format:str, 
        dir_export:str, 
        content_to_export:dict):
        r"""
        Purpose: 
            - To export the treatment plan file into a given export_format.
            The export_format can be either "RapidBrachyExport" or "WebAppExport".
        Inputs:
            - export_format := the export_format of the exported plan. options are:
                - "RapidBrachyExport":
                    "run_#.3ddose" or "run_#.minidos" or "run_#.nrrd",
                    "catheter_table.json"
                    "dwell_#.plan",
                    "run_#.mac", 
                    "ct.egsphant",
                    "ApplicatorMaterials"
                    "applicator_geometry.json",
                    "structure_set.json"
                     
                - "WebApp": Not implemented yet
                    "run_#.nrrd",
                    "dwell_#.json",
                    "run_#.json",

            - dir_export := the directory to which the plan will be exported.
            - content_to_export := a dictionary with which the user specifies what parts
            of the plan to export. everything is binary (True or False) except for 
            "dose type", which can be either ".3ddose", ".minidos", or ".nrrd".
            the keys of content_to_export are: 
            {
                "dose", "dose type", "uncertainty", "dose rate maps", 
                "catheter_table", "plan", "mac", "egsphant",
                "ApplicatorMaterials", applicator_geometry", "structure_set", 
            }. 
        
        Outputs:
            - Void := will export the available parts of a plan into the specified export_format. 
        """
        assert os.path.exists(dir_export), \
            "export directory does not exist. please make the directory first"

        if export_format =="WebApp":
            
            raise NotImplementedError("export to WebApp is not implemented yet")
            
        elif export_format =="RapidBrachyExport":
            
            if content_to_export["dose"]:
                self.export_dose(
                    dir_export, 
                    # content_to_export["uncertainty"], 
                    content_to_export["dose type"],
                    content_to_export["dose rate maps"])
                
            elif content_to_export["catheter_table"]:
                # assumes file name is "catheter_table.json"
                self.export_catheter_table(dir_export)
            
            elif content_to_export["plan"]:
                # assumes file name is "dwell_#.plan"
                self.export_plan(dir_export)
            
            elif content_to_export["mac"]:
                # assumes file name is "run_#.mac"
                self.export_mac(dir_export)
                
            elif content_to_export["egsphant"]:
                # assumes file name is "ct.egsphant"
                self.export_egsphant(dir_export)
            
            elif content_to_export["ApplicatorMaterials"]:
                # assumes file name is "ApplicatorMaterials"
                self.export_applicator_materials(dir_export)
            
            elif content_to_export["applicator_geometry"]:
                # assumes file name is "applicator_geometry.json"
                self.export_applicator_geometry(dir_export)
            
            elif content_to_export["structure_set"]:
                # assumes file name is "structure_set.json"
                self.export_structure_set(dir_export)
          
    def export_dose(
        self, 
        dir_export:str, 
        # uncertainty=False, 
        dose_type=".minidos", 
        dose_rate_maps=False):
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
        self.combined_dose.write_brachydose_to_file(dir_export+"/combined"+dose_type)
        
        if dose_rate_maps:
            if cpu_count() < 4:
                for i in self.dwell_numbers:
                    _export_single_dose_rate(
                        self.dose_rate_tensor[i-1], 
                        i,
                        self.combined_dose,
                        dir_export, 
                        dose_type,
                        self.uncertainty_tensor[i-1])
            else:
                with Pool(cpu_count()-2) as mp_pool:
                    mp_pool.starmap(
                        partial(
                            _export_single_dose_rate,
                            doseObj_template=self.combined_dose,
                            dir_export=dir_export,
                            dose_type=dose_type),
                        [(dose_grid, dwell_number, uncertainty) \
                            for dose_grid, dwell_number, uncertainty \
                                in zip(self.dose_rate_tensor, 
                                       self.dwell_numbers, 
                                       self.uncertainty_tensor)]
                    )

    def export_catheter_table(self, dir_export:str):
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
        # raise NotImplementedError("to be implemented soon")
        file_path = dir_export + "/catheter_table.json"
        with open(file_path, 'w') as file:
            json.dump(self.catheter_table, file, indent=4)
        
    def export_plan(self, dir_export:str):
        r"""
        Purpose:
        Inputs:
        Outputs:
        Dependencies:
        """
        raise NotImplementedError("to be implemented soon")
    
    def export_mac(self, dir_export:str):
        r"""
        Purpose:
        Inputs:
        Outputs:
        Dependencies:
        """
        raise NotImplementedError("to be implemented soon")
    
    def export_egsphant(self, dir_export:str):
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
        # raise NotImplementedError("to be implemented soon")
        file_path = dir_export + "/ct.egsphant"
        self.egsphant.write_to_ctegsphant(file_path)
        
    def export_applicator_materials(self, dir_export:str):
        r"""
        Purpose:
        Inputs:
        Outputs:
        Dependencies:
        """
        raise NotImplementedError("to be implemented soon")
    
    def export_applicator_geometry(self, dir_export:str):
        r"""
        Purpose:
        Inputs:
        Outputs:
        Dependencies:
        """
        raise NotImplementedError("to be implemented soon")
    
    def export_structure_set(self, dir_export:str, export_format:str="RapidBrachyExport"):
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
        # raise NotImplementedError("to be implemented soon")
        structure_set = []
        for structure in self.structure_list:
            structure_set.append(structure.to_dict(export_format))
            
        file_path = dir_export + "/structure_set.json"
        with open(file_path, 'w') as file:
            json.dump(structure_set, file, indent=4)
        
def _export_single_dose_rate(
    dose_grid:np.array, 
    dwell_number:int,
    uncertainty:np.array=None,
    doseObj_template:BrachyDose=None, 
    dir_export:str=None, 
    dose_type:str=None, 
    ):
    r"""
    Purpose:
        to write out a single dose rate map given the numpy grid for dose and uncertainty and 
        a template dose object that has the same origin, voxel spacing and axis. 
    Inputs:
        - dose_grid :=  
        - dwell_number:= 
        - doseObj_template :=  
        - dir_export:= 
        - dose_type :=
        - uncertainty := 
        
    Output:
        - Void := dose file is written to dir_export+f"/run_{dwell_number}"+dose_type
    """
    doseObj = dose_with_empty_grid_like(doseObj_template)
    doseObj.grid = dose_grid
    if uncertainty is not None:
        doseObj.uncertainty = uncertainty
    
    doseObj.write_brachydose_to_file(dir_export+f"/run_{dwell_number}"+dose_type)


def load_structure_mask(
    dir_structure_source:str,
    structure_name_list:list,
    structure_source_type:str=".dcm"):

    if structure_source_type == ".dcm":
        print("loading structure set from dicom files")
        structure_mask_dict = get_strcuture_mask_from_dicom(dir_structure_source, structure_name_list)

    elif structure_source_type == ".nrrd":    
        print("loading structure set from nrrd file")
        raise NotImplementedError("loading structure set from .nrrd file is not implemented yet")
    else:
        raise ValueError("structure source type is not recognized")
    
    return structure_mask_dict

def dvh_metric(
        dose:np.array, 
        num_bins:int, 
        total_dose_max:float, 
        threshold:float, 
        voxel_volume:float, 
        normalize_dose_by=None):
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
    
    histogram, bins_edges = np.histogram(dose, bins=num_bins, range=(0, total_dose_max+0.1))
    vol_hist = histogram * voxel_volume
    vol_hist = np.append(np.trim_zeros(vol_hist, trim='b'), 0)

    cum_dvh = np.cumsum(vol_hist[::-1])[::-1]
    normalized_cum_dvh = cum_dvh * 100 / cum_dvh[0]
    if normalize_dose_by is not None:
        dvh_dose_axis = bins_edges[:len(cum_dvh)]/normalize_dose_by
    else:
        dvh_dose_axis = bins_edges[:len(cum_dvh)]
    # for debugging{ let's plot the normalized dvh. nomralization is done both on dose and volume domains
    # dvh_plot = plt.plot(dvh_dose_axis, normalized_cum_dvh)
    # plt.show()
    # }
    f = interpolate.interp1d(normalized_cum_dvh, dvh_dose_axis, kind="linear")

    # in future, one could pass the DVH plot to be stored in the structure object. 
    return f(threshold), normalized_cum_dvh

def _load_single_dose_or_uncertainty_to_dict(
            pth_dose_rate:str,
            load_dose_or_uncertainty:str="both" 
            ):
        r""""
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
                dose_or_uncert_map = np.zeros_like(BrachyDose(pth_dose_rate).grid, dtype=np.float32)
                dose_or_uncert_map = dose_obj.uncertainty
            except:
                Warning(f"uncertainty map for dwell number {index} is not loaded from {pth_dose_rate}. Moving on...")
        elif load_dose_or_uncertainty == "dose":
            dose_or_uncert_map = np.zeros_like(BrachyDose(pth_dose_rate).grid, dtype=np.float32)
            dose_or_uncert_map = dose_obj.grid
        else:
            raise ValueError("load_dose_or_uncertainty should be either 'dose', 'uncertainty', or 'both'")
        
        return dose_or_uncert_map
        
def test_load_catheterTable_json():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    
    with open(pth_cathTable_json, 'r') as json_file:
        ground_truth_catheter_table = json.load(json_file)
    
    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)

    # print(plan_obj.catheter_table)
    assert [i for i in ground_truth_catheter_table if i not in plan_obj.catheter_table] ==[],\
        "loading catheter table did not work as expected"

def test_extract_dwell_numbers_times_coordinates_from_catheterTable():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    plan_obj.extract_dwell_numbers_times_coordinates_from_catheterTable()
    
    assert plan_obj.dwell_numbers is not None, "dwell numbers not extracted"
    assert plan_obj.dwell_times is not None, "dwell times not extracted"
    assert plan_obj.dwell_coordinates is not None, "dwell coordinates not extracted"
    
    print(f"The shape of the dwell_number is {plan_obj.dwell_numbers.shape}")
    print(f"The shape of the dwell_times is {plan_obj.dwell_times.shape}")
    print(f"The shape of the dwell_coordinates is {len(plan_obj.dwell_coordinates)}")

def test_load_dose_rate_or_uncertainty_tensor():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    plan_obj.extract_dwell_numbers_times_coordinates_from_catheterTable()

    plan_obj.load_dose_rate_or_uncertainty_tensor(dir_dose_rate, load_dose_or_uncertainty="both")
    print(f"The shape of the dose rate tensor is {plan_obj.dose_rate_tensor.shape}")
    print(f"The shape of the combined dose is {plan_obj.combined_dose.grid.shape}")

def test_set_dvh_metric_goals():
    dvh_metric_goals = {
        'D95%(ctv)': 15,
        'D1cc(rectum)': 11.25,
        'D0.1cc(urethra)': 18.75 
    }
    plan_obj = BrachyPlan()
    plan_obj.set_dvh_metric_goals(dvh_metric_goals)
    assert plan_obj.dvh_metric_goals == dvh_metric_goals, "dvh metric list not set correctly"
    print(plan_obj.dvh_metric_goals)

def test_create_structures_and_calc_dvh_metrics():
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    # dir_dose_rate = "../../data_test/prostate-glen-p1-dose"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose/"
    dvh_metric_goals = {
        'D95%(ctv)': 15,
        'D1cc(rectum)': 11.25,
        'D0.1cc(urethra)': 18.75
    }

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    plan_obj.extract_dwell_numbers_times_coordinates_from_catheterTable()
    plan_obj.load_dose_rate_or_uncertainty_tensor(dir_dose_rate, load_dose_or_uncertainty="both")
    plan_obj.set_dvh_metric_goals(dvh_metric_goals)

    plan_obj.create_structures(dir_dicom, True)
    plan_obj.calculate_DVH_metrics()
    for structure in plan_obj.structure_list:
        print(f"{structure.name}: {structure.dvh_metric_observed}")

def test_calculate_combined_uncertainty():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    plan_obj.extract_dwell_numbers_times_coordinates_from_catheterTable()

    plan_obj.load_dose_rate_or_uncertainty_tensor(dir_dose_rate, load_dose_or_uncertainty="both")
    plan_obj.calculate_combined_uncertainty()
    print(f"The shape of the combined uncertainty is {plan_obj.combined_dose.uncertainty.shape}")
    assert plan_obj.combined_dose.uncertainty.shape == plan_obj.combined_dose.grid.shape, \
        "combined uncertainty shape does not match combined dose shape"

def test_calculate_uncertainty_per_structure():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        'D95%(ctv)': 15,
        'D1cc(rectum)': 11.25,
        'D0.1cc(urethra)': 18.75
    }
    
    plan_obj = BrachyPlan(
        pth_cathTable_json,
        dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
        dir_structure_source=dir_dicom,
        dvh_metric_goals=dvh_metric_goals)
    
    plan_obj.calculate_uncertainty_per_structure()
    for structure in plan_obj.structure_list:
        print(f"{structure.name}: mean: {structure.uncertainty_mean},\n \
            std: {structure.uncertainty_std}, \n \
            max: {structure.uncertainty_max}, \n \
            min: {structure.uncertainty_min}")

def test_BrachyPlan():
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose/"
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        'D95%(ctv)': 15,
        'D1cc(rectum)': 11.25,
        'D0.1cc(urethra)': 18.75
    }
    t0 = time.time()
    plan_obj = BrachyPlan(
        pth_cathTable_json, 
        dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
        dir_structure_source=dir_dicom,
        dvh_metric_goals=dvh_metric_goals) 
    t1 = time.time()
    print(f"loading the plan took {t1-t0} seconds")

def test__load_single_dose_or_uncertainty_to_dict():
    pth_dose_rate = "../../data_test/prostate-glen-p1-dose/scaled_run_1.nrrd"
    _load_single_dose_or_uncertainty_to_dict(pth_dose_rate, "both")
    print(dose_rate_dict[1]["dose"].shape)
    print(dose_rate_dict[1]["uncertainty"].shape)

def test_all_exports():
    dir_export = "../../data_test/test_export_plan"
    
    # boiler plate to create a dose object.
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose/"
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        'D95%(ctv)': 15,
        'D1cc(rectum)': 11.25,
        'D0.1cc(urethra)': 18.75
    }
    t0 = time.time()
    plan_obj = BrachyPlan(
        pth_cathTable_json, 
        dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
        dir_structure_source=dir_dicom,
        dvh_metric_goals=dvh_metric_goals) 
    
    # test export dose
    # passed for minidos, nrrd, 3ddose 
    # plan_obj.export_dose(dir_export, dose_type=".nrrd", dose_rate_maps=True)
    # assert bool(os.listdir(dir_export)), "no dose files were exported"
    
    # test export catheter table
    plan_obj.export_catheter_table(dir_export=dir_export)
    
def test_update_after_change_in_plan():
    dir_export = "../../data_test/test_export_plan"
    
    # boiler plate to create a dose object.
    pth_cathTable_json = "../../data_test/prostate-glen-p1-planFiles/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose/"
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    dvh_metric_goals = {
        'D95%(ctv)': 15,
        'D1cc(rectum)': 11.25,
        'D0.1cc(urethra)': 18.75
    }
    t0 = time.time()
    plan_obj = BrachyPlan(
        pth_cathTable_json, 
        dir_dose_rate,
        load_dose_or_uncertainty="both",
        multi_processing=True,
        dir_structure_source=dir_dicom,
        dvh_metric_goals=dvh_metric_goals) 
    old_combined_dose = plan_obj.combined_dose
    
    plan_obj.update_after_change_in_plan()

    assert plan_obj.combined_dose.is_equal(old_combined_dose), "combined dose is not updated after change in plan"
    
if __name__ == "__main__":
    
    # running the test functions above: 
    # test_load_catheterTable_json()
    # test_extract_dwell_numbers_times_coordinates_from_catheterTable()
    # test_load_dose_rate_or_uncertainty_tensor()
    # test_set_dvh_metric_goals()
    # test_create_structures_and_calc_dvh_metrics()
    # test_calculate_combined_uncertainty()
    # test_calculate_uncertainty_per_structure()
    # test_BrachyPlan()
    # test__load_single_dose_or_uncertainty_to_dict()
    # test_all_exports()
    test_update_after_change_in_plan()
    