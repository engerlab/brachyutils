import os
from glob import glob
import json
import numpy as np
import gc
# from typing import Optional
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager
from functools import partial
import time

from dose_utils import BrachyDose
from copy import deepcopy

from dicom_utils import get_strcuture_mask_from_dicom
from scipy import ndimage

import re
from scipy import interpolate

class BrachyStructure:
    r"""
    Purpose:
        - this class holds the inexport_formation regarding a structure inside a brachytherapy 
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
        name:str = None
        mask:np.array = None # shape: (z, y, x)

        # dose volume histogram
        dvh_metric_name:str = None
        dvh_metric_clinical_goal:float = None
        dvh_metric_observed:float = None
        normalized_cummulative_dvh:np.array = None
        
        # uncertainty volume histogram
        uvh:np.array  = None
        uncertainty_mean:float = None
        uncertainty_std:float = None
        uncertainty_max:float = None
        uncertainty_min:float = None

        # optimization parameters
        name_in_gurobiModel:str = None
        bound_coordinates_in_gurobiModel:list = None
        penalty_weight:float = None

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
            histogram_limit = float(*re.findall('-?\d+\.?\d*', self.dvh_metric_name))/(voxel_volume*num_voxels_in_structure)*100
        else:
            raise ValueError("invalid name for DVH metric name. The metric should have percent sign (%) or cc.")

        self.dvh_metric_observed, self.normalized_cummulative_dvh = dvh_metric(structure_dose, num_bins, total_dose_max, histogram_limit, voxel_volume)

class BrachyPlan:
    r"""
    Purpose:
        - This class holds the inexport_formation regarding the brachytherapy treatment plan
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
        - dwell_coordinates:list := a list of dictionaries. each dictionary contains the keys "position", "rotation", and "relativePos"
        - organ_bounds:dict
        - dose_rate_tensor:np.array := dose rate from dwell position 1 to num_dwells. 
        matches the dwell_number_list. shape: (num_dwells, z, y, x)
        - uncertainty_tensor:np.array := uncertainty from dwell position 1 to num_dwells. shape: (num_dwells, z, y, x)
        - brachy_structure:list[BrachyStructure] := the list of patient structures in the plan
    
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
            dose_cropped_by_body:bool=True):
        r"""
        Purpose:
            - To initialize the BrachyPlan object.
        Inputs:
            # for loading catheter table:
            - pth_catheterTable_json:str := path to a json file containing the inexport_formation of the catheter table.
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
        self.num_dwells = None
        self.catheter_table = None
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

        # load the catheter table if the path is provided
        if pth_catheterTable_json is not None:
            self.load_catheterTable_json(pth_catheterTable_json)
            self.extract_dwell_numbers_times_coordinates_from_catheterTable()

        if dir_dose_rate is not None:
            self.load_dose_rate_or_uncertainty_tensor(
                dir_dose_rate, 
                type_dose_file=type_dose_file, 
                load_dose_or_uncertainty=load_dose_or_uncertainty, 
                multi_processing=multi_processing)

        if dir_structure_source is not None and dvh_metric_goals is not None:
            self.set_dvh_metric_goals(dvh_metric_goals)
            self.create_structures(dir_structure_source, dose_cropped_by_body)

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
        
        # load the json file
        with open(pth_catheterTable_json, 'r') as json_file:
            catheter_table = json.load(json_file)
       
        self.catheter_table = catheter_table
    
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
        dwell_counter = 1
        for catheter in self.catheter_table:
            for dwell in catheter['dwells']:
                self.dwell_numbers = np.append(self.dwell_numbers, dwell_counter)
                self.dwell_times = np.append(self.dwell_times, dwell['time'])
                self.dwell_coordinates.append(
                    {
                        "position": dwell['position'],
                        "rotation": dwell['rotation'],
                        "relativePos": dwell["relativePos"]
                    })
                dwell_counter += 1
        self.num_dwells = len(self.dwell_numbers)

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
        assert self.dwell_numbers is not None, "dwell numbers are not extracted"
        assert self.dwell_times is not None, "dwell times are not extracted"
        assert self.dwell_coordinates is not None, "dwell coordinates are not extracted"
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
        
        self.combined_dose = BrachyDose()
        self.combined_dose.grid = np.zeros_like(test_dose_obj.grid)
        self.combined_dose.num_voxels = test_dose_obj.num_voxels        
        self.combined_dose.vox_size = test_dose_obj.vox_size
        self.combined_dose.topleft = test_dose_obj.topleft
        self.combined_dose.calculate_voxel_edges()
            # for debugging{
            # assert np.array_equal(
            #     np.concatenate(self.combined_dose.voxel_edges), 
            #     np.concatenate(test_dose_obj.voxel_edges)), \
            #     "voxel edges of combined dose map and dwell dose rate map do not match"

            # assert self.combined_dose.is_not_empty(), "combined dose is empty"
            # }
        if load_dose_or_uncertainty != "uncertainty":
            # calculate the combined dose and store the result in the combined_dose attribute 
            # this implementation is a little slow, and very very memory efficient
            for i in range(self.num_dwells):
                self.combined_dose.grid += self.dose_rate_tensor[i] * self.dwell_times[i]
            # this implementation is a bit faster, but very memory inefficient
            # self.combined_dose.grid = np.sum(
            #     self.dose_rate_tensor * self.dwell_times[:, np.newaxis, np.newaxis, np.newaxis],
            #     axis=0)
        if load_dose_or_uncertainty != "dose":
            self.calculate_combined_uncertainty()
        
        
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
        export_format, 
        dir_export, 
        content_to_export:list):
        r"""
        Purpose: 
            - To export the treatment plan file into a given export_format.
            The export_format can be either "RapidBrachyExport" or "WebAppExport".
        Inputs:
            - export_format := the export_format of the exported plan. options are:
                - "RapidBrachyExport":
                    "run_#.3ddose",
                    "dwell_#.plan",
                    "run_#.mac", 
                    "ApplicatorMaterials"
                    "applicator_geometry.json"
                    "catheter_table.json"
                    "structure_set.json"
                     
                - "WebApp": Not implemented yet
                    "run_#.nrrd",
                    "dwell_#.json",
                    "run_#.json",

            - dir_export := the directory to which the plan will be exported.
            - content_to_export := a list of strings specifying the content to export.
            options are: dose, uncertainty, catheter_table, applicator_geometry, 
            simulation setup, structure_set. 
        
        Outputs:
            - Void := will export the available parts of a plan into the specified export_format. 
        """
        assert os.path.exists(dir_export), \
            "export directory does not exist. please make the directory first"

        if export_format =="WebApp":
            self.export_to_webapp(dir_export, content_to_export)
            
        elif export_format =="RapidBrachyExport":
            self.export_to_rapidbrachy(dir_export, content_to_export)
            
    def export_to_webapp(self, dir_export):
        r"""
        Purpose: 
            - To export the treatment plan file into the WebApp export_format.
        Inputs:
            - dir_export := the directory to which the plan will be exported.
            
        Outputs:
            - Void := will export the available parts of a plan into the specified export_format. 
        """
        raise NotImplementedError("export to WebApp is not implemented yet")
    
    def export_to_rapidbrachy(self, dir_export):
        r"""
        Purpose:
            - To export the treatment plan file into dir_export with the format of 
            "RapidBrachyExport", which has the following files:
                    "run_#.3ddose",                # Optional
                    "dwell_#.plan",                # Required
                    "run_#.mac",                   # Optional
                    "ApplicatorMaterials"          # Optional
                    "applicator_geometry.json"     # Optional
                    "catheter_table.json"          # Required
                    "structure_set.json"           # Required
        """
    
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
    
if __name__ == "__main__":
    
    # running the test functions above: 
    # test_load_catheterTable_json()
    # test_extract_dwell_numbers_times_coordinates_from_catheterTable()
    # test_load_dose_rate_or_uncertainty_tensor()
    # test_set_dvh_metric_goals()
    # test_create_structures_and_calc_dvh_metrics()
    # test_calculate_combined_uncertainty()
    test_calculate_uncertainty_per_structure()
    # test_BrachyPlan()
    # test__load_single_dose_or_uncertainty_to_dict()