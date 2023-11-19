import os
from glob import glob
import json
import numpy as np

# from typing import Optional
from tqdm import tqdm
from dose_utils import BrachyDose
from copy import deepcopy

from dicom_utils import get_strcuture_mask_from_dicom
from scipy import ndimage

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
    name:str
    mask:np.array # shape: (z, y, x)

    dvh_metric_name:str
    dvh_metric_clinical_goal:float
    dvh_metric_observed:float

    name_in_gurobiModel:str
    bound_coordinates_in_gurobiModel:list
    penalty_weight:float
    

    def __init__(self):
        pass

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
        - dwell_coordinates:list := a list of dictionaries. each dictionary contains the keys "position", "rotation", and "relativePos"
        - organ_bounds:dict
        - dose_rate_tensor:np.array := dose rate from dwell position 1 to num_dwells. 
        matches the dwell_number_list. shape: (num_dwells, z, y, x)
        - uncertainty_tensor:np.array := uncertainty from dwell position 1 to num_dwells. shape: (num_dwells, z, y, x)
        - brachy_structure:list[BrachyStructure] := the list of patient structures in the plan
    """
    num_dwells:int
    catheter_table:list
    dwell_numbers:np.array #shape: (num_dwells, 1)
    dwell_timess:np.array #shape: (num_dwells, 1)
    dwell_coordinates:list #shape: (num_dwells, 3) 
    dose_rate_tensor:np.array #shape: (num_dwells, z, y, x)
    combined_dose:BrachyDose
    uncertainty_tensor:np.array #shape: (num_dwells, z, y, x)

    # organ_bounds:dict
    dvh_metric_goals: list
    structure_list:list

    
    def __init__(self):
        self.num_dwells = None
        self.catheter_table = None
        self.dwell_numbers = np.array([], dtype=int)
        self.dwell_times = np.array([], dtype=np.float32)
        self.dwell_coordinates = []

        # self.organ_bounds = None
        self.dose_rate_tensor = np.array([], dtype=np.float32)
        self.uncertainty_tensor = np.array([], dtype=np.float32)
        self.combined_dose = None

        dvh_metric_goals = None
        self.structure_list = []
    
    def load_catheterTable_json(
        self, 
        pth_catheterTable_json:str, 
        drop_lastDwell_perCatheter:bool=False):
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
        
        # there is currently a bug in the tps where the last dwell position is repeated.
        # this block will fixe it {
        if drop_lastDwell_perCatheter:
            corrected_catheter_table = []
            for catheter in catheter_table:
                dwell_list = catheter['dwells']
                corrected_catheter_table.append(
                    {
                        'dwells': dwell_list[:-1],
                        "id": catheter["id"],
                        "points": catheter["points"]
                    })
            self.catheter_table = corrected_catheter_table
        # }
        else:
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

    def load_dose_rate_tensor(
        self, 
        dir_dose_rate:str,
        type_dose_file:str=".nrrd",
        load_uncertainty:bool=False,):
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
            - load_uncertainty := if True, the uncertainty tensor will be loaded as well
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
        assert len(dose_rate_files) == self.num_dwells, "number of dose rate files does not match the number of dwell positions"

        test_dose_obj = BrachyDose(dose_rate_files[0])
        self.dose_rate_tensor = np.zeros((self.num_dwells, *test_dose_obj.grid.shape), dtype=np.float32)
        self.uncertainty_tensor = np.zeros((self.num_dwells, *test_dose_obj.uncertainty.shape), dtype=np.float32)
        # load the dose rate tensor   
        for i, dwell_num in tqdm(zip(range(self.num_dwells), self.dwell_numbers)):
            # find the dose rate file corresponding to this dwell number
            query_string = f"run_{int(dwell_num)}{type_dose_file}"
            pth_dose_rate = list(filter(lambda x: query_string in x, dose_rate_files))[0]
            dose_obj = BrachyDose(pth_dose_rate)
            self.dose_rate_tensor[i] = dose_obj.grid
            if load_uncertainty:
                self.uncertainty_tensor[i] = dose_obj.uncertainty
        
        # calculate the combined dose and store the result in the combined_dose attribute 
        combined_dose_grid = np.sum(
            self.dose_rate_tensor * self.dwell_times[:, np.newaxis, np.newaxis, np.newaxis],
            axis=0)
        self.combined_dose = BrachyDose()
        self.combined_dose.grid = combined_dose_grid
        self.combined_dose.num_voxels = test_dose_obj.num_voxels        
        self.combined_dose.vox_size = test_dose_obj.vox_size
        self.combined_dose.topleft = test_dose_obj.topleft
        self.combined_dose.calculate_voxel_edges()
        
        assert np.array_equal(
            np.concatenate(self.combined_dose.voxel_edges), 
            np.concatenate(test_dose_obj.voxel_edges)), \
            "voxel edges of combined dose map and dwell dose rate map do not match"

        assert self.combined_dose.is_not_empty(), "combined dose is empty"

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
        size_uncropped_dose_grid: np.array):
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

        structure_name_list = []
        for dvh_metric in self.dvh_metric_goals:
            structure_obj = BrachyStructure()
            structure_obj.name = dvh_metric.split("(")[-1].split(")")[0]
            structure_name_list.append(structure_obj.name)
            structure_obj.dvh_metric_name = dvh_metric.split("(")[0]
            structure_obj.dvh_metric_clinical_goal = self.dvh_metric_goals[dvh_metric]
            self.structure_list.append(structure_obj)

        # load the structure mask
        structure_mask_dict = load_structure_mask(dir_structures_source, structure_name_list)
        for structure in self.structure_list:
            original_mask = structure_mask_dict[structure.name]
            structure.mask = ndimage.zoom(original_mask, size_uncropped_dose_grid/original_mask.shape, order=0)
            # print(structure.mask.shape)
            # print(size_uncropped_dose_grid)
            # print(self.combined_dose.grid.shape)

    def measure_DVH_metrics(self):
            r"""
            Purpose:
                - To get the observed value of the dvh metric for each structure in the BrachyPlan.
                the observed value is calculated from the combined dose map.
            Inputs:
                - self := the BrachyPlan object
            Outputs:
                - Void := will update the BrachyStructure.dvh_metric_observed attribute
            """
            raise NotImplementedError("this function is not implemented yet")
            # for structure_obj in self.structure_list:
            #     structure_obj.dvh_metric_observed = self.combined_dose.get_dvh_metric(
            #         structure_obj.name, 
            #         structure_obj.dvh_metric_name)


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


def test_load_catheterTable_json():
    pth_cathTable_json = "../../data_test/plan_files/optimized_plan_ctv/catheter_table.json"
    
    with open(pth_cathTable_json, 'r') as json_file:
        ground_truth_catheter_table = json.load(json_file)
    
    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)

    # print(plan_obj.catheter_table)
    assert [i for i in ground_truth_catheter_table if i not in plan_obj.catheter_table] ==[],\
        "loading catheter table did not work as expected"

def test_extract_dwell_numbers_times_coordinates_from_catheterTable():
    pth_cathTable_json = "../../data_test/plan_files/optimized_plan_ctv/catheter_table.json"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    plan_obj.extract_dwell_numbers_times_coordinates_from_catheterTable()
    
    assert plan_obj.dwell_numbers is not None, "dwell numbers not extracted"
    assert plan_obj.dwell_times is not None, "dwell times not extracted"
    assert plan_obj.dwell_coordinates is not None, "dwell coordinates not extracted"
    
    print(f"The shape of the dwell_number is {plan_obj.dwell_numbers.shape}")
    print(f"The shape of the dwell_times is {plan_obj.dwell_times.shape}")
    print(f"The shape of the dwell_coordinates is {len(plan_obj.dwell_coordinates)}")

def test_load_dose_rate_tensor():
    pth_cathTable_json = "../../data_test/plan_files/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    plan_obj.extract_dwell_numbers_times_coordinates_from_catheterTable()

    plan_obj.load_dose_rate_tensor(dir_dose_rate, load_uncertainty=True)
    print(f"The shape of the dose rate tensor is {plan_obj.dose_rate_tensor.shape}")
    print(f"The shape of the uncertainty tensor is {plan_obj.uncertainty_tensor.shape}")
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

def test_create_structures():
    dir_dicom = "../../data_test/prostate-glen-p1-dcm/"
    pth_cathTable_json = "../../data_test/plan_files/optimized_plan_ctv/catheter_table.json"
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose"
    dvh_metric_goals = {
        'D95%(ctv)': 15,
        'D1cc(rectum)': 11.25,
        'D0.1cc(urethra)': 18.75
    }

    plan_obj = BrachyPlan()
    plan_obj.load_catheterTable_json(pth_cathTable_json)
    plan_obj.extract_dwell_numbers_times_coordinates_from_catheterTable()
    plan_obj.load_dose_rate_tensor(dir_dose_rate, load_uncertainty=True)
    plan_obj.set_dvh_metric_goals(dvh_metric_goals)

    plan_obj.create_structures(dir_dicom, np.array([126, 380, 380]))

if __name__ == "__main__":
    
    # running the test functions above: 
    # test_load_catheterTable_json()
    # test_extract_dwell_numbers_times_coordinates_from_catheterTable()
    # test_load_dose_rate_tensor()
    # test_set_dvh_metric_goals()
    test_create_structures()
    