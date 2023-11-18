import os
from glob import glob
import json
import numpy as np

# from typing import Optional
from tqdm import tqdm
from dose_utils import BrachyDose

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
    name_in_gurobiModel:str
    bound_coordinates:list
    penalty_weight:float
    dvh_metric_name:str
    dvh_metric_clinical_goal:float
    dvh_metric_observed:float

    def __init__(self):
        pass

class BrachyPlan:
    r"""
    Purpose:
        - This class holds the information regarding the brachytherapy treatment plan
        as well as all the functions to support the necessary plan operations. 
    
    Attributes:
        - num_dwels:int
        - catheter_table:list := a list of catheter dictionaries. each catheter dictionary 
        contains the keys "dwells", "id", and points. the value belonging to the "dwells" key
        is a list of dwell position dictionary. The dwell position dictionary contains the keys: 
        "angle", "position", "relativePos", "rotation", "time", and "weight". for more info, 
        look at the function BrachyPlan.load_catheterTable_json() 
        - organ_bounds:dict
        - dose_rate_matrix:np.array := dose rate from dwell position 1 to n. the order 
        matches the dwell_number_list
        - brachy_structure:list[BrachyStructure] := the list of patient structures in the plan
    """
    num_dwells:int
    catheter_table:list
    dwell_numbers:np.array #shape: (num_dwells, 1)
    dwell_timess:np.array #shape: (num_dwells, 1)
    dwell_coordinates:list #shape: (num_dwells, 3) 
    organ_bounds:dict
    dose_rate_tensor:np.array #shape: (num_dwells, z, y, x)
    uncertainty_tensor:np.array #shape: (num_dwells, z, y, x)
    structure_set:list
    
    def __init__(self):
        self.num_dwells = None
        self.catheter_table = None
        self.dwell_numbers = np.array([], dtype=int)
        self.dwell_times = np.array([], dtype=np.float32)
        self.dwell_coordinates = []
        self.organ_bounds = None
        self.dose_rate_tensor = np.array([], dtype=np.float32)
        self.uncertainty_tensor = np.array([], dtype=np.float32)
        self.structure_set = None
    
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
        Inputs:
            - dir_dose_rate :=  path to the directory containing the dose rate files. we assume
            that the name of the dose rate files end as "run_1.nrrd", "run_2.nrrd", etc.
            - type_dose_file := the type of dose rate file. The type could be ".nrrd" or ".3ddose"
            consult BrachyDose in dose_utils.py for more info on the dose rate file types.
            - load_uncertainty := if True, the uncertainty tensor will be loaded as well
        Outputs:
            - Void := will update the BrachyPlan.dose_rate_tensor attribute
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
    dir_dose_rate = "../../data_test/plan_files/prostate-glen-p1-dose"

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
    
if __name__ == "__main__":
    
    # running the test functions above: 
    # test_load_catheterTable_json()
    # test_extract_dwell_numbers_times_coordinates_from_catheterTable()
    test_load_dose_rate_tensor()
    