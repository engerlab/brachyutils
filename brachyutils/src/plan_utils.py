import os
from glob import glob
import json

from typing import Optional

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
        - brachy_structure:BrachyStructure := a patient structure 
    """
    num_dwels:int
    catheter_table:list
    # dwell_number_list:np.list
    # dwell_times_list:np.list
    # dwell_coordinates:np.list
    organ_bounds:dict
    dose_rate_matrix:np.array
    brachy_structure:BrachyStructure
    
    def __init__(self):
        pass
    
    def load_catheterTable_json(self, pth_catheterTable_json:str):
        r"""
        Purpose:
            - To load the contents of a catheter table into the Brachy plan.
        Inputs:
            - pth_catheterTable_json := path to a json file having the info on the catheter table. 
            here is the expected contents of the catheter table json:
            [
                {
                    "dwells":[
                        "position":{
                            "x",
                            "y",
                            "z"
                        },
                        "relativePos":= dwell coordinate along the catheter from the reference point. increments of 5 mm
                        "rotation": {
                            "x",
                            "y",
                            "z"
                        },
                        "time" := dwell time for this dwell position
                        "weight" := ratio of this dwell time over the sum of all dwell times in all catheters
                    ],
                    "id":= the id of the caheter,
                    "points":[] := i do not know what this is. in all plans i have seen, it has been lefty empty
                }
            ]

        """
if __name__ == "__main__":
    return 0