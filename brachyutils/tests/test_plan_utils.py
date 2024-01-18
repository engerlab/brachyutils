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

#from plan_utils import BrachyStructure
from plan_utils import BrachyPlan
from plan_utils import _load_single_dose_or_uncertainty_to_dict

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
    dir_dose_rate = "../../data_test/prostate-glen-p1-dose/"
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
    dose_rate_dict = _load_single_dose_or_uncertainty_to_dict(pth_dose_rate, "both")
    print(dose_rate_dict[0].shape) #dose
    print(dose_rate_dict[1].shape) #uncertainty
    
if __name__ == "__main__":
    
    # running the test functions above: 
    # test_load_catheterTable_json() #passes
    # test_extract_dwell_numbers_times_coordinates_from_catheterTable() #passes
    # test_load_dose_rate_or_uncertainty_tensor() #TENSOR ISSUE!!!
    # test_set_dvh_metric_goals() #passes
    # test_create_structures_and_calc_dvh_metrics() #TENSOR ISSUE!!!
    # test_calculate_combined_uncertainty() #TENSOR ISSUE!!!!
    # test_calculate_uncertainty_per_structure() #Killed
    # test_BrachyPlan() #Killed
    # test__load_single_dose_or_uncertainty_to_dict() #passes
