from glob import glob
import json
import os
from tqdm import tqdm
import pytest
import typer

from dicom_utils import  get_structure_index_range
from egsphant_utils import _load_json, BrachyEgsphant
from dose_utils import BrachyDose
from typing import Optional
from typing_extensions import Annotated

from brachyutils import get_bodyContourRange_from_dicom_many_patients
from brachyutils import crop_dose_by_bodyContour_many_files
from brachyutils import convert_dose_many_files
from brachyutils import crop_egsphant_by_bodyContour_many_patients

from multiprocessing import Pool
from functools import partial
#app = typer.Typer()

def test_get_bodyContourRange_from_dicom_many_patients():
    input_dir = "../../data_test/prostate-glen-p1-dcm/"
    pth_json = "../../data_test/test_patient_body_bounds.json"
    
    get_bodyContourRange_from_dicom_many_patients(input_dir, pth_json)
    
    with open(pth_json, "r") as file:
        data_json = json.load(file)
    
    print(data_json)

def test_crop_egsphant_by_bodyContour_many_patients():
    # test on testing dataset
    pth_input = "../../data_test"
    pth_json = "../../data_test/patient_body_bounds.json"
    
    # test on all patients
    # pth_input = "/home/majd/data/patient_dose_simulations/prostate-glen-1mm"
    # pth_json = "../../data_test/patient_body_bounds.json"
    
    crop_egsphant_by_bodyContour_many_patients(pth_input, pth_json)

def test_convert_many_files():
    dir_in = "../../data_test/many_files/"
    #dir_in = "../../data_test/prostate-glen-p1-dose/"
    type_in = ".nrrd"
    type_out = ".minidos"
    
    convert_dose_many_files(dir_in, type_in, type_out)
    
    dir_in = os.path.abspath(dir_in)
    nrrd_list = glob(dir_in+".nrrd")
    
    
    for file_nrrd in nrrd_list:
        dose_obj_nrrd = BrachyDose()
        dose_obj_nrrd.load_file_to_brachydose(file_nrrd)
        
        file_3ddose = os.path.splitext(file_nrrd)[0]+".3ddose"
        dose_obj_3ddose = BrachyDose()
        dose_obj_3ddose.load_file_to_brachydose(file_3ddose)
        
        dose_obj_3ddose.is_equal(dose_obj_nrrd)

def test_crop_by_body_contour():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    pth_3ddose = "../../data_test/run_1_glen_prostate_p1.3ddose"

    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.info()
    dose_obj.crop_by_body_contour(pth_dicomRS)
    dose_obj.info()

def test_crop_dose_by_bodyContour_many_files():
    # pth_3ddose = "/home/majd/data/patient_dose_simulations/prostate-glen-1mm/p3/run_29.3ddose"
    pth_3ddose = "../../data_test/run_1_glen_prostate_p1.3ddose"
    pth_json = "../../data_test/patient_body_bounds.json"
    
    crop_dose_by_bodyContour_many_files(pth_3ddose, pth_json)

#def main():
#    app()

if __name__ == "__main__":
    # test_convert_many_files()
    test_crop_dose_by_bodyContour_many_files()