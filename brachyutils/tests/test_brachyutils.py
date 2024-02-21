from glob import glob
import json
import os
import numpy as np
from tqdm import tqdm
import pytest
import typer

from dicom_utils import  get_structure_index_range
from egsphant_utils import _load_json, BrachyEgsphant
from dose_utils import BrachyDose
from typing import Optional
from typing_extensions import Annotated

from brachyutils import get_body_contour_range_from_dicom_many_patients
from brachyutils import crop_dose_by_body_contour_many_files
from brachyutils import convert_dose_many_files
from brachyutils import crop_egsphant_by_body_contour_many_patients
from brachyutils import combined_dose_per_patient

from multiprocessing import Pool
from functools import partial

def test_get_body_contour_range_from_dicom_many_patients():
    input_dir = "../../data_test/prostate-glen-p1-dcm/"
    pth_json = "../../data_test/patient_body_bounds_output.json"

    get_body_contour_range_from_dicom_many_patients(input_dir, pth_json)

    with open(pth_json, "r") as file:
        data_json = json.load(file)

    print(data_json)

def test_crop_egsphant_by_body_contour_many_patients():
    # test on testing dataset
    pth_input = "../../data_test/prostate-glen-p1-planFiles/"
    pth_json = "../../data_test/test_patient_body_bounds.json"
    crop_egsphant_by_body_contour_many_patients(pth_input, pth_json)

def test_convert_many_files():
    dir_in = "../../data_test/many_files/"
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


def test_crop_dose_by_body_contour_many_files():
    pth_3ddose = "../../data_test/3ddose/p1"
    pth_json = "../../data_test/patient_body_bounds.json"

    crop_dose_by_body_contour_many_files(pth_3ddose, pth_json)

def test_combined_dose_per_patient():
    batch_directory = "../../data_test/batch_uncertainty_test/"
    combined_dose_per_patient(batch_directory, ".3ddose", ".3ddose", multi_proc=True)
    mp_combined = BrachyDose(f"{batch_directory}/combined.3ddose")
    combined_dose_per_patient(batch_directory, ".3ddose", ".3ddose", multi_proc=False)
    nmp_combined = BrachyDose(f"{batch_directory}/combined.3ddose")
    assert np.allclose(mp_combined.grid, nmp_combined.grid), "Combined dose grids should \
         be equal regardless of whether multiprocessing is used"
    assert np.allclose(mp_combined.uncertainty, nmp_combined.uncertainty), "Combined uncertainty grids should \
         be equal regardless of whether multiprocessing is used"
    assert np.all(mp_combined.grid - 1  < 0.1), "Mean dose should be close to the \
        mean of the individual doses"
    assert np.all(mp_combined.uncertainty < 0.1), "Mean uncertainty should be far less \
        than the uncertainty of the individual doses"