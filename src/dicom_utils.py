from numpy import array as nparray, zeros as npzeros, reshape
# from numpy import float as float
# from numpy import int as int
from numpy import ma
from numpy import dtype
import numpy as np
import re
import os

# from dicompylercore import dicomparser
from glob import glob
# from numericalunits import cm, mm, kg, J
# Gy = J/kg

import SimpleITK as sitk
import difflib
from typing import Optional
from collections.abc import Iterable

import pytest
# import uu
import lzma
import pickle
import pyzstd

import typer
import decimal

from tqdm import tqdm

# from rt_utils import RTStructBuilder
from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
import pydicom

import json

def get_body_index_range(pth_dir_dicom:str):
    r"""
    Purpose:
        to find the index extent of the body voxels along each axis using dicom RT structure file. 
    Inputs:
        - pth_dir_dicom := path to the directory with the dicom files of a patient. 
            it should contain both images and RTSTRUCT file
    Outputs:
        - body_index_range:np.array :=  a 3 x 2 array holding the min and max on x, y and axis
            [[x_min, x_max], [y_min, y_max], [z_min, z_max]],
        
        - body_mask_shape:np.array := 1 x 3 array holding the dimension of the original mask
            
    Dependencies:
        DicomRTTool: https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485
    """
    
    pth_dir_dicom = os.path.abspath(pth_dir_dicom)
    assert os.path.exists(pth_dir_dicom), "given dicom path does not exist"
    assert not not glob(pth_dir_dicom+"/*.dcm"), "there are no dicom files in this directory"
            
    pth_structure_dcm = glob(pth_dir_dicom+"/RS*.dcm")[0]
    
    # load the structure file into an rt_struct object
    dicom_reader = DicomReaderWriter(description="getting body mask", arg_max=True)
    dicom_reader.walk_through_folders(pth_dir_dicom)
    all_rois = dicom_reader.return_rois()
    
    # # find the name of the body structure inside the rt_structure object
    body_structure_name = [name for name in all_rois if "body" in name.lower()]
    
    # # get the numpy array of the body structure:
    assert len(body_structure_name) == 1, "body contour not found!"
    dicom_reader.set_contour_names_and_associations(contour_names=body_structure_name)
    
    dicom_reader.get_mask()
    mask_numpy = dicom_reader.mask
    
    # so we got the mask but the dimensions may not match the dimension of the dose
    # let's get the relative extent of the body mask compared to the whole grid and resample
    # the extents
    body_index_range = np.zeros([3, 2], dtype=int)
    for i in range(3):
        body_index_range[i, :] = np.floor(np.array([
            np.argwhere(mask_numpy==1)[:, i].min(), 
            # off set of +1 is added to acount for python stopping before range end
            np.argwhere(mask_numpy==1)[:, i].max()+1])).astype(int)
            # np.argwhere(mask_numpy==1)[:, i].max()+1]) / np.array(mask_numpy.shape[i]) * self.num_voxels[3-i-1]).astype(int)
        
    body_index_range = np.flip(body_index_range, axis=0)    
            
    return body_index_range, np.flip(np.array(mask_numpy.shape))

app = typer.Typer()

@app.command()
def get_body_contour_range_from_many_patients_dicom(
    input_dir:str, 
    pth_output_json:str 
):
    r"""
    Purpose:
        to exract body contour extent on each axis for all the patients in input_dir and save them
            to a json file located at "output_json"
    Input:
        - input_dir := path to the directory where folders of many patients with dicom files exist.
            this script will loop through patient folders
        - output_json := path to the json file where the following information for each patient is stored            
    Output: 
        - Void := the following content will be written to output_json for each patient:
            {
                patient_number:=str,
                body_index_range:[
                    [x_min:int, x_max:int],
                    [y_min:int, y_max:int],
                    [z_min:int, z_max:int],
                ]
                body_mask_shape:[len(x):int, len(y):int, len(z):int]
            }
    """
    
    input_dir = os.path.abspath(input_dir)
    
    patient_dir_list = glob(input_dir+"/*/")
    patient_dict_list = []
    
    for patient_dir in patient_dir_list:
        try:
            body_index_range , body_mask_shape = get_body_index_range(patient_dir)
            patient_dict_list.append(
            {
            "patient_number": patient_dir.split("/")[-2],
            "body_index_range": body_index_range.tolist(),
            "body_mask_shape": body_mask_shape.tolist()
            })
        except:
            print(f"WARNING: no body contour for patient {patient_dir}, moving on")
            # body_index_range , body_mask_shape = np.array([]), np.array([])
        
    
    json_object = json.dumps(patient_dict_list, indent=4)
    with open(pth_output_json, "w") as outfile:
        outfile.write(json_object)


def test_get_body_index_range():
    pth_dicomRS = "../data_test/prostate_glen_p1/"
    pth_3ddose = "../data_test/run_1_glen_prostate_p1.3ddose"


    print(get_body_index_range(pth_dicomRS))
    

def test_get_body_contour_range_from_many_patients_dicom():
    input_dir = "../data_test"
    pth_json = "../data_test/test_patient_body_bounds.json"
    
    get_body_contour_range_from_many_patients_dicom(input_dir, pth_json)
    
    with open(pth_json, "r") as file:
        data_json = json.load(file)
    
    print(data_json)

if __name__ == "__main__":
    
    app()

    # a Test for the following functions
    # test_get_body_index_range()
    # test_get_body_contour_range_from_many_patients_dicom()