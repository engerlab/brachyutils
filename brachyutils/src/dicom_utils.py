from numpy import array as nparray, zeros as npzeros, reshape
# from numpy import float as float
# from numpy import int as int
# from numpy import ma
# from numpy import dtype
import numpy as np
# import re
import os

# from dicompylercore import dicomparser
from glob import glob
# from numericalunits import cm, mm, kg, J
# Gy = J/kg

# import SimpleITK as sitk
# import difflib
# from typing import Optional
# from collections.abc import Iterable

# import pytest
# # import uu
# import lzma
# import pickle
# import pyzstd

# import typer
# import decimal

# from tqdm import tqdm

# from rt_utils import RTStructBuilder
from DicomRTTool.ReaderWriter import DicomReaderWriter#, ROIAssociationClass
# import pydicom

# import json

def get_structure_index_range(pth_dir_dicom:str, query_structure_list:list=["body"]) -> nparray:
    r"""
    Purpose:
        to find the index extent of the body voxels along each axis using dicom RT structure file. 
    Inputs:
        - pth_dir_dicom := path to the directory with the dicom files of a patient. 
            it should contain both images and RTSTRUCT file
    Outputs:
        - structure_index_range:np.array :=  a 3 x 2 array holding the min and max on x, y and axis
            [[x_min, x_max], [y_min, y_max], [z_min, z_max]],
        
        - body_mask_shape:np.array := 1 x 3 array holding the dimension of the original mask
            
    Dependencies:
        DicomRTTool: https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485
    """
    
    pth_dir_dicom = os.path.abspath(pth_dir_dicom)
    assert os.path.exists(pth_dir_dicom), "given dicom path does not exist"
    assert glob(pth_dir_dicom+"/*.dcm"), "there are no dicom files in this directory"
    
    output_dict = {}
    for query_structure_name in query_structure_list:

        # load the structure file into an rt_struct object
        dicom_reader = DicomReaderWriter(description=f"getting {query_structure_name} mask", arg_max=True)
        dicom_reader.walk_through_folders(pth_dir_dicom)
        all_rois = dicom_reader.return_rois()
        
        # # find the name of the body structure inside the rt_structure object
        dicom_structure_name = [name for name in all_rois if query_structure_name in name.lower()]
        
        # # get the numpy array of the body structure:
        assert len(dicom_structure_name) >= 1, "body contour not found!"
        dicom_reader.set_contour_names_and_associations(contour_names=dicom_structure_name)
        
        dicom_reader.get_mask()
        mask_numpy = dicom_reader.mask
        
        # so we got the mask but the dimensions may not match the dimension of the dose
        # let's get the relative extent of the body mask compared to the whole grid and resample
        # the extents
        structure_index_range = np.zeros([3, 2], dtype=int)
        for i in range(3):
            structure_index_range[i, :] = np.floor(np.array([
                np.argwhere(mask_numpy==1)[:, i].min(), 
                # off set of +1 is added to acount for python stopping before range end
                np.argwhere(mask_numpy==1)[:, i].max()+1])).astype(int)
                # np.argwhere(mask_numpy==1)[:, i].max()+1]) / np.array(mask_numpy.shape[i]) * self.num_voxels[3-i-1]).astype(int)

        structure_index_range = np.flip(structure_index_range, axis=0)        
        output_dict[query_structure_name] = {"structure_index_range":structure_index_range, "dicom_mask_shape":np.flip(np.array(mask_numpy.shape))}
            
    return output_dict

def test_get_structure_index_range():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    # pth_3ddose = "../../data_test/run_1_glen_prostate_p1.3ddose"
    print(get_structure_index_range(pth_dicomRS, ['body', 'urethra_brachy', 'rectum_brachy', 'ctv_brachy']))
    # print(get_structure_index_range(pth_dicomRS, ['body']))
    

if __name__ == "__main__":
    # app()
    # a Test for the following functions
    test_get_structure_index_range()
    # test_get_body_contour_range_from_many_patients_dicom()