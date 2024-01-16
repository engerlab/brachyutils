from numpy import array as nparray, zeros as npzeros, reshape
# from numpy import float as float
# from numpy import int as int
from numpy import ma
from numpy import dtype
import numpy as np
import re
import os

# from dicompylercore import dicomparser
# from glob import glob
# from numericalunits import cm, mm, kg, J
# Gy = J/kg

# import SimpleITK as sitk
# import difflib
from typing import Optional
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
# from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
# import pydicom

import json

# from dicom_utils import get_body_index_range

from egspgant_utils import BrachyDose

def test_crop_by_body_contour():
    pth_input = "../../data_test/glen_prostate_p1_3mm_ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)
    pth_dicomRS = "../../data_test/prostate_glen_p1/"

    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.info()
    
    egsphant_obj.crop_by_body_contour(pth_dicomRS)
    egsphant_obj.info()
    
    

def test_crop_by_index():
    pth_input = "../../data_test/glen_prostate_p1_3mm_ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)
    
    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.info()
    
    index=np.array([
        [30, 90],
        [30, 90],
        [0, 94]],dtype=np.float32)

    egsphant_obj.crop_by_index(index)
    egsphant_obj.info()
    egsphant_obj.write_to_ctegsphant(pth_output)

def test_write_to_egsphant():
    pth_input = "../../data_test/glen_prostate_p1_3mm_ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)
    
    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()
    
    egsphant_obj.write_to_ctegsphant(pth_output)
    new_egsphant_obj = BrachyEgsphant()
    new_egsphant_obj.load_from_ctegsphant(pth_output)
    
    egsphant_obj.is_equal(new_egsphant_obj)

def test_to_single_string():
    pth_input = "../../data_test/glen_prostate_p1_3mm_ct.egsphant"
    pth_output = os.path.dirname(pth_input) + "/test_"+os.path.basename(pth_input)
    
    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()
    
    _to_single_string(egsphant_obj.material_matrix.astype(str))     

def test_load_from_ctegsphant():
    pth_input = "../../data_test/glen_prostate_p1_3mm_ct.egsphant"
    
    egsphant_obj = BrachyEgsphant()
    egsphant_obj.load_from_ctegsphant(pth_input)
    egsphant_obj.assert_BrachyEgsphant_notEmpty()

# if __name__=="__main__":
#     app()
    # running tests top is the latest test written
    # test_crop_by_body_contour_many_files()
    # test_crop_by_body_contour()
    # test_crop_by_index()
    # test_to_single_string()
    # test_write_to_egsphant()
    # test_load_from_ctegsphant()