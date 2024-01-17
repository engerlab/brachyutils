import numpy as np
import os

# from dicompylercore import dicomparser
from glob import glob

from DicomRTTool.ReaderWriter import DicomReaderWriter#, ROIAssociationClass

import json

from dicom_utils import get_structure_index_range
from dicom_utils import get_strcuture_mask_from_dicom

def test_get_structure_index_range():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    # pth_3ddose = "../../data_test/run_1_glen_prostate_p1.3ddose"
    print(get_structure_index_range(pth_dicomRS, ['body', 'urethra', 'rectum', 'ctv']))
    # print(get_structure_index_range(pth_dicomRS, ['body']))

def test_get_strcuture_mask_from_dicom():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    get_strcuture_mask_from_dicom(pth_dicomRS, ['urethra', 'rectum', 'ctv'])

if __name__ == "__main__":
    # app()
    # a Test for the following functions
    # test_get_structure_index_range()
    test_get_strcuture_mask_from_dicom()