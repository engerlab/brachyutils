import os
import sys
import difflib
from typing import Optional
import lzma
import pickle
import logging
import pickle
import tkinter as tk
from glob import glob
from tkinter import filedialog as fd

import pymedphys
from numpy import array as nparray, zeros as npzeros, reshape
from numpy import ma
from numpy import dtype
import numpy as np
# from PyQt5.QtCore import QFile, QDataStream, QIODevice, QList
# from PyQt5.Qt3DExtras import QVector3D
from matplotlib import pyplot as plt
# from numpy import float as float
# from numpy import int as int
from scipy.interpolate import RegularGridInterpolator
import re
import os
from array import array

# from dicompylercore import dicomparser
# from numericalunits import cm, mm, kg, J
# Gy = J/kg

import SimpleITK as sitk

# from collections.abc import Iterable

# import pytest
# import uu

import pyzstd

# import typer
# import decimal

# from tqdm import tqdm

from dicom_utils import get_structure_index_range

# from rt_utils import RTStructBuilder
# from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
# import pydicom
# import json
# import numpy as np
from typing import List
# so_true = True

import json

from dose_utils import BrachyDose
from dose_utils import DoseComparison


def test_load_from_3ddose():
    # pth_3ddose =  "../../data_test/run_1_old.3ddose"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"

    dose_obj = BrachyDose()
    dose_obj.load_from_3ddose(pth_3ddose)
    dose_obj.is_not_empty()


def test_load_file_to_brachydose():
    # pth_3ddose =  "../../data_test/run_1_old.3ddose"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"

    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.is_not_empty()
# @pytest.mark.passed


def test_write_to_3ddose():
    # pth_3ddose =  "../../data_test/run_1_old.3ddose"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'_test.3ddose'

    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_3ddose(pth_out)
    new_dose_obj = BrachyDose().load_file_to_brachydose(pth_out)
    dose_obj.is_equal(new_dose_obj)
# @pytest.mark.passed


def test_convert_to_nrrd():
    r"""
    Purpose: 
        simulatenously test write_to_nrrd() and load_from_nrrd()
    """
    # 3 mm resolution
    # pth_3ddose =  "../../data_test/run_1_old.3ddose"
    # pth_nrrd = "../../data_test/run_1_old.nrrd"
#
    # 1 mm resolution
    # pth_3ddose =  "../../data_test/combined.3ddose"
    # pth_nrrd = "../../data_test/combined_old.nrrd"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.nrrd'

    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_nrrd(pth_out)

    dose_obj_from_nrrd = BrachyDose()
    dose_obj_from_nrrd.load_file_to_brachydose(pth_out)

    dose_obj.is_equal(dose_obj_from_nrrd)


def test_convert_to_npz_file():
    r"""
    Purpose: 
        simulatenously test write_to_npz() and load_from_npz()
    """
    # pth_3ddose =  "../../data_test/combined.3ddose"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.npz'
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_npz(pth_out)

    new_dose_obj = BrachyDose()
    new_dose_obj.load_from_npz(pth_out)
    dose_obj.is_equal(new_dose_obj)

# def test_write_to_minidos():
#     r"""
#     Purpose:
#         simulatenously test write_to_minidos() and load_from_minidos()
#     """
#     # pth_3ddose =  "../../data_test/combined.3ddose"

#     # testing on maude's file
#     pth_3ddose = "../../data_test/maude.3ddose"
#     pth_out = os.path.splitext(pth_3ddose)[0]+'.minidos'
#     dose_obj = BrachyDose()
#     dose_obj.load_file_to_brachydose(pth_3ddose)

#     dose_obj.write_to_minidos(pth_out, compress_program='zstd')

#     new_dose_obj = BrachyDose()
    # new_dose_obj.load_from_minidos(pth_out)
    # dose_obj.is_equal(new_dose_obj)


def test_write_to_xz():

    # pth_3ddose =  "../../data_test/combined.3ddose"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.xz'
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_xz(pth_out)


def test_write_to_zstd():

    # pth_3ddose =  "../../data_test/combined.3ddose"
    # pth_zstd = "../../data_test/combined.zst"

    # testing on maude's file
    pth_3ddose = "../../data_test/maude.3ddose"
    pth_out = os.path.splitext(pth_3ddose)[0]+'.zst'
    print(pth_out)
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)

    dose_obj.write_to_zstd(pth_out)


def test_crop_by_coordinates():
    pth_3ddose = "../../data_test/3ddose/p1/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.info()

    coords = np.array([
        [-14, 8],
        [3, 15],
        [-115, -100]], dtype=np.float32)

    dose_obj.crop_by_coordinates(coords)
    dose_obj.info()


def test_crop_by_index():
    pth_3ddose = "../../data_test/3ddose/p1/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.info()

    index = np.array([
        [30, 90],
        [30, 90],
        [0, 94]], dtype=np.float32)

    dose_obj.crop_by_index(index)

    dose_obj.info()


def test_crop_by_fraction():
    pth_3ddose = "../../data_test/3ddose/p1/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.info()

    fraction = 0.3

    dose_obj.crop_by_fraction(fraction)
    dose_obj.info()


def test_get_structure_index_range():
    pth_dicom_rs = "../../data_test/prostate-glen-p1-dcm/"
    list_query_rs = ['body', 'ctv_brachy', 'rectum_brachy', 'urethra_brachy']
    #pth_3ddose = "../../data_test/run_1_glen_prostate_p1.3ddose"
    print(get_structure_index_range(pth_dicom_rs, list_query_rs))


def test_convert_to_minidos():
    pth_input = "../../data_test/dwell1_1mm.nrrd"
    pth_minidos = os.path.splitext(pth_input)[0] + ".minidos"

def test_dose_comparison():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    pth_3ddose = "../../data_test/3ddose/p1/run_1_old.3ddose"
    pth_3ddose2 = "../../data_test/3ddose/p1/run_1_old.3ddose"
    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj2 = BrachyDose()
    dose_obj2.load_file_to_brachydose(pth_3ddose2)
    dose_comparison = DoseComparison(dose_obj, dose_obj2, 1, 1)
    # evaluate that the grid contains only 0
    assert (not np.any(dose_comparison.percent_difference.grid))
    # dose_comparison.compare_dose_distributions_2D(
    #    dose_obj.voxel_edges[2], dose_obj.voxel_edges[1], dose_obj.voxel_edges[0][0], 'z')

def test_crop_by_body_contour():
    pth_dicomRS = "../../data_test/prostate-glen-p1-dcm/"
    pth_3ddose = "../../data_test/3ddose/p1/run_1_glen_prostate_p1.3ddose"

    dose_obj = BrachyDose()
    dose_obj.load_file_to_brachydose(pth_3ddose)
    dose_obj.info()
    dose_obj.crop_by_body_contour(pth_dir_dicom=pth_dicomRS)