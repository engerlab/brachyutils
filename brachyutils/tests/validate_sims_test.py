import string
from typing import Any
import dose_utils
import os
import numpy as np
from dicompylercore import dose
import workplace
import scroll_dose
import glob
import matplotlib.pyplot as plt
import pymedphys
import pickle
import csv
import pandas as pd

from validate_sims import QA_TG186

def _test_QA_TG186_init(with_return=False):
        test_path = "/home/majd/data/TG186 Vallidation/rapidBrachyMCTPS/RapidBrachyMCTPS_Merged_DoseComparison/alana_newmuens_combined.3ddose"
        groundTruth_path = "/home/majd/data/TG186 Vallidation/rapidBrachyMCTPS/RapidBrachyMCTPS_Merged_DoseComparison/merged_combined.3ddose"
        qa_object = QA_TG186(test_path, groundTruth_path)
        if with_return:
                return qa_object

def _test_QA_TG186_dose_ratio():
        qa_object = _test_QA_TG186_init(True)
        qa_object.dose_ratio()
        print(qa_object.results_df)
        
def _test_QA_TG186_dose_percent_error():
        qa_object = _test_QA_TG186_init(True)
        qa_object.dose_percent_error()
        print(qa_object.results_df['mean_dose_ratio'])

def _test_QA_TG186_gamma_index():
        qa_object = _test_QA_TG186_init(True)
        qa_object.gamma_index()
        print(qa_object.results_df['mean_gamma_matrix'])
        print(qa_object.plot_results())
def _test_QA_TG186_run_QA():
        qa_object = _test_QA_TG186_init(True)
        qa_object.run_QA()

# a function to test the dose loading from .3ddose was successful
def testSimLoadSuccess(dose):
        try:
                print("Type of the input dose is: ", type(dose))
                print("-----------------")
                print("dimensions of the dose is:", np.shape(dose))
                print("-----------------")
                # print("the average uncertainty of the dose is:", dose_utils.get_average_uncert(dose))
                # print("-----------------")
                print("3ddose was loaded successfully")
                print("------------------")                
                return 1
        except:
                print("3ddose file did not load \n")
                print("-----------------")
                return 0
# a function to test the loading from dicom files
def testDicomLoadSuccess(dicom_object):
        try:
                print("Type of the input dose is: ", type(dicom_object))
                print("-----------------")
                print("here is the shape of the dose \n")
                print(dicom_object.shape)
                print("-----------------")
                print("-----------------")
                # {{for debugging only}}
                # print("here is the first few dose values \n")
                # print(dicom_object.dose_grid[0, 0, 98:101])
                # print("-----------------")
                return 1
        except:
                print("DICOM dose file did not load\n")
                print("-----------------")
                return 0

def _test_save_to_csv():
        a = np.arange(0, 12)
        b = np.arange(12, 24)
        c = np.arange(24, 36)
        file = "/home/majd/data/TG186 Vallidation/Elekta/Case1-OCB-MCNP6/simResults/test.csv"
        save_to_csv(file, a, b, c)

if __name__ =="__main__":
#      _test_QA_TG186_init()                            # test passed
        # _test_QA_TG186_dose_ratio()                   # test passed
        # _test_QA_TG186_dose_percent_error()           # test passed
        # _test_QA_TG186_gamma_index()                    # test passed but what does gamma mean???!!
        # _test_QA_TG186_run_QA()
        dose_file_3ddose = '/home/majd/data/TG186 Vallidation/Elekta/Case1-OCB-MCNP6/simResults/combined.3ddose'
        dose_file_dicom = "/home/majd/data/TG186 Vallidation/Elekta/Case1-OCB-MCNP6/dicom/RD_Case-1_MCNP6.dcm"
        qa_object = QA_TG186(dose_file_3ddose, dose_file_dicom)
        triPlanar_snapshot(qa_object.test_dose, "D_w(Source=Ir-192, t=10s)")