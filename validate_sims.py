"""
Date 
        2021/9/9
Purpose
        To compare the dose that was simulated according to TG-186 by MCTPS to the dose that was presented by AAPM as ground truth. 
        This script will loop through the AAPM cases, and validates our dose for each case to the ground truth. 
Author
        Hossein Jafarzadeh
        Enger Lab
        McGill University
Inputs
        a mother directory that contains the dicom files and simulated dose for each case. Shown below:
                -mother_dir
                        |-Case1/
                                |-dicom
                                        RD-...-.dcm
                                |-simResults 
                                        combined.3ddose
                        |-Case2/
                        |-Case3/
                        |etc...
                
Dependencies
        The following external packages:
                1. numpy
                2. glob
                3. matplotlib.pyplot
                4. dicompylercore (needs installation!)

        The following internal packages:
                1. workplace
                2. scroll_dose
                3. dose_utils
Outputs
        1. terminal outputs saying dose loading was successful
        2. two scrollable plots of the dose matrix; one for ground truth dose and one for our simulated dose
        3. result of gamma index analysis (to be added) 
"""

# needed libraries
import dose_utils
import os
import numpy as np
from dicompylercore import dose
import workplace
import scroll_dose
import glob
import matplotlib.pyplot as plt



# a function to test the dose loading from .3ddose was successful
def testSimLoadSuccess(dose):
        try:
                print("Type of the dose is: ", type(dose["grid"]))
                print("-----------------")
                print("dimensions of the dose is:", np.shape(dose["grid"]))
                print("-----------------")
                print("the average uncertainty of the dose is:", dose_utils.get_average_uncert(dose))
                print("-----------------")
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
                print("here is the shape of the dose \n")
                print(dicom_object.shape)
                print("-----------------")
                print("this is the type of the dicom_object:\n")
                print(type(dicom_object))
                print("-----------------")
                print("here is the first few dose values \n")
                print(dicom_object.dose_grid[0, 0, 98:101])
                print("-----------------")
                return 1
        except:
                print("DICOM dose file did not load\n")
                print("-----------------")
                return 0
     
# set directory of the project data
mother_dir = workplace._workplace(workplace.askForLocation())
# simfileDir = simfileDir + "/simResults"
print("------------------")
print("looking at the mother directory: \n")
print(mother_dir)
print("------------------")

# a for loop to iterate through dose files in the specified directory
for case in os.listdir(mother_dir):
        if "Case1" in case:
# let's load the ground truth from DICOM files
                dicomDoseFile = glob.glob(mother_dir + "/" + case + "/dicom/RD*")[0]
                print("looking at the DICOM file: \n")
                print(dicomDoseFile)
                print("------------------")
                dicom_object = dose.DoseGrid(dicomDoseFile)

# check if loading was successful
                testDicomLoadSuccess(dicom_object)
# plot the dicom dose file
                scroll_dose.plot_scrollable(dicom_object.dose_grid, "DICOM")
# obtain shape of dose in DICOM file to crop the dose in 3ddose file accordingly 
                dicom_shape = np.shape(dicom_object.dose_grid)
                print("here is the shape of the dicom dose file: \n", dicom_shape)
# ############################################

                simDoseFile = glob.glob(mother_dir + "/" + case + "/simResults/*.3ddose")[0]
                print("looking at the file: \n")
                print(simDoseFile)
                print("------------------")
                             
# extract the dose data out of the .3ddose file
                simDose = dose_utils.load_3ddose(simDoseFile)
                ''' there is a bug with cropping the dose grid. I should look into it later.
                perhaps, the "grid" is not the only attribute of the simDose that needs to be updated for the avg uncertainty calculations
                # simDose["grid"] = simDose["grid"][155:412, 155:412, 155:412]
                '''
# test in the dose loading was successful
                testSimLoadSuccess(simDose)

# scroll through the 3ddose files
                scroll_dose.plot_scrollable(simDose["grid"], "3ddose")
                
# cropping the dose grid of 3ddose file to match the dose of DICOM file
                a3ddose_shape = np.shape(simDose["grid"])

                # let's get the range of values from size of dicom dose
                crop_out = (a3ddose_shape[0] - dicom_shape[0])/2
                lower_bound = int(crop_out-1)
                upper_bound = int(a3ddose_shape[0]-crop_out-1)
                simDose["grid"] = simDose["grid"][lower_bound:upper_bound, lower_bound:upper_bound, lower_bound:upper_bound]
                print("------------------")
                print("here is the size of croped out 3ddose::::: \n", np.shape(simDose["grid"]))

# time to get % error
                ''' at the moment, %error does not work since the sizes of the arrays do not match
                mean_abs_percent_err = np.mean(np.abs((dicom_object.dose_grid - simDose["grid"])/dicom_object.dose_grid))*100
                print("The mean absolute percent error between the simulations and the ground truth is: \n")
                print(mean_abs_percent_err)
                '''

# let's do Gamma Variate analysis


