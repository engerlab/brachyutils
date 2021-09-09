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
                simDoseFile = glob.glob(mother_dir + "/" + case + "/simResults/*.3ddose")[0]
                print("looking at the file: \n")
                print(simDoseFile)
                print("------------------")
                             
# extract the dose data out of the .3ddose file
                simDose = dose_utils.load_3ddose(simDoseFile)

# test in the dose loading was successful
                testSimLoadSuccess(simDose)

# scroll through the 3ddose files
                scroll_dose.plot_scrollable(simDose["grid"], "3ddose")
                
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

# time to get % error
                ''' at the moment, %error does not work since the sizes of the arrays do not match
                mean_abs_percent_err = np.mean(np.abs((dicom_object.dose_grid - simDose["grid"])/dicom_object.dose_grid))*100
                print("The mean absolute percent error between the simulations and the ground truth is: \n")
                print(mean_abs_percent_err)
                '''

# let's do Gamma Variate analysis




