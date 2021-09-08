# needed libraries
import dose_utils
import os
import numpy as np
from dicompylercore import dose
import workplace
import scroll_dose
import glob


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
                print("3ddose file did not load successfully")
                return 0

def testDicomLoadSuccess(dicom_dose)
        try:
                print("dasf")
        except:
                print("asdfa")
     
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
                simDoseFile = glob.glob(mother_dir + "/" + case + "/simResults/*.3ddose")
                print("looking at the file: \n")
                print(simDoseFile)
                print("------------------")
                '''
# extract the dose data out of the .3ddose file
                simDose = dose_utils.load_3ddose(simDoseFile)

# test in the dose loading was successful
                testSimLoadSuccess(simDose)

# scroll through the 3ddose files
                scroll_3ddose.plot3ddose(simDose["grid"])
'''
# let's load the ground truth from DICOM files
                dicomDoseFile = glob.glob(mother_dir + "/" + case + "/dicom/RD*")
                print("looking at the file: \n")
                print(dicomDoseFile)
                print("------------------")
                dicom_object = dose.DoseGrid(dicomDoseFile)





# time to get % error!





