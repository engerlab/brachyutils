# needed libraries
import dose_utils
import os
import numpy as np
import dicompylercore
import workplace

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
                return 1
        except:
                print("3ddose file did not load successfully")
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
                doseFile = mother_dir + "/" + case + "/simResults/combined.3ddose"
                print("looking at the file: \n")
                print(doseFile)
                print("------------------")
# extract the dose data out of the .3ddose file
                dose = dose_utils.load_3ddose(doseFile)
# test in the dose loading was successful
                testSimLoadSuccess(dose)
                





