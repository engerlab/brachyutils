# needed libraries
import dose_utils
import os
import numpy as np
import dicompylercore

# set directory of the 3ddose files
simfileDir = "/home/hosseinj/data/Case1_elekta_simResults"
dicomfileDir = ""


# a for loop to iterate through dose files in the specified directory
for fileName in os.listdir(simfileDir):
# only pick .3ddose files
    if fileName.endswith(".3ddose"):
# get and print the full path of the file
        doseFile = os.path.join(simfileDir, fileName)
        print(doseFile)
# extract the dose data out of the .3ddose file
        dose_array = dose_utils.load_3ddose(doseFile)["grid"]
        print("Type of the dose_array is: ", type(dose_array))
        print("-----------------")
        print("dimensions of the dose_array is:", np.shape(dose_array))
        print("-----------------")

# load the dicomfiles 









