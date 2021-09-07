# needed libraries
import dose_utils
import os
import numpy as np
import dicompylercore
import workplace

# location = input("Where are you, home or jgh?")
location = workplace.askForLocation()

# set directory of the 3ddose files
if workplace._workplace(location):
        simfileDir = "/home/majd/data/TG186 Vallidation/Elekta/Case1-simResults"
        dicomfileDir = ""
else:
        simfileDir = "/home/hosseinj/data/Case1_elekta_simResults"
'''
# a for loop to iterate through dose files in the specified directory
for fileName in os.listdir(simfileDir):
# only pick .3ddose files
    if fileName.endswith(".3ddose"):
# get and print the full path of the file
        doseFile = os.path.join(simfileDir, fileName)
        print("loading file at:", doseFile)
# extract the dose data out of the .3ddose file
        dose = dose_utils.load_3ddose(doseFile)
        print("Type of the dose is: ", type(dose["grid"]))
        print("-----------------")
        print("dimensions of the dose is:", np.shape(dose["grid"]))
        print("-----------------")
        print("the average uncertainty of the dose is:", dose_utils.get_average_uncert(dose))

# load the dicomfiles 
'''








