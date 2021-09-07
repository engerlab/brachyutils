# needed libraries
import dose_utils
import os

# set directory of the 3ddose files
fileDir = "/home/hosseinj/data/Case1_elekta_simResults"

# a for loop to iterate through dose files in the specified directory
for fileName in os.listdir(fileDir):
# only pick .3ddose files
    if fileName.endswith(".3ddose"):
# get and print the full path of the file
        doseFile = os.path.join(fileDir, fileName)
        print(doseFile)
# extract the dose data out of the .3ddose file
        dose_array = dose_utils.load_3ddose(doseFile)["grid"]
        print("Type of the dose_array is:")
        print(type(dose_array))
        print("\n")




