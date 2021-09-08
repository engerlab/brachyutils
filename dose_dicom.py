# import necessary libraries 
from dicompylercore import dose
import os
import numpy as np
import workplace
'''
# set up the location of the compute u run at
dicom_dir = workplace._workplace(workplace.askForLocation())
print("looking at the directory: \n")
print(simfileDir)
print("------------------")

location = workplace.askForLocation()
if workplace._workplace:
    dicom_dir = "/home/majd/data/TG186 Vallidation/Elekta/Case1-OCB-MCNP6"
else:
    dicom_dir = ""
print("I am looking at this dicom dir: \n", dicom_dir)
print("-----------------")
fileName = "RD_Case-1_MCNP6.dcm"
fullFile = os.path.join(dicom_dir, fileName)
print("I am looking at this file: \n", fullFile)
print("-----------------")
'''

dicom_object = dose.DoseGrid(fullFile)

print("loading dicom dose file has been successful, here is the shape of the dose \n")
print(dicom_object.shape)
print("-----------------")
print("here is the first few dose values \n")
print(dicom_object.dose_grid[0, 0, 100])
'''
print("this is the type of the dicom_object:\n")
print(type(dicom_object))
print("-----------------")

# structures = dicom_object.GetStructures()
print("this is the type of the structure set: \n")
print(type(structures))
print("-----------------")

print("this is an example of a structure: \n")
print(structures)
print("-----------------")

# dose_aStructure = dvh.DVH.from_dicom_dvh(fullFile, 0)
# dose_aStructure.describe()

'''