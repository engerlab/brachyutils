from brachyutils import BrachyPhantom, BrachyEgsphant
import os
from opentps.core.data.images import ROIMask, Image3D
import numpy as np
from matplotlib import pyplot as plt
import copy

os.chdir(os.path.dirname(__file__))

#load the egsphant and phantom
cropped_egsphant = BrachyEgsphant("./RapidBrachy/breast_phantom_cropped.seq.nrrd")
#should be the same as the mask
uncropped_egsphant = BrachyEgsphant("./RapidBrachy/breast_phantom_unsampled.seq.nrrd")
#uncropped_egsphant.unit_length = "cm"

STRUCTURE_FILE = "RS.1.2.246.352.71.4.810100034225.661.20150513153609.dcm"
DICOM_DIR = f"./DICOM_INPUT/"
phantom = BrachyPhantom(DICOM_DIR, None, DICOM_DIR + STRUCTURE_FILE, None)

body_mask = phantom.get_structure_mask(["BODY"], ROIMask)["BODY"]
body_mask_image = body_mask
#body_mask_image = copy.deepcopy(uncropped_egsphant.material_image)
#body_mask_image.imageArray = body_mask.imageArray #np.swapaxes(body_mask.imageArray, 0, 2)
#body_mask_image.origin = body_mask_image.origin[::-1]
#body_mask_image.spacing = body_mask_image.spacing[::-1]
assert(np.all(body_mask_image.gridSize == body_mask_image.imageArray.shape))
#body_mask_image.gridSize = body_mask_image.gridSize[::-1]
#body_mask_image.update()

plt.imshow(body_mask_image.imageArray[:, :, body_mask_image.imageArray.shape[2]//2])
plt.figure()
plt.imshow(cropped_egsphant.material_image.imageArray[:, :, cropped_egsphant.material_image.imageArray.shape[2]//2])
plt.show()

from opentps.core.processing.imageProcessing.resampler3D import resampleImage3DOnImage3D

body_mask_resampled = resampleImage3DOnImage3D(body_mask_image, cropped_egsphant.material_image)
body_mask_resampled.imageArray = np.logical_not(body_mask_resampled.imageArray)
cropped_egsphant.material_image.imageArray = np.where(body_mask_resampled.imageArray, 0, cropped_egsphant.material_image.imageArray)
cropped_egsphant.density_image.imageArray = np.where(body_mask_resampled.imageArray, 0.001225, cropped_egsphant.density_image.imageArray)

dummy_phantom_1 = BrachyPhantom()
dummy_phantom_1.image_obj = body_mask_image
dummy_phantom_1.image_obj.imageArray = dummy_phantom_1.image_obj.imageArray.astype(np.uint8)
dummy_phantom_1.write_image_to_nrrd("./RapidBrachy/body_mask_uncropped.nrrd")

dummy_phantom_2 = BrachyPhantom()
dummy_phantom_2.image_obj = body_mask_resampled
dummy_phantom_2.image_obj.imageArray = dummy_phantom_2.image_obj.imageArray.astype(np.uint8)
dummy_phantom_2.write_image_to_nrrd("./RapidBrachy/body_mask_cropped.nrrd")

plt.figure()
plt.imshow(body_mask_resampled.imageArray[:,:,body_mask_resampled.imageArray.shape[2]//2])
plt.show()

cropped_egsphant.write_to_nrrd("./RapidBrachy/breast_phantom_cropped_cleaned.seq.nrrd")
cropped_egsphant.write_to_ctegsphant("./RapidBrachy/breast_phantom_cropped_cleaned.egsphant")

