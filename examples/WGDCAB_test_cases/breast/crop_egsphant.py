from brachyutils import BrachyEgsphant
import os
import numpy as np

# Change directory to where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Define the path to the egsphant file
uncropped_egsphant_file = "./RapidBrachyMCTPS_Export/ct.egsphant"
#uncropped_egsphant_file = "./RapidBrachy/breast_phantom_sampled.seq.nrrd"
uncropped_egsphant = BrachyEgsphant(uncropped_egsphant_file)

print(uncropped_egsphant.material_image.origin)
print(uncropped_egsphant.material_image.spacing)
print(uncropped_egsphant.material_image.imageArray.shape)

#write_rewrite_file = "./RapidBrachy/breast_phantom_uncropped_rewrite.seq.nrrd"
#uncropped_egsphant.write_to_nrrd(write_rewrite_file)

# Crop the egsphant file
cropped_egsphant_file = "./RapidBrachy/breast_phantom_cropped"
cropped_egsphant = uncropped_egsphant.crop_by_index(np.array([[171, 330], [67, 194], [5, 116]]), inplace=False)
cropped_egsphant.write_to_ctegsphant(cropped_egsphant_file + ".egsphant")
cropped_egsphant.write_to_nrrd(cropped_egsphant_file + ".seq.nrrd")