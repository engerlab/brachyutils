#!bin/bash
# This script contains demos on how to use dicom_utils.py, dose_utils.py and egsphant_utils.py 
# on the command line interface. You can also import these script 
source ../ENV/bin/activate

# # DICOM_UTILS.PY 
# # to extract body index limits from dicom folders. INPUT_DIR holds the dicom folders \
# # of many patients. Example:
# # INPUT_DIR/p1/
# # INPUT_DIR/p2/ ...
# # PTH_JSON is the json file path where patient folder name, body index range and the size
# # of the body mask is written. Example:
# [{
# # "patient_number": "p1",
# # "body_index_range": [
# #     [x_min, x_max],
# #     [y_min, y_max],
# #     [z_min,z_max]
# # ],
# # "body_mask_shape": [512, 512, 42]
# # }]
# python "src/dicom_utils.py" \
#     "get-body-contour-range-from-many-patients-dicom" \
#     "INPUT_DIR" \
#     "PTH_JSON"


# # EGSPHANT_UTILS.py
# # to crop the egsphant file inside the patient folders in the INPUT_DIR. 
# # In addition to the INPUT_DIR, it needs the PTH_JSON file path where the 
# # patient folder name and their respective body index range and body mask size
# # is storred. 
python "src/egsphant_utils.py" \
    "crop_by_body_contour_many_files" \
    "INPUT_DIR" \
    "PTH_JSON"

# # DOSE_UTILS.PY
# # to crop, padd and convert .3ddose files more compact formats such as nrrd or tar.zst. 
# # examples and details for each command is provided below:

# # to convert all 3ddose files located at INPUT_DIR to nrrd in the same folder.  
# python "src/dose_utils.py" \
#     convert-many-files \
#     'INPUT_DIR' \
#     '.3ddose' \
#     '.nrrd'

# # to crop all 3ddose files in INPUT_DIR folder by a certain fraction (0.6) and convert to nrrd. 
# # cropping 0.6 keeps the centeral 0.6 voxels on x and y axis. 
# python "src/dose_utils.py" \
#     crop-by-ratio-and-convert-many-files \
#     'INPUT_DIR' \
#     '0.6' \
#     '.3ddose' \
#     '.nrrd'

# # to padd all dose files in INPUT_DIR folder and save them as nrrd files. 
# python "src/dose_utils.py" \
#     padd_many_files \
#     'INPUT_DIR' \
#     '0.6' \
#     '.3ddose' \
#     '.nrrd'
