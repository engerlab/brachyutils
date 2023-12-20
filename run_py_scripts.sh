#!bin/bash
# This script contains demos on how to use dicom_utils.py, dose_utils.py and egsphant_utils.py 
# on the command line interface. You can also import these script 
source ENV/bin/activate

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
# python "src/egsphant_utils.py" \
#     "crop_by_body_contour_many_files" \
#     "INPUT_DIR" \
#     "PTH_JSON"

# # DOSE_UTILS.PY
# # to crop, padd and convert .3ddose files more compact formats such as nrrd or tar.zst. 
# # examples and details for each command is provided below:

# # # to convert all 3ddose files located at INPUT_DIR to nrrd in the same folder.  
# input_dir="/home/majd/data/patient_dose_simulations/prostate-glen"
# patient_folders=$input_dir'/*/'
# for folder in $patient_folders
# do 
#     echo $folder
#     brachyutils convert-many-dose-files $folder '.nrrd' '.minidos' --multi-proc
#     # rm $folder*.3ddose
# done

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

# # plan utils.py
# # to calculate uncertainity for all patients for a plan
dir_dicom_patients="/home/majd/data/patient_treatment_plans/dicom/prostate-glen-2023"
dir_dose_all_patients="/home/majd/data/patient_dose_simulations/prostate-glen-2023-1mm"
dir_plan_all_patients="/home/majd/data/patient_treatment_plans/tps_exported/prostate-glen-2023"
pth_uncertainty_all_patients=$dir_plan_all_patients
pth_dvh_metric_goals_json="/home/majd/Software/tg186-validation/data_test/dvh_metric_goals.json"

patient_folders=$dir_plan_all_patients'/*/'
for folder in $patient_folders
do
    # echo $folder 
    # Isolate the last subfolder from $folder
    last_subfolder=$(basename "$folder")
    echo "Last subfolder: $last_subfolder"
    dir_doseRate_map="$dir_dose_all_patients/$last_subfolder" 
    dir_plan="$dir_plan_all_patients/$last_subfolder" 
    dir_dicom="$dir_dicom_patients/$last_subfolder" 
    pth_uncertainty_json="$pth_uncertainty_all_patients/$last_subfolder/uncertainty.json" 
    # # Run the command

    echo "brachyutils get-uncertainty-one-patient \
        $dir_doseRate_map \
        $dir_plan \
        $dir_dicom \
        $pth_dvh_metric_goals_json \
        $pth_uncertainty_json \
        --multi-proc"

    echo "-----------------------------------"
    brachyutils get-uncertainty-one-patient\
    $dir_doseRate_map\
    $dir_plan $dir_dicom\
    $pth_dvh_metric_goals_json\
    $pth_uncertainty_json\
    --multi-proc

    # echo "$dir_doseRate_map"
    # echo "$dir_plan"
    # echo "$dir_dicom"
    # echo "$pth_dvh_metric_goals_json"
    # echo "$pth_uncertainty_json"        
    echo "-----------------------------------"
done