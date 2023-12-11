#!bin/bash
source /home/majd/Software/tg186-validation/ENV/bin/activate

input_dir="/home/majd/data/patient_dose_simulations/prostate-glen-1mm"
patient_folders=$input_dir'/*/'
for folder in $patient_folders
do 
    echo $folder
    folder_name=`basename $folder`
    echo $folder_name
    cd $folder
    # check if nrrd files exist in this folder
    count=`ls -l *nrrd 2>/dev/null | wc -l`
    echo $count
    if [ $count != 0 ]; then
        echo "nrrd files exist in this folder"
        continue
    fi
    tar --use-compress-program=zstd -xvf *.tar.zst
    # mv /home/hosseinj/scratch/brachySims/prostate-glen-1mm/$folder_name/*.nrrd $folder
    # rename 's/cropped_//' *.nrrd 

    # brachyutils convert-many-dose-files $folder '.nrrd' '.minidos' --multi-proc

done

# 