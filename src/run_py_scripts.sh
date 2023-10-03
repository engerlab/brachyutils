#!bin/bash

source ../ENV/bin/activate

PYTHONPATH=./src

start_run="$(date +%s)"

# # to conver from 3ddose to nrrd
# python "./dose_utils.py" \
#     convert-many-files \
#     '../data_test/many_files' \
#     '.3ddose' \
#     '.nrrd'

# # to crop by a 1/3 and convert to nrrd
python "./dose_utils.py" \
    crop-by-ratio-and-convert-many-files \
    '../data_test/' \
    '0.6' \
    '.3ddose' \
    '.nrrd'

duration=$[ $(date +%s) - ${start_run}  ]

echo ${duration} seconds