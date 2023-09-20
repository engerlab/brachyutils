#!bin/bash

source ../ENV/bin/activate

PYTHONPATH=./src

start_run="$(date +%s)"

python "./dose_utils.py" \
    convert-many-files \
    '../data_test/many_files' \
    '.3ddose' \
    '.nrrd'


duration=$[ $(date +%s) - ${start_run}  ]

echo ${duration} seconds