#!bin/bash

source ../ENV/bin/activate

PYTHONPATH=./src

start_run="$(date +%s)"

python "./dose_utils.py"

duration=$[ $(date +%s) - ${start_run}  ]

echo ${duration}