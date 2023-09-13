#!bin/bash

source /home/majd/Software/tg186-validation/ENV/bin/activate

PYTHONPATH=./src

start_run="$(date +%s)"

python "./dose_utils.py"

duration=$[ $(date +%s) - ${start_run} ]

echo ${duration}