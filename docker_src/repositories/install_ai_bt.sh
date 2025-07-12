#!/bin/bash
# # for local installation
# dir_software=${HOME}/Software
# # For docker image
dir_software=/app/Software
git clone -b packaging git@github.com:engerlab/AI_Assisted_Brachytherapy.git ${dir_software}/AI_Assisted_Brachytherapy
# python3.13 -m pip install ${dir_software}/AI_Assisted_Brachytherapy