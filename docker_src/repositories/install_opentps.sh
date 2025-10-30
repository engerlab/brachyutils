#!/bin/bash
# # for local installation
# dir_software=${HOME}/Software

# # For docker image
dir_software=/app/Software
rm -rf ${dir_software}/OpenTPS-brachyutils
git clone https://github.com/engerlab/OpenTPS-brachyutils ${dir_software}/OpenTPS-brachyutils
python3.13 -m pip install ${dir_software}/OpenTPS-brachyutils