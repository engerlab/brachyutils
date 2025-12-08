#!/bin/bash
# # for local installation
# dir_software=${HOME}/Software

# # For docker image
dir_software=/app/Software

apt install -y build-essential zlib1g zlib1g-dev libncurses5-dev \
    libgdbm-dev pkg-config libnss3-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev wget nano liblzma-dev libbz2-dev \
    libxrender1 libgl1 libglib2.0-0 tk-dev xz-utils llvm libgfortran5

# rm -rf ${dir_software}/brachyutils
# git clone git@github.com:engerlab/brachyutils.git ${dir_software}/brachyutils
# python3.13 -m pip install -e ${dir_software}/brachyutils
python3.13 -m amplpy.modules install highs gurobi xpress \
    cplex scip gcg coin