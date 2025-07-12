#!/bin/bash
# # for local installation
# dir_software=${HOME}/Software

# # For docker image
dir_software=/app/Software
python3.13 -m pip install ${dir_software}/brachyutils
python3.13 -m amplpy.modules install highs gurobi xpress cplex scip gcg