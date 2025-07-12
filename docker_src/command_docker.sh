#!/bin/bash
xhost +local: # to allow the container to access the host display
# set the environment variables for the container
export UID=$(id -u)
export GID=$(id -g)
export DISPLAY=:0
# to build the image and run the container
# export DOCKER_BUILDKIT=1
docker compose up --build -d BrachyUtils
# to run the container without building the image
# docker compose up --no-build -d BrachyUtils
# docker compose up --no-build -d DoseCalcMC
# docker compose up --no-build -d DoseCalcTG43
# docker compose up --no-build -d Pyplastimatch
# to enter the container
docker exec -it BrachyUtils bash
