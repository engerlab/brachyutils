#!/bin/bash
# to build the image and run the container
# docker compose up --build -d BrachyUtils
# to run the container without building the image
docker compose up --no-build -d BrachyUtils 
# docker compose up --no-build -d DoseCalcMC
docker compose up --no-build -d DoseCalcTG43
# docker compose up --no-build -d Pyplastimatch
# to enter the container
# docker exec -it BrachyUtils bash
