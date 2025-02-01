#!/bin/bash
# to build the image and run the container
docker compose up --build BrachyUtils
# to run the container without building the image
# docker compose up --no-build -d
# to enter the container
docker exec -it BrachyUtils bash
