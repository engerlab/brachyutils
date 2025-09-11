FROM ubuntu:latest

ARG HOST_HOME

ARG dir_software="/app/Software"
ARG clean_source="false"

WORKDIR ${dir_software}

# # install requirements needed for python 3.12
RUN apt update && apt install -y zip unzip wget curl net-tools\
    nano git openssh-client &&\
    mkdir -p /root/.ssh &&\
    ssh-keyscan github.com >> /root/.ssh/known_hosts
# 
COPY docker_src/repositories/install_python.sh ${dir_software}/install_python.sh
RUN bash install_python.sh

# # install opentps-brachyutils
COPY docker_src/repositories/install_opentps.sh ${dir_software}/install_opentps.sh
RUN --mount=type=ssh bash ${dir_software}/install_opentps.sh

# # install ai-assisted-brachytherapy
COPY docker_src/repositories/install_ai_bt.sh ${dir_software}/install_ai_bt.sh
RUN --mount=type=ssh bash ${dir_software}/install_ai_bt.sh

# # install brachyutils
COPY install_brachyutils.sh ${dir_software}/install_brachyutils.sh
RUN --mount=type=ssh bash ${dir_software}/install_brachyutils.sh
