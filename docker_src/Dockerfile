FROM ubuntu:latest
# FROM brachyutils

# ARG HOST_HOME
USER root

ARG dir_software="/app/Software"
ARG clean_source="false"

WORKDIR ${dir_software}

# # install requirements needed for python 3.12
RUN apt update && apt install -y zip unzip wget curl net-tools\
    nano git openssh-client &&\
    mkdir -p /root/.ssh &&\
    ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN apt install -y build-essential zlib1g zlib1g-dev libncurses5-dev \
    libgdbm-dev pkg-config libnss3-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev wget nano liblzma-dev libbz2-dev \
    libxrender1 libgl1 libglib2.0-0 tk-dev xz-utils llvm libgfortran5

COPY docker_src/repositories/install_python.sh ${dir_software}/install_python.sh
RUN --mount=type=ssh bash ${dir_software}/install_python.sh

# # install opentps-brachyutils
COPY docker_src/repositories/install_opentps.sh ${dir_software}/install_opentps.sh
RUN --mount=type=ssh bash ${dir_software}/install_opentps.sh

# # install ai-assisted-brachytherapy
# COPY docker_src/repositories/install_ai_bt.sh ${dir_software}/install_ai_bt.sh
# RUN --mount=type=ssh bash ${dir_software}/install_ai_bt.sh

# # install brachyutils
COPY install_brachyutils.sh ${dir_software}/install_brachyutils.sh
RUN --mount=type=ssh bash ${dir_software}/install_brachyutils.sh
