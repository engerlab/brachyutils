FROM ubuntu:latest

ARG HOST_HOME

ARG dir_software="/root/Software"
ARG clean_source="false"

WORKDIR ${dir_software}

# # install requirements needed for python 3.12
RUN apt update && apt install -y zip unzip wget curl net-tools nano

COPY install_python.sh ${dir_software}/install_python.sh
RUN bash install_python.sh

# # copy opentps-brachyutils and install it
COPY docker_src/repositories/OpenTPS-brachyutils-master.zip ${dir_software}
RUN unzip OpenTPS-brachyutils-master.zip\
    && rm OpenTPS-brachyutils-master.zip\
    && mv OpenTPS-brachyutils-master OpenTPS-brachyutils

# # TODO: Delete this line after pushing to github
COPY docker_src/repositories/install_opentps.sh ${dir_software}/OpenTPS-brachyutils/install_opentps.sh
# keep this line
RUN bash ${dir_software}/OpenTPS-brachyutils/install_opentps.sh

# # copy ai_assisted_brachy source code and install it
COPY docker_src/repositories/AI_Assisted_Brachytherapy-main.zip ${dir_software}
RUN unzip AI_Assisted_Brachytherapy-main.zip\
    && rm AI_Assisted_Brachytherapy-main.zip\
    && mv AI_Assisted_Brachytherapy-packaging AI_Assisted_Brachytherapy

COPY docker_src/repositories/install_ai_bt.sh ${dir_software}/AI_Assisted_Brachytherapy/install_ai_bt.sh
RUN bash ${dir_software}/AI_Assisted_Brachytherapy/install_ai_bt.sh

# copy brachyutils source code and install it
# COPY ${HOST_HOME}/Software/brachyutils ${dir_software}/brachyutils
# RUN bash ${dir_software}/brachyutils/installation_script.sh