FROM ubuntu:latest

ARG HOST_HOME

ARG dir_software="/root/Software"
ARG clean_source="false"

WORKDIR ${dir_software}

# # install python 3.13
RUN apt install -y build-essential zlib1g zlib1g-dev libncurses5-dev \
    libgdbm-dev pkg-config libnss3-dev libssl-dev libreadline-dev \
    libffi-dev libsqlite3-dev wget nano liblzma-dev

RUN wget -nc https://www.python.org/ftp/python/3.13.1/Python-3.13.1.tgz \
    && tar -xf Python-3.13.1.tgz \
    


# # copy opentps-brachyutils and install it

# # copy ai_assisted_brachy source code and install it

# # copy brachyutils source code and install it
COPY ${HOST_HOME}/Software/brachyutils ${dir_software}/brachyutils
RUN bash ${dir_software}/brachyutils/installation_script.sh