FROM ubuntu:latest

ARG dir_software="/root/Software"
ARG clean_source="false"
ARG num_threads=12

RUN mkdir -p $dir_software/brachyutils
COPY installation_script.sh $dir_software/brachyutils

RUN apt update && apt install -y git

RUN bash ${dir_software}/brachyutils/installation_script.sh\
    --dir-software=$dir_software\
    --dependencies="python3.13"\
    --num-threads=${num_threads}

COPY brachyutils $dir_software/brachyutils/brachyutils
COPY setup.py $dir_software/brachyutils/setup.py

RUN bash ${dir_software}/brachyutils/installation_script.sh\
    --dir-software=$dir_software\
    --dependencies="brachyutils"\
    --num-threads=${num_threads}

CMD ["/bin/bash"]
