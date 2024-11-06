FROM ubuntu:22.04

# Install dependencies
RUN apt update && apt install -y wget nano tar ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    libxrender1 libmkl-rt \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && mkdir /root/.conda \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm /tmp/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

# Initialize conda and create environment
RUN /root/miniconda3/bin/conda init bash

RUN conda create -n env_brachyutils python=3.11 -y
RUN echo "conda activate env_brachyutils" >> ~/.bashrc

# # Install brachyutils
COPY ./brachyutils /root/brachyutils/brachyutils/
COPY ./setup.py /root/brachyutils/
SHELL ["conda", "run", "-n", "env_brachyutils", "/bin/bash", "-c"]
RUN python -m pip install -e /root/brachyutils/.