
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN apt-get update && apt-get install -y apt-file && apt upgrade -y
RUN apt install -y build-essential curl git cmake pkg-config checkinstall
RUN apt-get install -y ca-certificates \
    git \
    bzip2 \
    libx11-6 \
    nano \
    libxext6 libxrender-dev \
    libgl1-mesa-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-py39_24.4.0-0-Linux-x86_64.sh
RUN bash Miniconda3-py39_24.4.0-0-Linux-x86_64.sh -b -p /opt/miniconda3
RUN conda update -y -n base -c defaults conda
RUN conda install -y pip

# Install dependencies from the env.yml file
ADD env.yml /tmp/env.yml
RUN conda env update -n base -f /tmp/env.yml --prune \
&& conda clean -ya


