FROM nvidia/cuda:11.4.0-devel-ubuntu18.04

RUN apt-get update --fix-missing; exit 0
RUN apt-get install -y \
    vim \
    wget
RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade pip && pip install numpy
WORKDIR /root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1-linux-x86_64.tar.gz && tar -zxvf cmake-3.23.1-linux-x86_64.tar.gz && ln -sf /root/cmake-3.23.1-linux-x86_64/bin/* /usr/local/bin/


