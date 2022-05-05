FROM nvidia/cuda:11.4.0-devel-ubuntu18.04

RUN apt-get update --fix-missing; exit 0
RUN apt-get install -y \
        vim \
        software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && y ; exit 0
RUN apt -y install python3.7
RUN apt -y install python3-pip
RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade pip && pip install numpy
