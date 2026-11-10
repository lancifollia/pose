FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y vim git gcc zip
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libsm6 libxext6 ffmpeg libgeos-dev
RUN apt-get install -y nvtop htop
RUN apt-get install -y wget build-essential libssl-dev openssl libtool

RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir matplotlib
RUN pip3 install --no-cache-dir jupyterlab

# gpu driver
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /home/dev

CMD ["/bin/bash"]
