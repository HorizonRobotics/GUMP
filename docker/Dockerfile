# Copyright (c) 2023 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For lower nvidia driver <470
ARG CUDA_BASE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# For higher nvidia driver >=470
# ARG CUDA_BASE=nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

######################## Build Vulkan #################################
FROM ubuntu:20.04 as vulkan-khronos

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libegl1-mesa-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libxrandr-dev \
    python3 \
    python3-distutils \
    curl wget && \
    rm -rf /var/lib/apt/lists/*

ARG VULKAN_VERSION=sdk-1.1.121.0

# Download and compile vulkan components
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    git clone https://github.com/KhronosGroup/Vulkan-ValidationLayers.git /opt/vulkan && \
    cd /opt/vulkan && git checkout "${VULKAN_VERSION}" && \
    mkdir build && cd build && ../scripts/update_deps.py && \
    cmake -C helper.cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . && make install && ldconfig && \
    mkdir -p /usr/local/lib && cp -a Vulkan-Loader/build/install/lib/* /usr/local/lib && \
    mkdir -p /usr/local/include/vulkan && cp -r Vulkan-Headers/build/install/include/vulkan/* /usr/local/include/vulkan && \
    mkdir -p /usr/local/share/vulkan/registry && \
    cp -r Vulkan-Headers/build/install/share/vulkan/registry/* /usr/local/share/vulkan/registry && \
    rm -rf /opt/vulkan

FROM ${CUDA_BASE}

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN dpkg --add-architecture i386 && apt-get update && apt-get install -y --no-install-recommends \
    libxau6 libxau6:i386 \
    libxdmcp6 libxdmcp6:i386 \
    libxcb1 libxcb1:i386 \
    libxext6 libxext6:i386 \
    libx11-6 libx11-6:i386 \
    libglvnd0 libglvnd0:i386 \
    libgl1 libgl1:i386 \
    libglx0 libglx0:i386 \
    libegl1 libegl1:i386 \
    libgles2 libgles2:i386 \
    pkg-config \
    libglvnd-dev libglvnd-dev:i386 \
    libgl1-mesa-dev libgl1-mesa-dev:i386 \
    libegl1-mesa-dev libegl1-mesa-dev:i386 \
    libgles2-mesa-dev libgles2-mesa-dev:i386 \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libxrandr-dev \
    libegl1-mesa-dev

COPY --from=vulkan-khronos /usr/local/bin /usr/local/bin
COPY --from=vulkan-khronos /usr/local/lib /usr/local/lib
COPY --from=vulkan-khronos /usr/local/include/vulkan /usr/local/include/vulkan
COPY --from=vulkan-khronos /usr/local/share/vulkan /usr/local/share/vulkan
COPY ./nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY ./10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

######################## Hobot dependencies #####################
RUN apt update

RUN apt update && apt install -y --no-install-recommends \
    software-properties-common build-essential \
    vim tree curl wget unzip git cmake \
    libxml2 libxml2-dev libxslt1-dev libfreetype6-dev \
    dirmngr gnupg2 lsb-release \
    xvfb kmod swig patchelf ffmpeg rsync \
    libopenmpi-dev libcups2-dev libssl-dev \
    python3.9 python3-pip python3.9-dev python3-setuptools \
    libboost-all-dev

RUN python3.9 -m pip install --upgrade pip
RUN ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3

# for opencv
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libxext-dev

# If you need gcc-11/g++-11
# RUN apt-get install -y gcc-11 g++-11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# for pytorch
# RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118  && pip cache purge
#RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 && pip cache purge
# RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116 && pip cache purge
RUN pip3 install torch torchvision torchaudio pytorch-lightning==2.2.5 --extra-index-url https://download.pytorch.org/whl/cu118 && pip cache purge

# Install pacakges from requirements.txt
COPY ./requirements.txt  /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt  && pip cache purge

# Test cnest
RUN pip --version
RUN python -c "import cnest"

# If you plan to use original nuplan-devkit
# RUN pip install nuplan-devkit@git+https://github.com/motional/nuplan-devkit.git@nuplan-devkit-v1.1#egg=nuplan-devkit  && pip cache purge
# If you plan to use horizon-nuplan
# ARG GITHUB_TOKEN
# ENV GITHUB_TOKEN=${GITHUB_TOKEN}
# RUN pip install nuplan-devkit@git+https://github.com/HorizonRobotics/nuplan-devkit.git && pip cache purge

# Enable EGL
ENV PYOPENGL_PLATFORM=egl
RUN python -c "from OpenGL.raw.EGL._types import *"
# Enable determinism for pytorch
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN apt update && apt install  -y --no-install-recommends libgl1-mesa-dev libglfw3-dev libglew-dev

RUN pip install numba && pip cache purge

# Clean up to make the resulting image smaller
RUN  rm -rf /var/lib/apt/lists/*
