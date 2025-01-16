# syntax=docker.io/docker/dockerfile:1.7-labs
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 as builder

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Torch
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip uninstall -y triton && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Install TensorRT
RUN pip install --no-cache-dir tensorrt && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y triton && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Prepare python package layer (excluding separate layer packages)
ENV SEPARATE_PACKAGES="torch|nvidia"
RUN mkdir -p /opt/layers /opt/layers/python-packages && \
    find /usr/local/lib/python3.10/dist-packages -mindepth 1 -maxdepth 1 \
    -not -path "/usr/local/lib/python3.10/dist-packages/nvidia*" \
    -not -path "/usr/local/lib/python3.10/dist-packages/torch*" \
    -exec cp -r {} /opt/layers/python-packages/ \;

# Prepare Nvidia layers
ENV NVIDIA_PACKAGES="cuda cublas cudnn cufft curand cusolver cusparse nccl nvjitlink nvtx"
RUN for pkg in ${NVIDIA_PACKAGES}; do \
        mkdir -p /opt/layers/${pkg} &&  \
        mkdir -p /opt/layers/${pkg}/nvidia/${pkg} && \
        cp -r /usr/local/lib/python3.10/dist-packages/nvidia_${pkg}* /opt/layers/${pkg} && \
        cp -r /usr/local/lib/python3.10/dist-packages/nvidia/${pkg}* /opt/layers/${pkg}/nvidia/; \
    done

# Prepare Torch layer
RUN mkdir -p /opt/layers/torch && \
    cp -r /usr/local/lib/python3.10/dist-packages/torch* /opt/layers/torch/


FROM nvidia/cuda:12.4.1-base-ubuntu22.04 as runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy python packages layer
COPY --from=builder /opt/layers/python-packages /usr/local/lib/python3.10/dist-packages

# Copy torch layer
COPY --from=builder /opt/layers/torch /usr/local/lib/python3.10/dist-packages

# Copy nvidia layers
COPY --from=builder /opt/layers/cuda /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/cublas /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/cudnn /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/cufft /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/curand /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/cusolver /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/cusparse /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/nccl /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/nvjitlink /usr/local/lib/python3.10/dist-packages
COPY --from=builder /opt/layers/nvtx /usr/local/lib/python3.10/dist-packages

# Link CUDA
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# Install StreamDiffusion
ARG SD_BRANCH=main
WORKDIR /app
RUN git clone https://github.com/cumulo-autumn/StreamDiffusion.git
WORKDIR /app/StreamDiffusion
RUN pip install --no-cache-dir -e . && \
    pip uninstall -y triton && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*
