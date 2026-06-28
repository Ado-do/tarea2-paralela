FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    libx11-dev \
    libjpeg-dev \
    libpng-dev \
    git \
    cuda-nsight-systems-12-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD git clone https://github.com/Ado-do/tarea2-paralela.git . && tail -f /dev/null
