# Base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---- system dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# ---- python deps ----
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---- copy entire repo ----
COPY . .

# ---- build C++ extension (pybind) ----
RUN python setup.py build_ext --inplace

# ---- ensure results directory exists ----
RUN mkdir -p /app/results

# ---- run full benchmark suite on container start ----
CMD ["bash", "run_all.sh"]
