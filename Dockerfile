# ---------- Builder stage: create the conda env with micromamba ----------
FROM mambaorg/micromamba:1.4.0 AS builder

# Copy the environment YAML into the builder
COPY environment.yml /tmp/env.yml

# Create environment under /opt/conda/envs/py310
RUN micromamba create -y -f /tmp/env.yml -p /opt/conda/envs/py310 \
    && micromamba clean --all --yes \
    && rm -f /tmp/env.yml || true

# Remove package caches to minimize size
RUN rm -rf /opt/conda/pkgs /root/.cache/pip || true


# ---------- Runtime stage: CUDA runtime with only the environment copied in ----------
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System-level dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    bzip2 \
    libglib2.0-0 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libbz2-1.0 \
    git \
    build-essential \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    vim \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the prepared conda environment from builder
COPY --from=builder /opt/conda/envs/py310 /opt/conda/envs/py310

# Clean up potential cache directories
RUN rm -rf /opt/conda/envs/py310/var /opt/conda/envs/py310/pkgs || true

# Add conda environment to PATH
ENV PATH=/opt/conda/envs/py310/bin:$PATH
ENV CONDA_DEFAULT_ENV=py310
ENV CONDA_PREFIX=/opt/conda/envs/py310

# Locale setup
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set workdir to /app and ensure non-root user owns it
WORKDIR /app
COPY . /app

# Default entrypoint
ENTRYPOINT [ "bash", "-lc" ]
CMD [ "python --version" ]

