FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv 
RUN pip install --no-cache-dir uv

WORKDIR /workspaces/${localWorkspaceFolderBasename}
