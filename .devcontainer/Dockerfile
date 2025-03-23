FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
# Install necessary build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m pip install --no-cache-dir Flask python-dotenv
RUN python3.12 -m pip cache purge
