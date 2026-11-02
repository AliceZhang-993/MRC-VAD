#!/bin/bash
# Build the three custom CUDA extensions required by FlowNet2.
# Needs a CUDA toolkit (nvcc) matching the installed PyTorch build.
set -e
cd "$(dirname "$0")/flownet_networks"

for pkg in correlation_package resample2d_package channelnorm_package; do
    cd "$pkg"
    rm -rf *_cuda.egg-info build dist __pycache__
    pip install -v -e .
    cd ..
done
