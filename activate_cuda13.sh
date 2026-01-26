#!/bin/bash
# Activate CUDA 13 libraries for PyTorch 2.9+ with RTX 50xx support
export LD_LIBRARY_PATH=/home/pbustos/software/pip_venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
echo "CUDA 13 libraries activated for RTX 5070 Ti"

