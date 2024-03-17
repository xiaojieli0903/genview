#!/bin/bash

# Check if the genview environment exists
if conda info --envs | grep -q "genview"; then
    echo "The genview environment already exists, activating it."
    conda activate genview
else
    # Step 1: Create a Conda environment
    echo "Step 1: Creating a Conda environment"
    conda create --name genview python=3.8 -y
    conda activate genview

    # Step 2: Install PyTorch using Conda or Pip
    echo "Step 2: Installing PyTorch using Conda or Pip"
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
    # Or
    # conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
fi

# Step 3: Download the code and install
echo "Step 3: Downloading the code and installing"
cd mmpretrain_genview
pip install -U openmim && mim install -e .