#!/bin/bash

# Load the right version of python
module load python/3.11.7

# Install the python module virtualenv
pip3 install virtualenv

# Create the virtual environment
python -m virtualenv virtenv_cuda124

# Activate the environment
source virtenv_cuda124/bin/activate

# Install the right version of torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
#pip3 install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio===0.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
#pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install other packages 
pip3 install albumentations==1.3.1 opencv-python==4.8.0.76 pandas==1.5.3 pillow==9.4.0

# Install YOLO
pip3 install ultralytics==8.3.0 
# pip3 install ultralytics==8.0.196

# Install roboflow
pip3 install roboflow onnx openvino

# Install the ipykernel
pip3 install ipykernel
# Install the virtual environment for juypter
python -m ipykernel install --user --name=virtenv_cuda124

# Deactivate the virtual environment
deactivate
