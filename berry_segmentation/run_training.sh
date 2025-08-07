#!/bin/bash -l

# SLURM parameters
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

# Below are example SLURM parameters. You may need to change them for your purpose:

#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=normal
#SBATCH --job-name=berrybox_segmentation_training
#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1


# Set error handling options
set -e
set -u
set -o pipefail

# Change the working directory
cd /path/to/berryboxai_training_public/berry_segmentation/

# Activate the environment
source /path/to/virtenv_cuda124/bin/activate


# Check nvidia
nvidia-smi

# Check number of GPUs
echo $CUDA_VISIBLE_DEVICES

# Run the python script
python train_yolov8_segmentation.py
