# BerryBoxAI Training

This repository contains code for training the deep learning models used in the [berryboxai](https://github.com/NeyhartLab/berryboxai) python package.

## Tasks

There are two deep learning inference tasks that are covered by the python package. Each of these tasks relies on a separate model trained with different datasets.

+ [Berry Segmentation](https://github.com/NeyhartLab/berryboxai_training_public/tree/main/berry_segmentation): separation of berry pixels from background and measurement of berry attributes.
+ [Fruit Rot Detection](https://github.com/NeyhartLab/berryboxai_training_public/tree/main/fruit_rot_detection): counting of sound and rotten fruit in an image.

## Data

Images and annotation data for each task is available from the USDA National Agriculture Library Ag Data Commons from the following links:
+ Berry Segmentation: (LINK HERE)
+ Fruit rot detection: (LINK HERE)

## Code

### Set up virtual environment

The code requires a conda environment to run. The code for setting up that conda environment is provided in the `setup_virtenv_cuda124_atlas.sh` script.

### Run code

Model training and evaluation for each task uses code that is designed to work with high-performance computing resources running the SLURM job scheduler. Example code is provided, but **users must modify this code for their systems**. 

The model training and evaluation code is formatted similarly for each task and includes two scripts:
+ `train_yolov8_[task].py` - the main training script, where "[task]" is either "segmentation" or "detection". 
+ `run_training.sh` - this is a shell script for batch submission of jobs to the SLURM scheduler.



