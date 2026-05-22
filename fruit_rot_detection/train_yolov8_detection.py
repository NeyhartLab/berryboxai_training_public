# BerryBox Fruit Rot Object Detection

## Train YOLO V8 Object Detection Model

## Step 0. Setup

### User settings

#### Data settings
data_path = "" # Path to the dataset. MAKE SURE THIS DOES NOT END WITH /

# Proportion of images to use for train/test/validation
p_val = 0.10
p_test = 0.10
p_train = 1 - (p_val + p_test)

shuffle_data = True # Should data be shuffled into train, val, and test?
resume_training = False # should training be resumed?

#### Model parameters
##### specify training parameters
epochs = 600 # number of epochs to train
patience = 100 # early stopping patience
batch_size = 2 # -1 for auto else specify 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 etc
img_size = 2400
cache = True # cache images for faster training
workers = 1
model_size = "s" # ["n", "s", "m", "l"]
project = "berrybox_fruit_rot_detection" # project name
exist_ok = True # existing project name, new name, or 'resume'
optimizer = "auto" # optimizer to use for training, options are "SGD", "Adam", "AdamW", "Adamax", "RMSprop", "RAdam", "auto"
verbose = True # print verbose output

##### Augmentation hyperparameters
hsv_h = 0.2
hsv_s = 0.2
hsv_v = 0.2
degrees = 15
translate = 0.1
scale = 0.5
shear = 0
perspective = 0
flipud = 0.3
fliplr = 0.3
mosaic = 0
mixup = 0
copy_paste = 0
crop_fraction = 0



### Load libraries 

# Load YOLO
import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Load other libraries
import os
import torch
import time
import subprocess
import gc
import random
import shutil
import pandas as pd

proj_dir = os.getcwd()

# clear gpu cache if gpu is available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = "0" # '0' or "0,1,2,3" depending on the number of GPUs available
    device = ",".join([str(x) for x in range(torch.cuda.device_count())])
    print("\nRunning on device: " + str(device))

# clear garbage
gc.collect()


"""3. Shuffle the dataset"""

# Shuffle the dataset using homemade code
data_path = dataset_dir
data_path_shuf = data_path + "_shuffled"

if shuffle_data:
    # Delete the older shuffled dataset if present
    if os.path.exists(data_path_shuf):
        shutil.rmtree(data_path_shuf)

    # Copy the entire folder
    # !cp -r {data_path} {data_path_shuf}
    subprocess.run(["cp", "-r", data_path, data_path_shuf], check=True)

    # Edit the yaml file
    with open(os.path.join(data_path_shuf, "data.yaml"), "r") as f:
        lines = f.readlines()
    newlines = []
    line0 = lines[0]
    lines[0] = "path: " + data_path_shuf + "\n" + line0
    lines[1:4] = [line.replace("../", "") for line in lines[1:4]]
    # Edit the path to train, valid, and test
    for line in lines:
        if line.startswith("test:") or line.startswith("train:") or line.startswith("val:"):
            continue
        else:
            newlines.append(line)
    test_train_val = ['test: ../test/images\n', 'train: train/images\n', 'val: valid/images\n']
    newlines.extend(test_train_val)

    with open(os.path.join(data_path_shuf, "data.yaml"), "w") as f:
        f.writelines(newlines)

    # Rename the "train" folder "all"
    # !mv {data_path_shuf}/train {data_path_shuf}/all
    subprocess.run(["mv", data_path_shuf + "/train", data_path_shuf + "/all"], check=True)

    # List the basename of the images in the "all" folder
    all_image_path = data_path_shuf + "/all/images/"
    all_labels_path = data_path_shuf + "/all/labels/"
    # List basenames (i.e. sans extension)
    all_basenames = [".".join(x.split(".")[:-1]) for x in os.listdir(all_image_path)]

    # Determine the number of train/test/val images
    n_images = len(all_basenames)
    n_val = int(p_val * n_images)
    n_test = int(p_test * n_images)
    n_train = n_images - (n_val + n_test)

    # Randomly sample basename for training/test/val
    # 1. Make a copy to avoid altering the original list, then shuffle it
    shuffled_basenames = list(all_basenames)
    random.shuffle(shuffled_basenames)

    # 2. Slice the list using the counts
    train_basenames = shuffled_basenames[:n_train]
    val_basenames = shuffled_basenames[n_train : n_train + n_val]
    test_basenames = shuffled_basenames[n_train + n_val:]

    # Print the results
    print("Number of training images: " + str(n_train))
    print("Number of validation images: " + str(n_val))
    print("Number of testing images: " + str(n_test))

    # Create train/val/test folders (and delete them if they already exist)
    split_dirnames = ["train", "valid", "test"]
    # Iterate
    for dirname in split_dirnames:
        # splitdir = dataset.location + "/" + dirname
        splitdir = data_path_shuf + "/" + dirname
        if os.path.exists(splitdir):
            shutil.rmtree(splitdir)

        if dirname == "train":
            images = [x for x in os.listdir(all_image_path) if any([y in x for y in train_basenames])]
            labels = [x for x in os.listdir(all_labels_path) if any([y in x for y in train_basenames])]
        elif dirname == "valid":
            images = [x for x in os.listdir(all_image_path) if any([y in x for y in val_basenames])]
            labels = [x for x in os.listdir(all_labels_path) if any([y in x for y in val_basenames])]
        else:
            images = [x for x in os.listdir(all_image_path) if any([y in x for y in test_basenames])]
            labels = [x for x in os.listdir(all_labels_path) if any([y in x for y in test_basenames])]

        os.mkdir(splitdir)
        subdirs = [splitdir + "/" + x for x in ["images", "labels"]]
        for subdir in subdirs:
            os.mkdir(subdir)

        # Copy images
        for image in images:
            shutil.copy(all_image_path + "/" + image, splitdir + "/images/" + image)

        # Copy labels
        for label in labels:
            shutil.copy(all_labels_path + "/" + label, splitdir + "/labels/" + label)

save_folder = data_path_shuf if shuffle_data else data_path
yaml_dir = os.path.join(save_folder, "data.yaml")




## Set the name of the project for the model fitting
name = os.path.basename(save_folder) + "_" + model_size.title() + "_" + str(img_size) # name of the model

name_val = name + "_val" # validation name
name_pred = name + "_pred" # Prediction output name

data = yaml_dir # path to yaml file


"""## Step 1. Run model training"""

# clear garbage
gc.collect()

# Set model parameters
model_param = {
    "project": project,
    "name": name,
    "data": data,
    "epochs": epochs,
    "batch": batch_size,
    "imgsz": img_size,
    "optimizer": optimizer,
    "device": device,
    "verbose": verbose,
    "exist_ok": exist_ok,
    "patience": patience,
    "cache": cache,
    "workers":  workers,
    "hsv_h":  hsv_h,
    "hsv_s":  hsv_s,
    "hsv_v":  hsv_v,
    "degrees": degrees,
    "translate":  translate,
    "scale": scale,
    "shear": shear,
    "perspective":  perspective,
    "flipud":  flipud,
    "fliplr": fliplr,
    "mosaic": mosaic,
    "mixup": mixup,
    "copy_paste": copy_paste,
    "augment": True
}

# model training
# If resume training is false, train a new model
if not resume_training:
    model = YOLO('yolov8' + model_size + '.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8' + model_size + '.pt').load(weights)

    # train the model
    results = model.train(**model_param)

else:
    path_last = os.path.join(project, name.replace("_shuffled", "") + "_shuffled", "weights", "last.pt")
    model = YOLO(path_last)
    results = model.train(resume = True, **model_param)

"""## Step 2. Run model validation"""

# Load the last model
model_path = proj_dir + "/" + project + "/" + name + "/weights/best.pt"
print(model_path)
model = YOLO(model_path)

# validate the model
val_results = model.val(
    data = data,
    project = project,
    name = name_val,
    batch = batch_size,
    imgsz = img_size,
    verbose = verbose,
    exist_ok = exist_ok,
    workers = workers
)


# Save the results of model validation to a CSV
precision = val_results.box.p
recall = val_results.box.r
f1 = val_results.box.f1
map50 = val_results.box.ap50
map5090 = val_results.box.ap

keys = [key for key in val_results.names if val_results.names[key] == "sound" or val_results.names[key] == "rotten"]

# Create a dictionary with the metrics
metrics_dict = {
    'Class_index': keys,
    'Class': [val_results.names[key] for key in keys],
    'Precision': precision,
    'Recall': recall,
    'F1': f1,
    'mAP50': map50,
    'mAP50-90': map5090
}

# Convert to a pandas DataFrame
df = pd.DataFrame(metrics_dict)
# Save the DataFrame to a CSV file
df_filename = os.path.join(str(val_results.save_dir), "validation_results.csv")
df.to_csv(df_filename, index=False)

print("Validation results saved to " + df_filename)

