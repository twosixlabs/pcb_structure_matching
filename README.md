# PCB Structure Matching

[Paper](XXX) - [Model Weights](https://drive.google.com/file/d/1WMV33SLfCNG7BiuwMfSm0KvW-jxkkRAk/view?usp=sharing) - [Matched Dataset for Evaluation](https://drive.google.com/file/d/1h6Wt9YmRpldLmUeDB7fM4U-sZSkQjGbj) - [Analysis Intermediate Values](https://drive.google.com/drive/u/0/folders/1JktWOJz4ewmXGd13wHKGcN-RLbuQ6UJ1) - [Only Pre-trained Model Weights](https://drive.google.com/file/d/1PVZcf-OhmC1n73QuZe2zTB1_LRMwfjtG/view?usp=sharing)



# Installation

1. Install pytorch (>1.10) with your desired cuda version
2. Clone this repo
3. Install it (optionally with -e to make the packages in src editable and with \[scripts\] if you want to use the scripts)
```
pip install -e pcb_comp_detector[scripts]
```


# General Usage

## Use trained component detector

Use a component detector with model weights, a config yaml file, and a classes json file:

```
# Use model

import cv2
import pcb_comp_detector as cd

detector, cfg, classes = cd.load_model("path/to/model_weights.pth", device="cpu")  # cfg.yaml and classes.json should be in the same directory as model.pth
im = cv2.imread("path/to/image.jpg")

detections = detector({"image": im})["instances"].to("cpu")


# Visualize detections (requires matplotplib, installed with [scripts])

from pcb_comp_detector.utils import draw_detections

im_annotated = draw_detections(im, detections, classes)
cv2.imwrite("im_annotated.jpg", im_annotated)
```

## Train component detector

### 1. Format data

In order to train a component detector, the datasets must be downloaded and formatted. This demonstrates formatting [FICS PCB](https://www.trust-hub.org/#/data/pcb-images) and [PCB WACV](https://sites.google.com/view/graph-pcb-detection-wacv19) and combining them but there is also a script for PCB DSLR, which only contains annotations for ICs.

##### Format PCB WACV

First, download and unzip [PCB WACV](https://sites.google.com/view/graph-pcb-detection-wacv19). Then, from the parent directory, run:

```
python data_scripts/format_dataset.py wacv [PATH_TO_UNZIPPED_FOLDER] --visualize 10
```

JSON files for train set, test set, and classes will be created in datasets/wacv/6class_20t (6 classes because this script will, by default, use only the six classes present in FICS PCB to facilitate combining the datasets; 20t means 80/20 train/test split). This command will save 10 images from the dataset with annotations visualized on them into the dataset folder. Take a look at them to make sure they look right.

##### Format FICS PCB

First, download and unzip [FICS PCB](https://www.trust-hub.org/#/data/pcb-images). This one requires first reorganizing them into a flat folder. Then, run:

```
python data_scripts/format_dataset.py fics [PATH_TO_UNZIPPED_FOLDER] --visualize 10
```

As with PCB WACV, check the visualizations to make sure the annotations are being loaded correctly.

The dataset files should be created in datasets/fics/6class_20t

##### Combine datasets

Now, combine the two datasets:

```
python data_scripts/combine_datasets.py datasets/wacv/6class_20t/ datasets/fics/6class_20t/ --visualize 10
```

This will output to datasets/combined/wacv_fics_6class_20t. Again, check the visualizations to make sure everything is good (especially that the classes are correct).


### 2. Train detector

Train a detector using scripts/train_detector.py with --dataset datasets/combined/wacv_fics_6class. Run...

```
python scripts/train_detector.py --help
```

...to learn how to specify all other arguments.

### 3. Evaluate detector

Use scripts/evaluate_detector.py.

### 4. Share detector

Detector can be used by others by sharing the config file, model weights, and classes.json from the output folder.

# Reproduce Analyses in Paper

The script in 'scripts/evaluate_comparison_algs.py' runs the top k similarities analysis as seen in the paper.

Specify which image comparison techniques you want to compare with --approaches, where "emb_\[MODEL_NAME\]" tests an embedding-based approach for a specific model and "sm_\[MODEL_PATH\]" tests structure matching with an object detector model at the given file path. For similar set comparison mode (which should be used if using our matched dataset), add the "-ss" flag. Example:
```
python scripts/evaluate_comparison_algs.py IMG_DIR -ss --approaches emb_dino emb_clip sm_pcb_comp_detector
```

### Download intermediate calculations

The script saves intermediate computations (first, the embeddings or detections, and second, the percentiles for each data sample). Because we have not explicitly released the algorithm code, in order to compare against our structure matching algorithm, you need to download the intermediate results (linked at the top) and specify --emb_dir as that directory. This folder also contains our intermediate embeddings and percentiles for the embedding-based approaches, which can be used in order to avoid having to download and run all the models. However, if you want to recalculate those, you can simply delete or rename any of the intermediate result files (or not specify --emb_dir).

There is an empty "compare" function in "src/pcb_structure_matching/structure_matching.py" in case you want to implement your own comparison algorithm given the same inputs as ours.


# Scripts

### Object Detection Scripts

- train_detector.py: Train a model
- evaluate_detector.py: Evaluate model on the holdout set in its config
- run_detector.py: Run a model and save its output for a folder of images

### Object Detection Data Scripts

- format_dataset.py: Formats dataset annotations and resizes the images if too large
- combine_datasets.py: Combines two dataset descriptors/annotations as generated by format_*_anno.py
  (e.g. ../datasets/fics/6class and ../datasets/wacv/6class combines to ../datasets/combined/wacv_fics_6class)
- convert_tif_jpg.py: Converts tif images to jpg images and replaces them
- change_dataset_path.py: If moving dataset image file location, use this script to change the dataset jsons to reflect the new location

### Structure Matching Scripts

- structure_matching_debug.py: Runs structure matching in debug mode on the first two images in a folder. Requires that you run_detector.py on the folder first and provide the detections.json.
- evaluate_comparison_algs.py: Runs analysis from the paper to generate top k similarities curve and areas under it using specified approaches.
- (data script) format_ss_data.py: Formats "similar sets" data for use in evaluate_comparison_algs.py, which requires nested folders. This script converts a flat image folder with a json file denoting the sets to a nested folder structure.
