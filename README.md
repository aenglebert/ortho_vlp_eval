# Evaluation scripts for "Self-supervised Vision-Language Deep Learning for Bone Radiography Analysis"

(preprint available soon)

## Installation

Install requirements with `pip install -r requirements.txt`

## Configuration

This repository uses hydra configuration files in the "config" folder.
You may need to adapt the location of your datasets in the respective configuration files in the config/dataset/ folder.

It uses the following datasets:
- MURA
- FracAtlas
- OAI

## Classification

To run the classification script, do as follow:

`python classification.py dataset=dataset eval_type=eval_type vision_model=vision_model`

while replacing "dataset", "eval_type" and "vision_model" by the desired configuration file name for each.


## Regression

To run the regression script, do as follow:

`python regression.py dataset=dataset eval_type=eval_type vision_model=vision_model`

while replacing "dataset", "eval_type" and "vision_model" by the desired configuration file name for each.


nb: The development of the detection script was abandoned, but we've kept it available in case anyone is interested.
