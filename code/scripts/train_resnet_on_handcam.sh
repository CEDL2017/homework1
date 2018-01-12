#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a resnet model on the HandCam training set.
# 2. Evaluates the model on the HandCam validation set.
#
# Usage:
# ./scripts/train_resnet_on_handcam.sh


# Where the dataset is saved to.
DATASET_DIR=/home/nvlab/cedl/hw01

if [ -z "$1" ]
then
set 0
fi
cls=( fa ges obj )
cns=(2, 13, 24)
netn=mobilenet
# mobilenet
# Where the checkpoint and logs will be saved to.
# m: ori train; m2: pre+finetune-last; m3: pre+f-l+f-all
# m4: pre+f-all bl setting
# m5: pre+f-last (20k)
# v: ori train
TRAIN_DIR=./output/handcam${cls[$1]}/${netn}5
CP_PATH=./output/pret
#CP_PATH=./output/handcam${cls[$1]}/${netn}
#CP_PATH=./output/handcam${cls[$1]}/${netn}2
mkdir -p ${TRAIN_DIR}

# Run training.
# ORI: lr=0.1, nstep=1000000
#  --trainable_scopes=MobileNet/fc_16 \
:< '
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --ignore_missing_vars=True \
  --dataset_name=handcam${cls[$1]} \
  --checkpoint_exclude_scopes=MobileNet/fc_16 \
  --checkpoint_path=${CP_PATH} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${netn} \
  --preprocessing_name=${netn} \
  --width_multiplier=1.0 \
  --max_number_of_steps=20000 \
  --batch_size=64 \
  --que_batch=20 \
  --save_interval_secs=240 \
  --save_summaries_secs=240 \
  --log_every_n_steps=100 \
  --optimizer=yellowfin \
  --rmsprop_decay=0.9 \
  --opt_epsilon=1.0\
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --momentum=0.9 \
  --num_epochs_per_decay=30.0 \
  --weight_decay=0.0 \
  --num_clones=1 \
  --gpu_memp=0.5
# '
# Run evaluation

#:< '
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=handcam${cls[$1]} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${netn} \
  --gpu_memp=0.7
# '
