#!/usr/bin/env bash
OPTS="--model_dir=/home/user/bf/determ \
      --results_dir=/home/user/bf/determ \
      --mapping_file=./datasets/breakfast/mapping.txt \
      --vid_list_file_test=./datasets/breakfast/splits/test.split1.bundle \
      --vid_list_file=./datasets/breakfast/splits/train.split1.bundle \
      --gt_path=./datasets/breakfast/groundTruth/ \
      --features_path=/home/user/bf/features/ \
      --split=1 \
      --use_features \
      --use_inp_ch_dropout \
      --part_obs \
      --layer_type gated \
      --action=train \
      --ds=bf \
      --bz=16 \
      --lr=0.0005 \
      --model=pred-tcn \
      --num_epochs=100 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.4 \
      --sample_rate=3"

python ./src/main.py $OPTS



