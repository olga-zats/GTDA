#!/usr/bin/env bash
OPTS="--model_dir=/home/user/bf/diff \
      --results_dir=/home/user/bf/diff\
      --mapping_file=./datasets/breakfast/mapping.txt \
      --vid_list_file_test=./datasets/breakfast/splits/test.split4.bundle \
      --vid_list_file=./datasets/breakfast/splits/train.split4.bundle \
      --gt_path=./datasets/breakfast/groundTruth/ \
      --features_path=/home/user/bf/features/ \
      --split=4 \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --layer_type gated \
      --part_obs \
      --num_diff_timesteps 1000 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=train \
      --ds=bf \
      --bz=16 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=100 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.4 \
      --sample_rate=3"

python ./src/main.py $OPTS


