#!/usr/bin/env bash
OPTS="--model_dir=/home/user/bf/diff \
      --results_dir=/home/user/bf/diff \
      --mapping_file=./datasets/breakfast/mapping.txt \
      --vid_list_file_test=./datasets/breakfast/splits/test.split4.bundle \
      --vid_list_file=./datasets/breakfast/splits/train.split4.bundle \
      --gt_path=./datasets/breakfast/groundTruth/ \
      --features_path=/home/user/bf/features/ \
      --split=4 \
      --ds=bf \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --part_obs \
      --layer_type gated \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 50 \
      --diff_obj pred_x0 \
      --num_samples 25 \
      --action=val \
      --bz=1 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=85 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.4 \
      --sample_rate=3"

python ./src/main.py $OPTS



