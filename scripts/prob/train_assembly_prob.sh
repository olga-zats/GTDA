#!/usr/bin/env bash
OPTS="--model_dir=/home/user/assembly/diff \
      --results_dir=/home/user/assembly/diff \
      --gt_path=/home/user/assembly101 \
      --features_path=/home/user/assembly101 \
      --use_features \
      --use_inp_ch_dropout \
      --layer_type gated \
      --part_obs \
      --conditioned_x0 \
      --num_diff_timesteps 1000 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=train \
      --ds=assembly \
      --bz=16 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=100 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob=0.3 \
      --sample_rate=6"

python ./src/main.py $OPTS


