#!/usr/bin/env bash
OPTS="--split=4 \
      --conditioned_x0 \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 50 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --num_samples 25 \
      --test_num_samples 25 \
      --ds=bf \
      --model=bit-diff-pred-tcn \
      --epoch=85 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --sample_rate=3"

python ./src/main_diff_evaluate.py $OPTS



