
#!/usr/bin/env bash
OPTS="--model_dir=/home/user/assembly/determ \
      --results_dir=/home/user/assembly/determ \
      --gt_path=/home/user/assembly101 \
      --features_path=/home/user/assembly101 \
      --use_features \
      --use_inp_ch_dropout \
      --part_obs \
      --layer_type gated \
      --action=val \
      --split=1 \
      --ds=assembly \
      --bz=1 \
      --lr=0.0005 \
      --model=pred-tcn \
      --num_epochs=100 \
      --epoch=0 \
      --num_stages=5 \
      --obs_stages=0 \
      --ant_stages=5 \
      --num_layers=9 \
      --channel_dropout_prob 0.4 \
      --sample_rate=6"
python ./src/main.py $OPTS



