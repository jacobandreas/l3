#!/bin/sh

#export CUDA_VISIBLE_DEVICES=""

python -u ../../pbd.py \
  --hint_type=none \
  --predict_hyp=false \
  --infer_hyp=true \
  --use_true_hyp=true \
  --use_task_hyp=true \
  --learning_rate 0.001 \
  --train \
  --n_epochs=200 \
  --test \
  > train.out \
  2> train.err

