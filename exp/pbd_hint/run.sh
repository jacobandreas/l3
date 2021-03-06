#!/bin/sh

#export CUDA_VISIBLE_DEVICES=""

python -u ../../pbd.py \
  --hint_type=nl \
  --predict_hyp=true \
  --infer_hyp=true \
  --n_sample_hyps=1 \
  --learning_rate 0.001 \
  --train \
  --n_epochs=750 \
  --test \
  > train.out \
  2> train.err

