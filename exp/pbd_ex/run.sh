#!/bin/sh

python -u ../../pbd.py \
  --hint_type=none \
  --predict_hyp=false \
  --infer_hyp=false \
  --learning_rate 0.001 \
  --train \
  --n_epochs=2000 \
  --test \
  > train.out \
  2> train.err

