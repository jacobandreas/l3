#!/bin/sh

python -u ../../pbd.py \
  --hint_type=nl \
  --predict_hyp=true \
  --infer_hyp=false \
  --learning_rate 0.001 \
  --train \
  --n_epochs=2000 \
  --test \
  > train.out \
  2> train.err

