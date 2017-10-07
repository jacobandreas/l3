#!/bin/sh

python -u ../../cls.py \
  --task=birds \
  --learning_rate 0.0001 \
  --predict_hyp=true \
  --infer_hyp=true \
  --train \
  --n_epochs=30 \
  --test \
  > train.out \
  2> train.err

