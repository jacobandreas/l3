#!/bin/sh

python -u ../../cls.py \
  --model=sim \
  --learning_rate 0.001 \
  --predict_hyp=true \
  --infer_hyp=true \
  --train \
  --n_epochs=100 \
  --test \
  > train.out \
  2> train.err

