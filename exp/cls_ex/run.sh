#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --predict_hyp=false \
  --infer_hyp=false \
  --learning_rate 0.001 \
  --train \
  --n_epochs=100 \
  --test \
  > train.out \
  2> train.err

