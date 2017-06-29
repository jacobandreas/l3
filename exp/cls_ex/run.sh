#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --predict_hyp=true \
  --infer_hyp=true \
  --learning_rate 0.001 \
  --n_sample_hyps=10 \
  --train \
  --n_epochs=10 \
  --test

