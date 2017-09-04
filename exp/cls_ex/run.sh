#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --learning_rate 0.003 \
  --predict_hyp=false \
  --infer_hyp=false \
  --train \
  --n_epochs=50 \
  --test \
  --use_true_hyp=false \
  --infer_by_likelihood=false \
  > train.out \
  2> train.err

