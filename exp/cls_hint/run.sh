#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --learning_rate 0.001 \
  --predict_hyp=true \
  --infer_hyp=true \
  --train \
  --n_epochs=13 \
  --test \
  --test_same \
  --use_true_hyp=false \
  > train.out \
  2> train.err

