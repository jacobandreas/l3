#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --predict_hyp=true \
  --infer_hyp=true \
  --learning_rate 0.001 \
  --train \
  --n_epochs=100 \
  --test \
  #> train.out \
  #2> train.err

