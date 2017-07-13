#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --predict_hyp=true \
  --infer_hyp=true \
  --learning_rate 0.001 \
  --infer_by_likelihood \
  --n_sample_hyps=10 \
  --train \
  --n_epochs=100 \
  --test \
  > train2.out \
  2> train2.err

  #--use_true_hyp=true \
