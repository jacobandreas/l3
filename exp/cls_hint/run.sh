#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --learning_rate 0.003 \
  --predict_hyp=true \
  --infer_hyp=true \
  --train \
  --n_epochs=50 \
  --test \
  --use_true_hyp=false \
  --infer_by_likelihood=true \
  --n_sample_hyps=1 \
  > train.out \
  2> train.err
  #--n_sample_hyps=5 \

