#!/bin/sh

python -u ../../pbd.py \
  --hint_type=re \
  --n_sample_hyps=10 \
  --predict_hyp \
  --infer_hyp \
  --infer_by_likelihood \
  --use_true_eval \
  --learning_rate 0.001 \
  --train \
  --n_epochs=50 \
  --test \
  > train.out \
  2> train.err

