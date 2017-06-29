#!/bin/sh

python -u ../../pbd.py \
  --hint_type=nl \
  --predict_hyp=true \
  --infer_hyp=true \
  --infer_by_likelihood=true \
  --n_sample_hyps=100 \
  --restore="../pbd_rhint/model" \
  --test \
  > train.out \
  2> train.err

