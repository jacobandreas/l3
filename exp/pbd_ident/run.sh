#!/bin/sh

python -u ../../pbd.py \
  --hint_type=none \
  --model=identity \
  --hint_type=none \
  --predict_hyp=false \
  --infer_hyp=false \
  --test \
  > eval.out \
  2> eval.err

