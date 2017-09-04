#!/bin/sh

export CUDA_VISIBLE_DEVICES=""

python -u ../../rl.py \
  --test \
  --n_epochs=20 \
  --predict_hyp=false \
  --infer_hyp=false \
  --restore="../rl_ex" \
  > eval.out \
  2> eval.err

