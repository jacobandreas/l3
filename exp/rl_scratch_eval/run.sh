#!/bin/sh

export CUDA_VISIBLE_DEVICES=""

python -u ../../rl.py \
  --test \
  --n_epochs=20 \
  --adapt_reprs=1 \
  --adapt_samples=1 \
  --predict_hyp=false \
  --infer_hyp=false \
  --restore="../rl_scratch" \
  > eval.out \
  2> eval.err

