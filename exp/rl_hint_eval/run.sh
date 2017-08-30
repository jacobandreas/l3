#!/bin/sh

export CUDA_VISIBLE_DEVICES=""

python -u ../../rl.py \
  --test \
  --n_epochs=100 \
  --max_steps=20 \
  --concept_prior=false \
  --predict_hyp=true \
  --infer_hyp=true \
  --restore="../rl_hint" \
  #> train.out \
  #2> train.err

