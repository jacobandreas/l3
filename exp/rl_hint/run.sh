#!/bin/sh

python -u ../../rl.py \
  --train \
  --n_epochs=250 \
  --predict_hyp=true \
  --infer_hyp=true \
  --use_expert=true \
  --concept_prior=false \
  > train.out \
  2> train.err

