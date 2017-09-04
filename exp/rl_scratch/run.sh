#!/bin/sh

python -u ../../rl.py \
  --train \
  --n_epochs=0 \
  --predict_hyp=false \
  --infer_hyp=false \
  --use_expert=true \
  > train.out \
  2> train.err

