#!/bin/sh

python -u ../../rl.py \
  --train \
  --n_epochs=300 \
  --predict_hyp=true \
  --use_expert=true \
  > train.out \
  2> train.err

