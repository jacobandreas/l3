#!/bin/sh

python -u ../../rl.py \
  --train \
  --n_epochs=750 \
  --test \
  #> train.out \
  #2> train.err

