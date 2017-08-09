#!/bin/sh

python -u ../../rl.py \
  --test \
  --n_epochs=100 \
  --rl_restore="../rl_hint" \
  #> train.out \
  #2> train.err

