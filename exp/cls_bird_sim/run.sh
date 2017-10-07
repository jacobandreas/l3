#!/bin/sh

python -u ../../cls.py \
  --task=birds \
  --model=sim \
  --learning_rate 0.001 \
  --train \
  --n_epochs=10 \
  --test \
  > train.out \
  2> train.err

