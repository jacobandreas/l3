#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

# surprisingly, data augmentation hurts here

python -u ../../cls.py \
  --learning_rate 0.0001 \
  --predict_hyp=true \
  --infer_hyp=false \
  --train \
  --n_epochs=28 \
  --test \
  --test_same \
  --augment=true \
  > train.out \
  2> train.err

