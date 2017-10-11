#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --learning_rate=0.0001 \
  --train \
  --n_epochs=45 \
  --predict_hyp=true \
  --infer_hyp=true \
  --test \
  --test_same \
  --use_true_hyp=true \
  --augment \
  > train_gold.out \
  2> train_gold.err
