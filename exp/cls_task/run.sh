#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --learning_rate 0.0001 \
  --predict_hyp=true \
  --infer_hyp=true \
  --train \
  --n_epochs=100 \
  --n_sample_hyps=10 \
  --use_true_hyp=true \
  --use_task_hyp=true \
  --augment=true \
  > train.out \
  2> train.err

