#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --learning_rate 0.0001 \
  --predict_hyp=true \
  --infer_hyp=true \
  --restore="../cls_task/model.chk" \
  --n_epochs=4 \
  --use_true_hyp=true \
  --use_task_hyp=true \
  --augment=true \
  --train_on_eval \
  > train.out \
  2> train.err

