#!/bin/sh

#export CUDA_VISIBLE_DEVICES=""

python -u ../../pbd.py \
  --hint_type=none \
  --predict_hyp=false \
  --infer_hyp=true \
  --use_true_hyp=true \
  --use_task_hyp=true \
  --restore="../pbd_task/model.chk" \
  --learning_rate 0.001 \
  --n_epochs=200 \
  --train_on_eval \
  > train.out \
  2> train.err

