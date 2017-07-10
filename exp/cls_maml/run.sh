#!/bin/sh

python -u ../../cls.py \
  --predict_hyp=false \
  --infer_hyp=false \
  --infer_maml=false \
  --learning_rate 0.00001 \
  --train \
  --n_epochs=2000 \
  --test \
  #> train.out \
  #2> train.err
  #--learning_rate 0.0003 \

