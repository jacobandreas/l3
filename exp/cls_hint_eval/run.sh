#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --restore ../cls_hint/model.chk \
  --predict_hyp=true \
  --infer_hyp=true \
  --test \
  --test_same \
  --use_true_hyp=false \
  --n_sample_hyps=50 \
  #> eval.out \
  #2> eval.err

