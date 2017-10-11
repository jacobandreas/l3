#!/bin/sh

export PYTHONPATH=".:../../../shapeworld"

python -u ../../cls.py \
  --restore="../cls_hint/model.chk" \
  --predict_hyp=true \
  --infer_hyp=true \
  --vis \
  --n_sample_hyps=10 \
  #> vis.out \
  #2> vis.err

