#!/bin/sh

python -u ../../pbd.py \
  --hint_type=re \
  --predict_hyp=true \
  --infer_hyp=true \
  --infer_by_likelihood=true \
  --n_sample_hyps=100 \
  --restore="../pbd_rhint/model.chk" \
  --test \
  > eval.out \
  2> eval.err

