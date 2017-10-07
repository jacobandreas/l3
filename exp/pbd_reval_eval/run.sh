#!/bin/sh

python -u ../../pbd.py \
  --hint_type=re \
  --predict_hyp \
  --infer_hyp \
  --infer_by_likelihood \
  --n_sample_hyps=500 \
  --use_true_eval \
  --restore="../pbd_reval/model.chk" \
  --test \
  > eval.out \
  2> eval.err

python -u ../../pbd.py \
  --hint_type=re \
  --predict_hyp \
  --infer_hyp \
  --use_true_eval \
  --use_true_hyp \
  --restore="../pbd_reval/model.chk" \
  --test \
  > eval_gold.out \
  2> eval_gold.err
