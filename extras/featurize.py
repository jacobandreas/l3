#!/usr/bin/env python2

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception, vgg
from tensorflow.contrib import slim

#image_size = inception.inception_v3.default_image_size
image_size = vgg.vgg_16.default_image_size

BATCH_SIZE = 100


t_input = tf.placeholder(tf.float32, (None, 64, 64, 3))

t_proc = tf.image.resize_bilinear(t_input, [image_size, image_size], align_corners=False)
t_proc = tf.subtract(t_proc, 0.5)
t_proc = tf.multiply(t_proc, 2.0)

logits, layers = vgg.vgg_16(t_proc, is_training=False)
#t_output = layers["vgg_16/fc7"]
#for name, tensor in layers.items():
#    print name, tensor.shape
#exit()

#t_output_pre = layers["vgg_16/pool5"]
#t_output = tf.reduce_mean(t_output_pre, axis=(1, 2))
#N_FEATS = 512

t_output = layers["vgg_16/pool5"]
t_output = slim.avg_pool2d(t_output, (3, 3), 2)
t_output = slim.flatten(t_output)
N_FEATS = t_output.shape[1]

#t_output = layers["vgg_16/fc7"]
#t_output = slim.flatten(t_output)
#N_FEATS = t_output.shape[1]

init_fn = slim.assign_from_checkpoint_fn(
    "/data/jda/concepts_features/vgg_16.ckpt",
    slim.get_model_variables())

with tf.Session() as session:
    init_fn(session)

    for fold in ("train", "val", "test", "val_same", "test_same"):
        print fold
        print "EXAMPLES"
        with open("%s/examples.npy" % fold) as ex_f:
            ex = np.load(ex_f)
        n_inp = ex.shape[0]
        n_ex = ex.shape[1]
        ex_feats = np.zeros((n_inp, n_ex, N_FEATS))
        for i in range(0, n_inp, BATCH_SIZE/10):
            if i % 1000 == 0:
                print i
            batch = ex[i:i+BATCH_SIZE/10, ...]
            n_batch = batch.shape[0]
            batch = batch.reshape((n_batch * n_ex, batch.shape[2],
                batch.shape[3], batch.shape[4]))
            feats = session.run(t_output, {t_input: batch})
            feats = feats.reshape((n_batch, n_ex, N_FEATS))
            ex_feats[i:i+BATCH_SIZE/10, ...] = feats
        np.save("%s/examples.feats.npy" % fold, ex_feats)

        print "INPUTS"
        with open("%s/inputs.npy" % fold) as inp_f:
            inp = np.load(inp_f)
        n_inp = inp.shape[0]
        inp_feats = np.zeros((n_inp, N_FEATS))
        for i in range(0, n_inp, BATCH_SIZE):
            if i % 1000 == 0:
                print i
            batch = inp[i:i+BATCH_SIZE, ...]
            feats = session.run(t_output, {t_input: batch})
            feats = feats.reshape((-1, N_FEATS))
            inp_feats[i:i+BATCH_SIZE, :] = feats
        np.save("%s/inputs.feats.npy" % fold, inp_feats)

