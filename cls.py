#!/usr/bin/env python2

import models
from models import ClsModel
from tasks import shapes

import gflags
import os
import sys

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean("train", False, "do a training run")
gflags.DEFINE_boolean("test", False, "evaluate on held-out concepts")
gflags.DEFINE_boolean("test_same", False, "evaluate on training concepts")
gflags.DEFINE_integer("n_epochs", 0, "number of epochs to run for")
gflags.DEFINE_integer("n_batch", 100, "batch size")
gflags.DEFINE_boolean("augment", False, "data augmentation")
models._set_flags()

def main():
    task = shapes.ShapeworldTask()
    model = ClsModel(task)

    if FLAGS.train:
        for i_epoch in range(FLAGS.n_epochs):
            e_loss = 0.
            for i_batch in range(100):
                batch = task.sample_train(FLAGS.n_batch, augment=FLAGS.augment)
                b_loss = model.train(batch)
                e_loss += b_loss

            batch = task.sample_train(FLAGS.n_batch, augment=FLAGS.augment)
            e_acc = model.predict(batch)

            v_batch = task.sample_val(same=False)
            e_v_acc = model.predict(v_batch)

            vs_batch = task.sample_val(same=True)
            e_vs_acc = model.predict(vs_batch)

            print("[iter]    %d" % i_epoch)
            print("[loss]    %01.4f" % e_loss)
            print("[trn_acc] %01.4f" % e_acc)
            print("[val_acc] %01.4f" % e_v_acc)
            print("[val_same_acc] %01.4f" % e_vs_acc)
            print("[val_mean_acc] %01.4f" % ((e_v_acc + e_vs_acc) / 2))
            print

            if i_epoch % 10 == 0:
                model.save()

    if FLAGS.test:
        v_batch = task.sample_val(same=False)
        e_v_acc = model.predict(v_batch)

        t_batch = task.sample_test(same=False)
        e_t_acc = model.predict(t_batch)

        print("[FINAL val_acc] %01.4f" % e_v_acc)
        print("[FINAL tst_acc] %01.4f" % e_t_acc)

    if FLAGS.test_same:
        v_batch = task.sample_val(same=True)
        e_v_acc = model.predict(v_batch)

        t_batch = task.sample_test(same=True)
        e_t_acc = model.predict(t_batch)

        print("[FINAL val_same_acc] %01.4f" % e_v_acc)
        print("[FINAL tst_same_acc] %01.4f" % e_t_acc)

if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    main()
