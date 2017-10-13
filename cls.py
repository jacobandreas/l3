#!/usr/bin/env python2

import models
from models import ClsModel, SimModel

import gflags
import os
import shutil
import sys

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean("train", False, "do a training run")
gflags.DEFINE_boolean("test", False, "evaluate on held-out concepts")
gflags.DEFINE_boolean("test_same", False, "evaluate on training concepts")
gflags.DEFINE_boolean("vis", False, "generate output visualizations")
gflags.DEFINE_integer("n_epochs", 0, "number of epochs to run for")
gflags.DEFINE_integer("n_batch", 100, "batch size")
gflags.DEFINE_boolean("augment", False, "data augmentation")
gflags.DEFINE_string("task", "shapes", "which task to use")
gflags.DEFINE_string("model", "cls", "which model to use")
gflags.DEFINE_boolean("train_on_eval", False, "fine-tune")
models._set_flags()

def main():
    if FLAGS.task == "shapes":
        from tasks import shapes
        task = shapes.ShapeworldTask()
    elif FLAGS.task == "birds":
        from tasks import birds
        task = birds.BirdsTask()
    else:
        assert False

    if FLAGS.model == "cls":
        model = ClsModel(task)
    elif FLAGS.model == "sim":
        model = SimModel(task)
    else:
        assert False

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

    if FLAGS.train_on_eval:
        for i_batch in range(FLAGS.n_epochs):
            vbatch = task.sample_val_safe(same=False)
            vsbatch = task.sample_val_safe(same=True)
            batch = task.sample_test_safe(same=False)
            sbatch = task.sample_test_safe(same=True)
            model.train(vbatch, task_only=True)
            model.train(vsbatch, task_only=True)
            model.train(batch, task_only=True)
            model.train(sbatch, task_only=True)

            vacc = model.predict(vbatch)
            vsacc = model.predict(vsbatch)
            acc = model.predict(batch)
            sacc = model.predict(sbatch)

            t_batch = task.sample_test()
            t_sbatch = task.sample_test(same=True)
            t_acc = model.predict(t_batch)
            t_sacc = model.predict(t_sbatch)

            v_batch = task.sample_val()
            v_sbatch = task.sample_val(same=True)
            v_acc = model.predict(v_batch)
            v_sacc = model.predict(v_sbatch)

            print("[FT val_acc] %01.4f (%01.4f)" % (v_acc, vacc))
            print("[FT val_same_acc] %01.4f (%01.4f)" % (v_sacc, vsacc))
            print("[FT val_mean_acc] %01.4f" % ((v_acc + v_sacc)/2))
            print("[FT tst_mean_acc] %01.4f" % ((t_acc + t_sacc)/2))

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

    if FLAGS.vis:
        v_batch = task.sample_val(same=True)
        preds, labels, hyps = model.predict(v_batch, debug=True)
        if os.path.exists("vis"):
            shutil.rmtree("vis")
        os.mkdir("vis")
        for i in range(10):
            task.visualize(v_batch[i], hyps[i], preds[i], "vis/%d" % i)

if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    main()
