#!/usr/bin/env python2

from rl_models import Policy
from tasks import nav

import sys
import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean("train", False, "do a training run")
gflags.DEFINE_boolean("test", False, "do a testing run")
gflags.DEFINE_integer("n_epochs", 0, "number of epochs to run for")
gflags.DEFINE_integer("n_batch", 5000, "batch size")
gflags.DEFINE_float("discount", 0.95, "discount factor")
gflags.DEFINE_integer("max_steps", 100, "max rollout length")

def main():
    task = nav.NavTask()
    policy = Policy(task)

    if FLAGS.train:
        for i_epoch in range(FLAGS.n_epochs):
            buf = []
            total_rew = 0
            n_rollouts = 0
            while len(buf) < FLAGS.n_batch:
                rollout, rew = do_rollout(task, policy)
                buf.extend(rollout)
                total_rew += rew
                n_rollouts += 1
            print total_rew / n_rollouts, n_rollouts
            print policy.train(buf)
            print

def do_rollout(task, policy):
    state = task.sample_train()
    buf = []
    for _ in range(FLAGS.max_steps):
        action, = policy.act((state,))
        state_, reward, stop = state.step(action)
        buf.append((state, action, state_, reward))
        if stop:
            break
        state = state_
    discounted_buf = []
    forward_r = 0
    total_r = 0
    for s, a, s_, r in reversed(buf):
        forward_r *= FLAGS.discount
        r_ = r + forward_r
        discounted_buf.append((s, a, s_, r_))
        forward_r += r
    if discounted_buf[0][3] > 0:
        total_r += 1.
    return discounted_buf, total_r

if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    main()
