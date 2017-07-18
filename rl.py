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

N_PAR = 10

def main():
    task = nav.NavTask()
    policy = Policy(task)

    if FLAGS.train:
        for i_epoch in range(FLAGS.n_epochs):
            buf = []
            total_rew = 0
            n_rollouts = 0
            while len(buf) < FLAGS.n_batch:
                rollouts, rews = do_rollout(task, policy)
                for rollout, rew in zip(rollouts, rews):
                    buf.extend(rollout)
                    total_rew += rew
                    n_rollouts += 1
            print total_rew / n_rollouts, n_rollouts
            print policy.train(buf)
            print

def do_rollout(task, policy):
    #state = task.sample_train()
    states = [task.sample_train() for _ in range(N_PAR)]
    bufs = [[] for _ in range(N_PAR)]
    done = [False for _ in range(N_PAR)]
    for _ in range(FLAGS.max_steps):
        actions = policy.act(states)
        for i_state in range(N_PAR):
            if done[i_state]:
                continue
            state_, reward, stop = states[i_state].step(action)
            bufs[i].append((state, action, state_, reward))
            states[i_state] = state_
            if stop:
                done[i] = True
        if all(done):
            break

    discounted_bufs = []
    total_rs = []
    for buf in bufs:
        forward_r = 0
        total_r = 0
        for s, a, s_, r in reversed(buf):
            forward_r *= FLAGS.discount
            r_ = r + forward_r
            discounted_buf.append((s, a, s_, r_))
            forward_r += r
        if discounted_buf[0][3] > 0:
            total_r += 1.
        discounted_bufs.append(buf)
        total_rs.append(total_r)
    return discounted_bufs, total_rs

if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    main()
