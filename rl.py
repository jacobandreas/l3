#!/usr/bin/env python2

from rl_models import Policy
#from tasks import minicraft2
from tasks import nav
from misc import util

import sys
import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean("train", False, "do a training run")
gflags.DEFINE_boolean("test", False, "do a testing run")
gflags.DEFINE_integer("n_epochs", 0, "number of epochs to run for")
gflags.DEFINE_integer("n_batch", 100, "batch size")
gflags.DEFINE_float("discount", 0.95, "discount factor")
gflags.DEFINE_integer("max_steps", 100, "max rollout length")

N_PAR = 10

random = util.next_random()

def main():
    #task = minicraft.CraftTask()
    #task = minicraft2.Minicraft2World()
    task = nav.NavTask()
    policy = Policy(task)

    if FLAGS.train:
        for i_epoch in range(FLAGS.n_epochs):
            total_rew = 0
            total_err = 0
            n_rollouts = 0
            for i in range(10):
                buf = []
                while len(buf) < FLAGS.n_batch:
                    states = [task.sample_train() for _ in range(N_PAR)]
                    rollouts, rews = do_rollout(task, policy, states,
                            expert=random.randint(2))
                    for rollout, rew in zip(rollouts, rews):
                        buf.extend(rollout)
                        total_rew += rew
                        n_rollouts += 1
                total_err += policy.train_dagger(buf)
                #total_err += policy.train(buf)

            test_states = [task.sample_test() for _ in range(100)]
            _, test_rews = do_rollout(task, policy, test_states, vis=False)
            total_test_rew = sum(test_rews)

            print total_err / 10
            print total_rew / n_rollouts, n_rollouts
            print total_test_rew / len(test_rews)
            print

def do_rollout(task, policy, states, vis=False, expert=False):
    states = list(states)
    bufs = [[] for _ in states]
    done = [False for _ in states]
    for t in range(FLAGS.max_steps):
        actions = policy.act(states)
        for i_state in range(len(states)):
            if done[i_state]:
                continue
            state = states[i_state]
            if i_state == 0 and vis:
                print state.render()
                #print state.features[:, :, 0]
            action = state.expert_a if expert else actions[i_state]
            state_, reward, stop = state.step(action)
            bufs[i_state].append((state, action, state_, reward))
            states[i_state] = state_
            if stop:
                done[i_state] = True
                if i_state == 0 and vis:
                    print state_.render()
                    #print state.features[:, :, 0]
        if all(done):
            break

    discounted_bufs = []
    total_rs = []
    for buf in bufs:
        forward_r = 0
        total_r = 0
        discounted_buf = []
        for s, a, s_, r in reversed(buf):
            forward_r *= FLAGS.discount
            r_ = r + forward_r
            discounted_buf.append((s, a, s_, r_))
            forward_r += r
        #if discounted_buf[0][3] > 0:
        #if r > 0:
        if buf[-1][3] > 0:
            total_r += 1.
        discounted_bufs.append(discounted_buf)
        total_rs.append(total_r)
    return discounted_bufs, total_rs

if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    main()
