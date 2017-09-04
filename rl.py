#!/usr/bin/env python2

import rl_models
from rl_models import Policy
from tasks import minicraft2
from tasks import nav
from misc import util

import sys
import gflags

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean("train", False, "do a training run")
gflags.DEFINE_boolean("test", False, "do a testing run")
gflags.DEFINE_integer("n_epochs", 0, "number of epochs to run for")
gflags.DEFINE_integer("n_batch", 5000, "batch size")
gflags.DEFINE_float("discount", 0.9, "discount factor")
gflags.DEFINE_integer("max_steps", 50, "max rollout length")
gflags.DEFINE_boolean("use_expert", False, "DAGGER training with expert feedback")
rl_models._set_flags()

N_PAR = 10

random = util.next_random()

def main():
    task = nav.NavTask()
    policy = Policy(task)

    if FLAGS.train:
        if FLAGS.n_epochs == 0:
            policy.save()

        for i_epoch in range(FLAGS.n_epochs):
            total_rew = 0
            total_err = 0
            n_rollouts = 0
            for i in range(10):
                buf = []
                while len(buf) < FLAGS.n_batch:
                    states = [task.sample_train() for _ in range(N_PAR)]
                    use_expert = FLAGS.use_expert and random.rand() < (0.95 ** i_epoch)
                    rollouts, rews = do_rollout(task, policy, states, expert=use_expert)
                    for rollout, rew in zip(rollouts, rews):
                        buf.extend(rollout)
                        if not use_expert:
                            total_rew += rew
                            n_rollouts += 1
                if FLAGS.use_expert:
                    total_err += policy.train_dagger(buf)
                else:
                    total_err += policy.train(buf)

            test_states = [task.sample_test(random.choice(task.test_ids)) for _ in range(100)]
            _, test_rews = do_rollout(task, policy, test_states, vis=False)
            total_test_rew = sum(test_rews)

            print("[iter]    %d" % i_epoch)
            print("[loss]    %01.4f" % (total_err / 10))
            if n_rollouts > 0:
                print("[trn_rew] %01.4f" % (total_rew / n_rollouts))
            print("[tst_rew] %01.4f" % (total_test_rew / len(test_rews)))
            hyps = policy.sample_hyps(3)
            for hyp in hyps:
                print " ".join(policy.task.vocab.get(w) for w in hyp)
            print

            if i_epoch % 10 == 0:
                policy.save()

    if FLAGS.test:
        for i_datum in task.test_ids:
            print "TEST DATUM", i_datum
            policy.restore(FLAGS.restore)
            policy.reset()
            state = task.sample_test(i_datum, erase_hint=False)
            print " ".join(task.vocab.get(w) for w in state.instruction)
            for i_epoch in range(FLAGS.n_epochs):
                total_rew = 0.
                n_rollouts = 0
                buf = []
                episodes = []
                while len(buf) < FLAGS.n_batch:
                    states = [task.sample_test(i_datum, erase_hint=True) for _ in range(N_PAR)]
                    states = policy.annotate(states)
                    rollouts, rews = do_rollout(task, policy, states, expert=False)
                    total_rew += sum(rews)
                    n_rollouts += len(rews)
                    for rollout, rew in zip(rollouts, rews):
                        buf.extend(rollout)
                        episodes.append((rollout, rew))

                policy.adapt(buf, episodes)
                print("[FINAL tst_rew] %01.4f" % (total_rew / n_rollouts))

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
            action = state.expert_a if expert else actions[i_state]
            state_, reward, stop = state.step(action)
            bufs[i_state].append((state, action, state_, reward))
            states[i_state] = state_
            if stop:
                done[i_state] = True
                if i_state == 0 and vis:
                    print state_.render()
        if all(done):
            break

    discounted_bufs = []
    total_rs = []
    for buf in bufs:
        forward_r = 0.
        total_r = 0.
        discounted_buf = []
        for s, a, s_, r in reversed(buf):
            forward_r *= FLAGS.discount
            r_ = r + forward_r
            discounted_buf.append((s, a, s_, r_))
            forward_r += r
            total_r += r
        discounted_bufs.append(discounted_buf)
        total_rs.append(total_r)
    return discounted_bufs, total_rs

if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    main()
