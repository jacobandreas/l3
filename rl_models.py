from models import _encode, Decoder
from models import N_HIDDEN as N_DEC_HIDDEN
from net import _mlp, _linear, _embed_dict
from misc import util

import gflags
import numpy as np
import tensorflow as tf

FLAGS = gflags.FLAGS
# TODO!!!
gflags.DEFINE_string("rl_restore", None, "model to restore")

N_EMBED = 64
N_HIDDEN = 64

random = util.next_random()

class Policy(object):
    def __init__(self, task):
        self.task = task

        self.t_state = tf.placeholder(tf.float32, (None, task.n_features))
        self.t_action = tf.placeholder(tf.int32, (None,))
        self.t_reward = tf.placeholder(tf.float32, (None,))
        self.t_hint = tf.placeholder(tf.int32, (None, None))
        self.t_hint_len = tf.placeholder(tf.int32, (None,))

        self.t_last_hyp = tf.placeholder(tf.int32, (None,), "last_hyp")
        self.t_last_hyp_hidden = tf.placeholder(tf.float32, (None, N_DEC_HIDDEN),
                "last_hyp_hidden")
        t_hyp_init = tf.get_variable("hyp_init", shape=(1, N_DEC_HIDDEN),
                initializer=tf.uniform_unit_scaling_initializer())
        self.t_n_batch = tf.shape(self.t_state)[0]
        #self.t_n_batch = tf.placeholder(tf.int32, ())
        t_hyp_tile = tf.tile(t_hyp_init, (self.t_n_batch, 1))

        t_hint_vecs = tf.get_variable(
                "hint_vec", (len(task.vocab), N_EMBED),
                initializer=tf.uniform_unit_scaling_initializer())
        t_hint_repr = tf.reduce_mean(_embed_dict(self.t_hint, t_hint_vecs), axis=1)
        self.hyp_decoder = Decoder(
                "decode_hyp", t_hyp_tile, self.t_hint, self.t_last_hyp,
                self.t_last_hyp_hidden, t_hint_vecs)

        with tf.variable_scope("features"):
            t_features = _mlp(self.t_state, (N_HIDDEN, N_HIDDEN), (tf.nn.tanh, tf.nn.tanh))
        with tf.variable_scope("hint_param"):
            t_hint_param = _linear(t_hint_repr, N_HIDDEN * task.n_actions)
            t_hint_mat = tf.reshape(t_hint_param, (-1, N_HIDDEN, task.n_actions))
        self.t_score = tf.einsum("ij,ijk->ik", t_features, t_hint_mat)

        self.t_logprob = tf.nn.log_softmax(self.t_score)
        t_prob = tf.nn.softmax(self.t_score)
        t_entropy = -tf.reduce_mean(tf.reduce_sum(t_prob * self.t_logprob, axis=1))

        with tf.variable_scope("baseline"):
            t_baseline = tf.squeeze(_linear(tf.stop_gradient(t_features), 1))
        
        t_chosen_logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.t_score, labels=self.t_action)
        t_loss_surrogate = -tf.reduce_mean(
                t_chosen_logprob * (self.t_reward - tf.stop_gradient(t_baseline)))
        t_baseline_err = tf.reduce_mean((t_baseline - self.t_reward) ** 2)

        self.t_loss = t_loss_surrogate + t_baseline_err - 0.001 * t_entropy
        self.t_dagger_loss = -tf.reduce_mean(t_chosen_logprob)

        if FLAGS.predict_hyp:
            self.t_loss = self.t_loss + self.hyp_decoder.t_loss
            self.t_dagger_loss = self.t_dagger_loss + self.hyp_decoder.t_loss

        optimizer = tf.train.AdamOptimizer(0.001)
        self.o_train = optimizer.minimize(self.t_loss)
        self.o_dagger_train = optimizer.minimize(self.t_dagger_loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if FLAGS.rl_restore is not None:
            self.restore(FLAGS.rl_restore)

    def load_hint(self, states):
        max_len = max(len(s.instruction) for s in states)
        hint = np.zeros((len(states), max_len))
        hint_len = np.zeros((len(states),))
        for i, state in enumerate(states):
            hint[i, :len(state.instruction)] = state.instruction
            hint_len[i] = len(state.instruction)
        return hint, hint_len

    def act(self, states):
        hint, hint_len = self.load_hint(states)
        feed_dict = {
            self.t_state: [s.features for s in states],
            self.t_hint: hint,
            self.t_hint_len: hint_len
        }
        logprobs, = self.session.run([self.t_logprob], feed_dict)
        probs = np.exp(logprobs)
        actions = []
        for i in range(len(states)):
            action = random.choice(self.task.n_actions, p=probs[i, :])
            actions.append(action)
        return actions

    def train(self, transitions):
        states, actions, _, rewards = zip(*transitions)
        features = [s.features for s in states]
        hint, hint_len = self.load_hint(states)
        feed_dict = {
            self.t_state: features,
            self.t_action: actions,
            self.t_reward: rewards,
            self.t_hint: hint,
            self.t_hint_len: hint_len
        }
        loss, _ = self.session.run([self.t_loss, self.o_train], feed_dict)
        return loss

    def train_dagger(self, transitions):
        states = [t[0] for t in transitions]
        actions = [s.expert_a for s in states]
        features = [s.features for s in states]
        hint, hint_len = self.load_hint(states)
        feed_dict = {
            self.t_state: features,
            self.t_action: actions,
            self.t_hint: hint,
            self.t_hint_len: hint_len
        }
        loss, _ = self.session.run([self.t_dagger_loss, self.o_dagger_train], feed_dict)
        return loss

    def sample(self):
        init = [self.task.vocab[self.task.START] for _ in range(100)]
        hyps = self.hyp_decoder.decode(
                init,
                self.task.vocab[self.task.STOP], 
                {self.t_n_batch: len(init)},
                self.session,
                temp=1)
        words = [" ".join(self.task.vocab.get(w) for w in hyp) for hyp in hyps]
        return words

    def adapt(self, transitions):
        pass

    def hypothesize(self, states):
        return [() for state in states]

    def save(self):
        self.saver.save(self.session, "model.chk")

    def restore(self, path):
        self.saver.restore(self.session, tf.train.latest_checkpoint(path))
