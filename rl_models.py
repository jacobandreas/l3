from models import _encode
from net import _mlp, _linear, _embed_dict
import util

import numpy as np
import tensorflow as tf

N_EMBED = 64
#N_HIDDEN = 256
N_HIDDEN = 64

random = util.next_random()

class Policy(object):
    def __init__(self, task):
        self.task = task

        #self.t_state = tf.placeholder(tf.float32, (None,) + task.feature_shape)
        self.t_state = tf.placeholder(tf.float32, (None, task.n_features))
        self.t_action = tf.placeholder(tf.int32, (None,))
        self.t_reward = tf.placeholder(tf.float32, (None,))
        self.t_hint = tf.placeholder(tf.int32, (None, None))
        self.t_hint_len = tf.placeholder(tf.int32, (None,))

        t_hint_vecs = tf.get_variable(
                "hint_vec", (len(task.vocab), N_EMBED),
                initializer=tf.uniform_unit_scaling_initializer())
        t_hint_repr = _encode(
                "hint_repr", self.t_hint, self.t_hint_len, t_hint_vecs)
        #t_hint_repr = tf.reduce_mean(_embed_dict(self.t_hint, t_hint_vecs), axis=1)

        #t_features = tf.concat((self.t_state, t_hint_repr), axis=1)
        with tf.variable_scope("features"):
            t_features = _mlp(self.t_state, (N_HIDDEN, N_HIDDEN), (tf.nn.tanh, tf.nn.tanh))
        with tf.variable_scope("hint_param"):
            t_hint_param = _linear(t_hint_repr, N_HIDDEN * task.n_actions)
            t_hint_mat = tf.reshape(t_hint_param, (-1, N_HIDDEN, task.n_actions))
        #self.t_score = _mlp(
        #        t_features,
        #        (N_HIDDEN, task.n_actions),
        #        (tf.nn.relu, None))
        self.t_score = tf.einsum("ij,ijk->ik", t_features, t_hint_mat)

        #with tf.variable_scope("hint_param"):
        #    n_r, n_c, n_chan = task.feature_shape
        #    t_n_batch = tf.shape(self.t_state)[0]

        #    t_hint_param = _linear(t_hint_repr, 5 * 5 * n_chan)
        #    t_hint_kernel = tf.reshape(t_hint_param, (t_n_batch, 5, 5, n_chan))

        #    t_hint_kernel_d = tf.transpose(t_hint_kernel, (1, 2, 0, 3))
        #    t_hint_kernel_d = tf.reshape(t_hint_kernel_d, (5, 5, t_n_batch * n_chan, 1))

        #    t_state_d = tf.transpose(self.t_state, (1, 2, 0, 3))
        #    t_state_d = tf.reshape(t_state_d, (1, n_r, n_c, t_n_batch * n_chan))

        #    t_conv_d = tf.nn.depthwise_conv2d(t_state_d, t_hint_kernel_d, (1, 1, 1, 1), "SAME")
        #    t_conv = tf.reshape(t_conv_d, (n_r, n_c, t_n_batch, n_chan))
        #    t_conv = tf.reduce_sum(t_conv, axis=3)
        #    t_conv = tf.transpose(t_conv, (2, 0, 1))
        #    t_features = tf.reshape(t_conv, (-1, n_r * n_c))
        #    #t_features = tf.reshape(self.t_state[:, :, :, 0], (-1, n_r * n_c))

        #with tf.variable_scope("score"):
        #    self.t_score = _linear(t_features, task.n_actions)
        #    self.t_logprob = tf.nn.log_softmax(self.t_score)

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

        optimizer = tf.train.AdamOptimizer(0.001)
        self.o_train = optimizer.minimize(self.t_loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

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
