from net import _mlp, _embed_dict, _linear
from misc import util

import scipy.misc
import gflags
import numpy as np
import sys
import tensorflow as tf
import os

FLAGS = gflags.FLAGS

def _set_flags():
    gflags.DEFINE_boolean("predict_hyp", False, "train to predict hypotheses")
    gflags.DEFINE_boolean("infer_hyp", False, "use hypotheses at test time")
    gflags.DEFINE_boolean("infer_by_likelihood", False, "use likelihood (rather than accuracy) to rank hypotheses")
    gflags.DEFINE_boolean("use_true_hyp", False, "predict using ground-truth description")
    gflags.DEFINE_integer("n_sample_hyps", 5, "number of hypotheses to sample")
    gflags.DEFINE_float("learning_rate", 0.001, "learning rate")
    gflags.DEFINE_string("restore", None, "model to restore")
    gflags.DEFINE_boolean("use_true_eval", False, "score with true evaluation function")
    gflags.DEFINE_boolean("use_task_hyp", False, "task as hypothesis")

USE_IMAGES = False

N_EMBED = 32
N_EMBED_WORD = 128
N_HIDDEN = 512
N_PBD_EX = 5
N_CLS_EX = 4

N_CONV1_SIZE = 5
N_CONV1_FILTS = 16
N_CONV2_SIZE = 3
N_CONV2_FILTS = 32
N_CONV3_SIZE = 3
N_CONV3_FILTS = 32

tf.set_random_seed(0)

def _encode(name, t_input, t_len, t_vecs, t_init=None):
    multi = len(t_input.get_shape()) == 3
    assert multi or len(t_input.get_shape()) == 2
    cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
    if multi:
        t_shape = tf.shape(t_input)
        t_n_batch, t_n_multi, t_n_toks = t_shape[0], t_shape[1], t_shape[2]
        t_input = tf.reshape(t_input, (t_n_batch*t_n_multi, t_n_toks))
        t_len = tf.reshape(t_len, (t_n_batch*t_n_multi,))
        if t_init is not None:
            t_init = tf.tile(tf.expand_dims(t_init, 1), (1, t_n_multi, 1))
            t_init = tf.reshape(t_init, (t_n_batch*t_n_multi, N_HIDDEN))
    t_embed = _embed_dict(t_input, t_vecs)
    with tf.variable_scope(name):
        _, t_encode = tf.nn.dynamic_rnn(
                cell, t_embed, t_len, dtype=tf.float32, initial_state=t_init)
    if multi:
        t_encode = tf.reshape(t_encode, (t_n_batch, t_n_multi, N_HIDDEN))
    return t_encode

def _conv_layer(t_input, n_filts, n_size, i_layer):
    n_channels = t_input.get_shape()[3].value
    weight = tf.get_variable("conv_w_%d" % i_layer, (n_size,
    n_size, n_channels, n_filts),
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
    bias = tf.get_variable("conv_b_%d" % i_layer, (n_filts),
            initializer=tf.constant_initializer(0))
    t_mul = tf.nn.conv2d(t_input, weight, (1, 1, 1, 1), "SAME")
    t_trans = t_mul + bias
    t_out = tf.nn.relu(t_trans)
    return t_out

def _convolve(name, t_input, t_dropout):
    multi = len(t_input.get_shape()) == 5
    t_keep = 1 - t_dropout
    assert multi or len(t_input.get_shape()) == 4
    if multi:
        t_shape = tf.shape(t_input)
        t_n_batch, t_n_multi = t_shape[0], t_shape[1]
        t_w, t_h, t_c = t_input.get_shape()[2:]
        t_input = tf.reshape(t_input, (t_n_batch * t_n_multi, t_w.value, t_h.value, t_c.value))

    with tf.variable_scope(name) as scope:
        t_conv1 = _conv_layer(t_input, N_CONV1_FILTS, N_CONV1_SIZE, 1)
        t_pool1 = tf.layers.max_pooling2d(t_conv1, 4, 4)
        t_conv2 = _conv_layer(t_pool1, N_CONV2_FILTS, N_CONV2_SIZE, 2)
        t_pool2 = tf.layers.max_pooling2d(t_conv2, 4, 4)
        t_conv3 = _conv_layer(t_pool2, N_CONV3_FILTS, N_CONV3_SIZE, 3)
        t_pool3 = tf.layers.average_pooling2d(t_conv3, 4, 4)
        final_w, final_h = t_pool3.get_shape()[1:3]
        final_feats = final_w.value * final_h.value * N_CONV3_FILTS

        t_flat = tf.reshape(t_pool3, (t_n_batch * t_n_multi, final_feats))
        t_repr = _linear(t_flat, N_HIDDEN)
        t_repr = tf.nn.dropout(t_repr, keep_prob=t_keep)

    if multi:
        t_repr = tf.reshape(t_repr, (t_n_batch, t_n_multi, N_HIDDEN))

    return t_repr

class Decoder(object):
    def __init__(self, name, t_init, t_target, t_last, t_last_hidden, t_vecs):
        self.t_init = t_init
        self.t_last = t_last
        self.t_last_hidden = t_last_hidden

        multi = len(t_init.get_shape()) == 3

        assert multi or len(t_init.get_shape()) == 2
        cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
        if multi:
            t_shape = tf.shape(t_target)
            t_n_batch, t_n_multi, t_n_toks = t_shape[0], t_shape[1], t_shape[2]
            t_init = tf.reshape(t_init, (t_n_batch*t_n_multi, N_HIDDEN))
            t_target = tf.reshape(t_target, (t_n_batch*t_n_multi, t_n_toks))

            t_shape = tf.shape(t_last)
            t_n_batch_d, t_n_multi_d = t_shape[0], t_shape[1]
            t_last = tf.reshape(t_last, (t_n_batch_d*t_n_multi_d,))
            t_last_hidden = tf.reshape(t_last_hidden, (t_n_batch_d*t_n_multi_d, N_HIDDEN))

        t_emb_target = _embed_dict(t_target, t_vecs)
        t_emb_last = _embed_dict(t_last, t_vecs)
        n_vocab = t_vecs.get_shape()[0].value

        with tf.variable_scope(name) as scope:
            v_proj = tf.get_variable("w",
                    shape=(N_HIDDEN, n_vocab),
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.43))
            b_proj = tf.get_variable("b",
                    shape=(n_vocab,),
                    initializer=tf.constant_initializer(0))

            t_dec_state, _ = tf.nn.dynamic_rnn(
                    cell, t_emb_target, initial_state=t_init, scope=scope)
            t_pred = tf.einsum("ijk,kl->ijl", t_dec_state, v_proj) + b_proj
            t_dec_err = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=t_target[:, 1:], logits=t_pred[:, :-1])
            t_dec_loss = tf.reduce_mean(tf.reduce_sum(t_dec_err, axis=1))
            t_scores = -tf.reduce_sum(t_dec_err, axis=1)

            scope.reuse_variables()

            t_next_hidden, _ = cell(t_emb_last, t_last_hidden)
            t_next_pred = tf.einsum("ij,jk->ik", t_next_hidden, v_proj) + b_proj

        if multi:
            t_next_hidden = tf.reshape(t_next_hidden, (t_n_batch_d, t_n_multi_d, N_HIDDEN))
            t_next_pred = tf.reshape(t_next_pred, (t_n_batch_d, t_n_multi_d, n_vocab))
            t_scores = tf.reshape(t_scores, (t_n_batch, t_n_multi))

        self.t_scores = t_scores
        self.t_loss = t_dec_loss
        self.t_next_hidden = t_next_hidden
        self.t_next_pred = t_next_pred
        self.multi = multi
        self.random = None

    def score(self, init, feed, session):
        scores, = session.run([self.t_scores], feed)
        return scores

    def reset_seed(self):
        self.random = np.random.RandomState(0) 

    def decode(self, init, stop, feed, session, temp=None):
        # reset random generator to ensure consistency across choice of eval
        # data
        last_hidden, = session.run([self.t_init], feed)
        last = init
        if self.multi:
            out = [[[w] for w in b] for b in init]
            out_scores = [[0 for w in b] for b in init]
        else:
            out = [[w] for w in init]
            out_scores = [0 for w in init]
        for t in range(20):
            next_hidden, next_pred = session.run(
                    [self.t_next_hidden, self.t_next_pred],
                    {self.t_last: last, self.t_last_hidden: last_hidden})
            if temp is None:
                preds = np.argmax(next_pred, axis=-1)
                lse = scipy.misc.logsumexp(next_pred, axis=-1)
                next_out_logits = next_pred - lse[..., np.newaxis]
                #next_out = zip(next_out, list(np.max(next_out_logits, axis=-1)))
                if self.multi:
                    next_out = [
                            [(preds[i, j], next_out_logits[i, j]) for j in range(preds.shape[1])]
                            for i in range(preds.shape[0])]
                else:
                    next_out = [(preds[i], next_out_logits[i]) for i in range(preds.shape[0])]
            else:
                def sample(logits):
                    probs = np.exp(logits / temp)
                    probs /= np.sum(probs)
                    choice = self.random.choice(len(probs), p=probs)
                    return choice, logits[choice]

                if self.multi:
                    next_out = [[sample(logits) for logits in batch] for batch in next_pred]
                else:
                    next_out = [sample(logits) for logits in next_pred]

            if self.multi:
                for batch, batch_so_far in zip(next_out, out):
                    for (w, _), so_far in zip(batch, batch_so_far):
                        if so_far[-1] != stop:
                            so_far.append(w)
                for i in range(len(next_out)):
                    for j in range(len(next_out[i])):
                        out_scores[i][j] += next_out[i][j][1]
                next_out_choices = [[w for (w, _) in b]  for b in next_out]
            else:
                for (w, _), so_far in zip(next_out, out):
                    if so_far[-1] != stop:
                        so_far.append(w)
                for i in range(len(next_out)):
                    out_scores[i] += next_out[i][1]
                next_out_choices = [w for (w, _) in next_out]

            last_hidden = next_hidden
            last = next_out_choices
        return out, out_scores

class SimModel(object):
    def __init__(self, task):
        self.task = task

        self.t_ex = tf.placeholder(tf.float32, (None, None, task.n_features))
        self.t_input = tf.placeholder(tf.float32, (None, task.n_features))
        self.t_output = tf.placeholder(tf.float32, (None,))
        t_enc_ex = tf.reduce_mean(self.t_ex, axis=1)

        t_ex_norm = tf.nn.l2_normalize(t_enc_ex, 1)
        t_input_norm = tf.nn.l2_normalize(self.t_input, 1)
        t_sim = tf.reduce_sum(t_ex_norm * t_input_norm, axis=1)
        self.t_score = t_sim + tf.get_variable("bias", shape=(),
                initializer=tf.constant_initializer(0))
        t_err = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.t_output, logits=self.t_score)
        self.t_loss = tf.reduce_mean(tf.reduce_sum(t_err))
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.o_train = optimizer.minimize(self.t_loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def feed(self, batch):
        inp = np.zeros((len(batch), self.task.n_features))
        example = np.zeros((len(batch), N_CLS_EX, self.task.n_features))
        out = np.zeros((len(batch),), dtype=np.int32)
        for i, datum in enumerate(batch):
            example[i, ...] = datum.ex_inputs
            inp[i, :] = datum.input
            out[i] = datum.label
        feed_dict = {
            self.t_ex: example,
            self.t_input: inp,
            self.t_output: out
        }
        return feed_dict

    def train(self, batch):
        feed = self.feed(batch)
        loss, _ = self.session.run([self.t_loss, self.o_train], feed)
        return loss

    def predict(self, batch):
        feed = self.feed(batch)
        scores = self.session.run(self.t_score, feed)
        preds = scores.ravel() > 0
        labels = [d.label for d in batch]
        accs = (preds == labels)
        return np.mean(accs)

    def save(self):
        pass

    def restore(self, path):
        assert False

class ClsModel(object):
    def __init__(self, task):
        self.task = task

        self.t_hint = tf.placeholder(tf.int32, (None, None), "hint")
        self.t_task = tf.placeholder(tf.int32, (None,), "task")
        self.t_hint_len = tf.placeholder(tf.int32, (None,), "hint_len")

        if USE_IMAGES:
            self.t_ex = tf.placeholder(
                    tf.float32, (None, None, task.width, task.height, task.channels))

            self.t_input = tf.placeholder(
                    tf.float32, (None, None, task.width, task.height, task.channels))
        else:
            self.t_ex = tf.placeholder(
                    tf.float32, (None, None, task.n_features))
            self.t_input = tf.placeholder(
                    tf.float32, (None, None, task.n_features))
        self.t_output = tf.placeholder(tf.float32, (None, None))

        self.t_last_hyp = tf.placeholder(tf.int32, (None,), "last_hyp")
        self.t_last_hyp_hidden = tf.placeholder(tf.float32, (None, N_HIDDEN), "last_hyp_hidden")
        self.t_dropout = tf.constant(0.2)

        if FLAGS.use_task_hyp:
            t_hint_vecs = tf.get_variable(
                    "hint_vec", shape=(len(task.task_index), N_HIDDEN), # N_EMBED_WORD
                    initializer=tf.uniform_unit_scaling_initializer())
        else:
            t_hint_vecs = tf.get_variable(
                    "hint_vec", shape=(len(task.hint_vocab), N_HIDDEN), # N_EMBED_WORD
                    initializer=tf.uniform_unit_scaling_initializer())

        if USE_IMAGES:
            t_enc_ex_all = _convolve("encode_ex", self.t_ex, self.t_dropout)
        else:
            with tf.variable_scope("encode_ex"):
                t_enc_ex_all = self.t_ex
                t_enc_ex_all = _mlp(t_enc_ex_all, (N_HIDDEN, N_HIDDEN), (tf.nn.relu, None))

        with tf.variable_scope("reduce_ex"):
            t_enc_ex = tf.reduce_mean(t_enc_ex_all, axis=1)

        t_enc_hint = _encode(
                "encode_hint", self.t_hint, self.t_hint_len, t_hint_vecs)

        if FLAGS.infer_hyp:
            t_concept = t_enc_hint
        else:
            t_concept = t_enc_ex

        if USE_IMAGES:
            t_enc_input = _convolve("encode_input", self.t_input, self.t_dropout)
        else:
            with tf.variable_scope("encode_input"):
                t_enc_input = self.t_input
                t_enc_input = _mlp(t_enc_input, (N_HIDDEN, N_HIDDEN), (tf.nn.relu, None))

        self.hyp_decoder = Decoder(
                "decode_hyp", t_enc_ex, self.t_hint, self.t_last_hyp,
                self.t_last_hyp_hidden, t_hint_vecs)

        t_bc_concept = tf.expand_dims(t_concept, 1)
        #t_bc_concept = tf.tile(t_bc_concept, (1, tf.shape(t_enc_input)[1], 1))

        # TODO bilinear?
        self.t_score = tf.reduce_sum(t_bc_concept * t_enc_input, axis=2)
        #t_comb = tf.concat((t_bc_concept, t_enc_input), axis=2)
        #self.t_score = _mlp(t_comb, (N_HIDDEN, 1), (tf.nn.relu, None))
        #self.t_score = tf.squeeze(self.t_score, axis=2)

        t_err = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.t_output, logits=self.t_score)
        t_pred_loss = tf.reduce_mean(tf.reduce_sum(t_err, axis=1))

        if FLAGS.predict_hyp:
            self.t_loss = t_pred_loss + self.hyp_decoder.t_loss
        else:
            self.t_loss = t_pred_loss

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.o_train = optimizer.minimize(self.t_loss)
        self.o_train_task = optimizer.minimize(self.t_loss, var_list=[t_hint_vecs])

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if FLAGS.restore is not None:
            self.restore(FLAGS.restore)

    def feed(self, batch, input_examples=False, dropout=True):
        if input_examples:
            n_input = N_CLS_EX
        else:
            n_input = 1

        if USE_IMAGES:
            width, height, channels = self.task.width, self.task.height, self.task.channels
            example = np.zeros((len(batch), N_CLS_EX, width, height, channels))
            inp = np.zeros((len(batch), n_input, width, height, channels))
        else:
            n_features = self.task.n_features
            example = np.zeros((len(batch), N_CLS_EX, n_features))
            inp = np.zeros((len(batch), n_input, n_features))

        max_hint_len = max(len(d.hint) for d in batch)
        hint = np.zeros((len(batch), max_hint_len), dtype=np.int32)
        hint_len = np.zeros((len(batch),), dtype=np.int32)
        out = np.zeros((len(batch), n_input))

        for i, datum in enumerate(batch):
            if FLAGS.use_task_hyp:
                hint[i, 0] = datum.task_id
                hint_len[i] = 1
            else:
                hint[i, :len(datum.hint)] = datum.hint
                hint_len[i] = len(datum.hint)
            example[i, ...] = datum.ex_inputs
            if input_examples:
                inp[i, ...] = datum.ex_inputs
                out[i, ...] = 1
            else:
                inp[i, 0, ...] = datum.input
                out[i, 0] = datum.label

        feed_dict = {
            self.t_hint: hint,
            self.t_hint_len: hint_len,
            self.t_ex: example,
            self.t_input: inp,
            self.t_output: out
        }
        if not dropout:
            feed_dict[self.t_dropout] = 0
        return feed_dict

    def train(self, batch, task_only=False):
        o_train = self.o_train
        if task_only:
            o_train = self.o_train_task
        feed = self.feed(batch)
        loss, _ = self.session.run([self.t_loss, o_train], feed)
        return loss

    def hypothesize(self, batch):
        hyp_init = [self.task.hint_vocab[self.task.START] for _ in batch]
        hyp_stop = self.task.hint_vocab[self.task.STOP]
        feed = self.feed(batch, dropout=False)

        best_score = [-np.inf] * len(batch)
        best_hyps = [None] * len(batch)
        worst_score = [np.inf] * len(batch)

        found_gold = [False] * len(batch)

        self.hyp_decoder.reset_seed()
        for i in range(FLAGS.n_sample_hyps):
            hyps, gen_scores = self.hyp_decoder.decode(
                    hyp_init, hyp_stop, feed, self.session,
                    temp=None if i == 0 else 1)
            hyp_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
            hyp_feed = self.feed(hyp_batch, input_examples=True, dropout=False)

            scores, = self.session.run([self.t_score], hyp_feed)
            preds = scores > 0

            for j in range(len(batch)):
                if FLAGS.infer_by_likelihood:
                    score = scores[j, :].sum()# + gen_scores[j]
                else:
                    score = preds[j, :].sum()
                ex_here = hyp_feed[self.t_output][j, ...]
                if score > best_score[j]:
                    best_score[j] = score
                    best_hyps[j] = hyps[j]
                if score < worst_score[j]:
                    worst_score[j] = score
                found_gold[j] = found_gold[j] or hyps[j] == batch[j].hint

        hyps = best_hyps

        #print "pred, gold"
        #for i in range(3):
        #    print " ".join(self.task.hint_vocab.get(c) for c in hyps[i]),
        #    print " ".join(self.task.hint_vocab.get(c) for c in feed[self.t_hint][i])

        #agree = 0
        #for i in range(len(batch)):
        #    h = hyps[i]
        #    g = [c for c in feed[self.t_hint][i].tolist() if c]
        #    agree += (1 if h == g else 0)

        #print "[gold_chosen]", 1. * agree / len(batch)
        #print "[gold_found] ", 1. * np.mean(found_gold)
        #print

        return hyps

    def predict(self, batch, debug=False):
        if FLAGS.infer_hyp and not FLAGS.use_true_hyp:
            hyps = self.hypothesize(batch)
            pred_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
        else:
            pred_batch = batch

        pred_feed = self.feed(pred_batch, dropout=False)
        scores, = self.session.run([self.t_score], pred_feed)
        preds = scores.ravel() > 0
        labels = [d.label for d in batch]
        accs = (preds == labels)
        if debug:
            return preds, labels, hyps
        else:
            return np.mean(accs)

    def save(self):
        self.saver.save(self.session, "model.chk")

    def restore(self, path):
        self.saver.restore(self.session, path)

class IdentityModel(object):
    def __init__(self, task):
        self.task = task

    def train(self):
        pass

    def predict(self, batch):
        preds = [d.input for d in batch]
        outputs = [d.output for d in batch]
        accs = [p == o for p, o in zip(preds, outputs)]
        return np.mean(accs)

class TransducerModel(object):
    def __init__(self, task):
        self.task = task

        self.t_hint = tf.placeholder(tf.int32, (None, None), "hint")
        self.t_hint_len = tf.placeholder(tf.int32, (None,), "hint_len")

        self.t_ex = tf.placeholder(tf.int32, (None, None, None), "ex")
        self.t_ex_len = tf.placeholder(tf.int32, (None, None), "ex_len")

        self.t_input = tf.placeholder(tf.int32, (None, None, None), "input")
        self.t_input_len = tf.placeholder(tf.int32, (None, None), "input_len")
        self.t_output = tf.placeholder(tf.int32, (None, None, None), "output")

        self.t_last_out = tf.placeholder(tf.int32, (None, None), "last_out")
        self.t_last_out_hidden = tf.placeholder(tf.float32, (None, None, N_HIDDEN), "last_out_hidden")

        self.t_last_hyp = tf.placeholder(tf.int32, (None,), "last_hyp")
        self.t_last_hyp_hidden = tf.placeholder(tf.float32, (None, N_HIDDEN), "last_hyp_hidden")

        t_str_vecs = tf.get_variable(
                "str_vec", shape=(len(task.str_vocab), N_EMBED),
                initializer=tf.uniform_unit_scaling_initializer())
        if FLAGS.use_task_hyp:
            t_hint_vecs = tf.get_variable(
                    "hint_vec", shape=(task.n_tasks, N_EMBED_WORD),
                    initializer=tf.uniform_unit_scaling_initializer())
        else:
            t_hint_vecs = tf.get_variable(
                    "hint_vec", shape=(len(task.hint_vocab), N_EMBED_WORD),
                    initializer=tf.uniform_unit_scaling_initializer())

        t_enc_ex_all = _encode(
                "encode_ex", self.t_ex, self.t_ex_len, t_str_vecs)
        t_enc_hint = _encode(
                "encode_hint", self.t_hint, self.t_hint_len, t_hint_vecs)
        t_enc_ex = tf.reduce_mean(t_enc_ex_all, axis=1)

        if FLAGS.infer_hyp:
            t_concept = t_enc_hint
        else:
            t_concept = t_enc_ex

        t_enc_input = _encode(
                "encode_input", self.t_input, self.t_input_len, t_str_vecs,
                t_init=t_concept)

        self.hyp_decoder = Decoder(
                "decode_hyp", t_enc_ex, self.t_hint, self.t_last_hyp,
                self.t_last_hyp_hidden, t_hint_vecs)
        self.out_decoder = Decoder(
                "decode_out", t_enc_input, self.t_output, self.t_last_out,
                self.t_last_out_hidden, t_str_vecs)

        if FLAGS.predict_hyp:
            self.t_loss = self.out_decoder.t_loss + self.hyp_decoder.t_loss
        else:
            self.t_loss = self.out_decoder.t_loss

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.o_train = optimizer.minimize(self.t_loss)
        self.o_train_task = optimizer.minimize(self.t_loss, var_list=[t_hint_vecs])

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if FLAGS.restore is not None:
            self.restore(FLAGS.restore)

    def feed(self, batch, input_examples=False):
        if FLAGS.use_task_hyp:
            max_hint_len = 1
        else:
            max_hint_len = max(len(d.hint) for d in batch)
        max_einp_len = max(len(e) for d in batch for e in d.ex_inputs)
        max_eout_len = max(len(e) for d in batch for e in d.ex_outputs)
        max_inp_len = max(len(d.input) for d in batch)
        max_inp_len = max(max_inp_len, max_einp_len)
        max_out_len = max(len(d.output) for d in batch)
        max_out_len = max(max_out_len, max_eout_len)
        max_ex_len = max_einp_len + max_eout_len + 1

        hint = np.zeros((len(batch), max_hint_len), dtype=np.int32)
        hint_len = np.zeros((len(batch),), dtype=np.int32)
        example = np.zeros((len(batch), N_PBD_EX, max_ex_len), dtype=np.int32)
        example_len = np.zeros((len(batch), N_PBD_EX), dtype=np.int32)
        if input_examples:
            n_input = N_PBD_EX
        else:
            n_input = 1
        inp = np.zeros((len(batch), n_input, max_inp_len), dtype=np.int32)
        inp_len = np.zeros((len(batch), n_input), dtype=np.int32)
        out = np.zeros((len(batch), n_input, max_out_len), dtype=np.int32)
        for i, datum in enumerate(batch):
            examples = zip(datum.ex_inputs, datum.ex_outputs)
            target_inp = datum.input
            target_out = datum.output
            if input_examples:
                for j, (e_inp, e_out) in enumerate(examples):
                    inp[i, j, :len(e_inp)] = e_inp
                    inp_len[i, j] = len(e_inp)
                    out[i, j, :len(e_out)] = e_out
            else:
                inp[i, 0, :len(target_inp)] = target_inp
                inp_len[i, 0] = len(target_inp)
                out[i, 0, :len(target_out)] = target_out

            if FLAGS.use_task_hyp:
                hint[i, 0] = datum.task_id
                hint_len[i] = 1
            else:
                hint[i, :len(datum.hint)] = datum.hint
                hint_len[i] = len(datum.hint)
            for j, (e_inp, e_out) in enumerate(examples):
                exp = e_inp + [self.task.str_vocab[self.task.SEP]] + e_out
                example[i, j, :len(exp)] = exp
                example_len[i, j] = len(exp)

        return {
            self.t_hint: hint,
            self.t_hint_len: hint_len,
            self.t_ex: example,
            self.t_ex_len: example_len,
            self.t_input: inp,
            self.t_input_len: inp_len,
            self.t_output: out
        }

    def train(self, batch, task_only=False):
        o_train = self.o_train
        if task_only:
            o_train = self.o_train_task
        feed = self.feed(batch)
        loss, _ = self.session.run([self.t_loss, o_train], feed)
        return loss

    def hypothesize(self, batch):
        hyp_init = [self.task.hint_vocab[self.task.START] for _ in batch]
        hyp_stop = self.task.hint_vocab[self.task.STOP]
        feed = self.feed(batch)

        best_score = [-np.inf] * len(batch)
        best_hyps = [None] * len(batch)
        worst_score = [np.inf] * len(batch)
        found_gold = [False] * len(batch)
        found_exact = [False] * len(batch)
        chose_gold = [False] * len(batch)

        self.hyp_decoder.reset_seed()
        for i in range(FLAGS.n_sample_hyps):
            hyps, _ = self.hyp_decoder.decode(
                    hyp_init, hyp_stop, feed, self.session,
                    temp=None if i == 0 else 1)
            hyp_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
            hyp_feed = self.feed(hyp_batch, input_examples=True)

            init = self.task.str_vocab[self.task.START] * np.ones(hyp_feed[self.t_ex_len].shape, dtype=np.int32)
            stop = self.task.str_vocab[self.task.STOP]
            if FLAGS.use_true_eval:
                scores, preds = self.task_eval(hyp_feed)
            else:
                scores = self.out_decoder.score(init, hyp_feed, self.session)
                preds, _ = self.out_decoder.decode(init, stop, hyp_feed, self.session)

            for j in range(len(batch)):
                l_score = scores[j, :].sum()
                ex_here = hyp_feed[self.t_output][j, ...]
                m_score = 0
                for ex, pred in zip(ex_here, preds[j]):
                    if np.all(ex[:len(pred)] == pred):
                        m_score += 1

                if FLAGS.infer_by_likelihood:
                    score = l_score
                else:
                    score = m_score

                found_gold[j] = found_gold[j] or hyps[j] == batch[j].hint
                found_exact[j] = found_exact[j] or m_score == ex_here.shape[0]
                if score > best_score[j]:
                    best_score[j] = score
                    best_hyps[j] = hyps[j]
                    chose_gold[j] = hyps[j] == batch[j].hint
                if score < worst_score[j]:
                    worst_score[j] = score

        hyps = best_hyps
        print >>sys.stderr, best_score

        print >>sys.stderr, "\n".join(" ".join(self.task.hint_vocab.get(c) for c in hyp) for hyp in hyps[:3])
        print >>sys.stderr, "\n".join(" ".join(self.task.hint_vocab.get(c) for c in hyp if c) for hyp in feed[self.t_hint][:3])
        print >>sys.stderr

        print "[found_gold]  %0.2f" % np.mean(found_gold)
        print "[chose_gold]  %0.2f" % np.mean(chose_gold)
        print "[found_exact] %0.2f" % np.mean(found_exact)

        return hyps

    def predict(self, batch, debug=False):
        if FLAGS.infer_hyp and not FLAGS.use_true_hyp:
            hyps = self.hypothesize(batch)
            pred_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
        else:
            pred_batch = batch

        init = [[self.task.str_vocab[self.task.START]] for _ in batch]
        stop = self.task.str_vocab[self.task.STOP]

        pred_feed = self.feed(pred_batch)
        if FLAGS.use_true_eval:
            _, preds = self.task_eval(pred_feed)
        else:
            preds, _ = self.out_decoder.decode(init, stop, pred_feed, self.session)
        accs = []
        for i, (pred, gold) in enumerate(zip(preds, pred_feed[self.t_output])):
            pred = pred[0]
            gold = gold[0].tolist()
            gold = gold[:gold.index(self.task.str_vocab[self.task.STOP])+1]
            accs.append(pred == gold)

        if debug:
            return preds, pred_feed[self.t_output], hyps
        else:
            return np.mean(accs)

    def save(self):
        self.saver.save(self.session, "model.chk")

    def restore(self, path):
        self.saver.restore(self.session, path)

    def task_eval(self, hyp_feed):
        scores = []
        preds = []

        hints = hyp_feed[self.t_hint]
        hint_lens = hyp_feed[self.t_hint_len]
        inputs = hyp_feed[self.t_input]
        input_lens = hyp_feed[self.t_input_len]
        outputs = hyp_feed[self.t_output]

        for i in range(hints.shape[0]):
            hint = list(hints[i, :hint_lens[i]])
            inps = inputs[i, ...]
            inp_lens = input_lens[i, :]
            inps = [list(inp[:ilen]) for inp, ilen in zip(inps, inp_lens)]
            outs = outputs[i, ...]
            outs = [list(w for w in out if w != 0) for out in outs]

            score, prds = self.task.execute(hint, inps, outs)
            scores.append(score)
            preds.append(prds)

        return np.asarray(scores), preds
