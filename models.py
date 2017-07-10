from net import _mlp, _embed_dict, _linear
import util

import gflags
import numpy as np
import sys
import tensorflow as tf
import os

FLAGS = gflags.FLAGS
gflags.DEFINE_boolean("predict_hyp", False, "train to predict hypotheses")
gflags.DEFINE_boolean("infer_hyp", False, "use hypotheses at test time")
gflags.DEFINE_boolean("infer_maml", False, "use MAML")
gflags.DEFINE_boolean("infer_by_likelihood", False, "use likelihood (rather than accuracy) to rank hypotheses")
gflags.DEFINE_boolean("use_true_hyp", False, "predict using ground-truth description")
gflags.DEFINE_integer("n_sample_hyps", 5, "number of hypotheses to sample")
gflags.DEFINE_float("learning_rate", 0.001, "learning rate")
gflags.DEFINE_string("restore", None, "model to restore")

N_EMBED = 32
N_EMBED_WORD = 128
N_HIDDEN = 512
N_PBD_EX = 5
N_CLS_EX = 3

N_CONV1_SIZE = 5
N_CONV1_FILTS = 16
N_CONV2_SIZE = 3
N_CONV2_FILTS = 32
N_CONV3_SIZE = 3
N_CONV3_FILTS = 64

random = util.next_random()
tf.set_random_seed(0)

def fake_log_loss(labels, logits):
    logprobs = tf.nn.log_softmax(logits, dim=2)
    r, c = tf.meshgrid(
            tf.range(tf.shape(logprobs)[0]),
            tf.range(logprobs.get_shape()[1].value),
            indexing="ij")
    coords = tf.stack([r, c, labels], axis=2)
    return -tf.gather_nd(logprobs, coords)

def _encode(name, t_input, t_len, t_vecs, t_init=None, reuse=False):
    multi = len(t_input.get_shape()) == 3
    assert multi or len(t_input.get_shape()) == 2
    cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
    if multi:
        t_shape = tf.shape(t_input)
        t_n_batch, t_n_multi, t_n_toks = t_shape[0], t_shape[1], t_shape[2]
        t_n_toks = t_input.get_shape()[2].value
        t_input = tf.reshape(t_input, (t_n_batch*t_n_multi, t_n_toks))
        t_len = tf.reshape(t_len, (t_n_batch*t_n_multi,))
        if t_init is not None:
            t_init = tf.tile(tf.expand_dims(t_init, 1), (1, t_n_multi, 1))
            t_init = tf.reshape(t_init, (t_n_batch*t_n_multi, N_HIDDEN))
    t_embed = _embed_dict(t_input, t_vecs)
    with tf.variable_scope(name, reuse=reuse):
        _, t_encode = tf.nn.static_rnn(
                cell, tf.unstack(t_embed, axis=1), dtype=tf.float32, initial_state=t_init)
        #_, t_encode = tf.nn.dynamic_rnn(
        #        cell, t_embed, t_len, dtype=tf.float32, initial_state=t_init)
    if multi:
        t_encode = tf.reshape(t_encode, (t_n_batch, t_n_multi, N_HIDDEN))
    return t_encode

def _conv_params():
    nn_filts = [N_CONV1_FILTS, N_CONV2_FILTS, N_CONV3_FILTS]
    nn_chan = [3, N_CONV1_FILTS, N_CONV2_FILTS]
    nn_size = [N_CONV1_SIZE, N_CONV2_SIZE, N_CONV3_SIZE]
    params = []
    for i_layer, n_filts, n_chan, n_size in zip(range(3), nn_filts, nn_chan, nn_size):
        w = tf.get_variable(
                "conv_w_%d" % i_layer, (n_size, n_size, n_chan, n_filts),
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable(
                "conv_b_%d" % i_layer, (n_filts),
                initializer=tf.constant_initializer(0))
        params.append(w)
        params.append(b)
    # TODO
    w = tf.get_variable(
            "linear_w", (4096, 1),
            initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(
            "linear_b", (1,),
            initializer=tf.contrib.layers.xavier_initializer())
    params += [w, b]
    return params

#def _conv_layer(t_input, n_filts, n_size, i_layer, weight, bias):
#    n_channels = t_input.get_shape()[3].value
#    weight = use_weight or tf.get_variable("conv_w_%d" % i_layer, (n_size, n_size, n_channels, n_filts),
#            initializer=tf.contrib.layers.xavier_initializer_conv2d())
#    bias = use_bias or tf.get_variable("conv_b_%d" % i_layer, (n_filts),
#            initializer=tf.constant_initializer(0))
#    t_mul = tf.nn.conv2d(t_input, weight, (1, 1, 1, 1), "SAME")
#    t_trans = t_mul + bias
#    t_out = tf.nn.relu(t_trans)
#    return t_trans, weight, bias

def _conv_layer(t_input, weight, bias):
    #print(t_input.get_shape())
    #print(weight.get_shape())
    #exit()
    t_n_batch = tf.shape(t_input)[1]
    tf.assert_equal(t_n_batch, 1)
    t_mul = tf.nn.conv3d(t_input, weight, (1, 1, 1, 1, 1), "SAME")
    exp_bias = tf.expand_dims(bias, 0)
    exp_bias = tf.expand_dims(exp_bias, 2)
    exp_bias = tf.expand_dims(exp_bias, 3)
    t_trans = t_mul + exp_bias
    t_out = tf.nn.relu(t_trans)
    return t_out

def _convolve(name, t_input, params):
    w1, b1, w2, b2, w3, b3, wl, bl = params

    multi = len(t_input.get_shape()) == 5
    assert multi or len(t_input.get_shape()) == 4
    t_shape = tf.shape(t_input)
    if multi:
        t_n_batch, t_n_multi = t_shape[0], t_shape[1]
        #t_w, t_h, t_c = t_input.get_shape()[2:]
        #t_input = tf.reshape(t_input, (t_n_batch * t_n_multi, t_w.value, t_h.value, t_c.value))
        t_input = tf.transpose(t_input, (1, 0, 2, 3, 4))
    else:
        t_input = tf.expand_dims(t_input, 0)
        t_n_batch = t_shape[0]
        t_n_multi = 1

    t_conv1 = _conv_layer(t_input, w1, b1)
    t_pool1 = tf.layers.max_pooling3d(t_conv1, (1, 2, 2), (1, 2, 2))
    t_conv2 = _conv_layer(t_pool1, w2, b2)
    t_pool2 = tf.layers.max_pooling3d(t_conv2, (1, 2, 2), (1, 2, 2))
    t_conv3 = _conv_layer(t_pool2, w3, b3)
    t_pool3 = tf.layers.average_pooling3d(t_conv3, (1, 2, 2), (1, 2, 2))

    # TODO
    t_flat = tf.reshape(t_pool3, (t_n_multi, t_n_batch, 4096))
    #print(t_flat.get_shape())
    #print(wl.get_shape())

    t_repr = tf.einsum("ijk,jkl->ijl", t_flat, wl) + tf.expand_dims(bl, 0)
    t_repr = tf.squeeze(t_repr, (-1,))
    #t_repr = tf.reduce_mean(t_conv1, axis=(2, 3, 4))

    #print(t_flat.get_shape())
    #print(wl.get_shape(), bl.get_shape())
    #t_repr = wl * t_flat + bl

    if multi:
        t_repr = tf.transpose(t_repr, (1, 0))
    else:
        t_repr = tf.squeeze(t_repr, (0,))

    return t_repr

class Decoder(object):
    def __init__(self, name, t_init, t_target, t_last, t_last_hidden, t_vecs, reuse=False):
        self.t_init = t_init
        self.t_last = t_last
        self.t_last_hidden = t_last_hidden

        multi = len(t_init.get_shape()) == 3

        assert multi or len(t_init.get_shape()) == 2
        cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
        if multi:
            t_shape = tf.shape(t_target)
            t_n_batch, t_n_multi, t_n_toks = t_shape[0], t_shape[1], t_shape[2]
            t_n_toks = t_target.get_shape()[2].value
            t_init = tf.reshape(t_init, (t_n_batch*t_n_multi, N_HIDDEN))
            t_target = tf.reshape(t_target, (t_n_batch*t_n_multi, t_n_toks))

            t_shape = tf.shape(t_last)
            t_n_batch_d, t_n_multi_d = t_shape[0], t_shape[1]
            t_last = tf.reshape(t_last, (t_n_batch_d*t_n_multi_d,))
            t_last_hidden = tf.reshape(t_last_hidden, (t_n_batch_d*t_n_multi_d, N_HIDDEN))

        t_emb_target = _embed_dict(t_target, t_vecs)
        t_emb_last = _embed_dict(t_last, t_vecs)
        n_vocab = t_vecs.get_shape()[0].value

        with tf.variable_scope(name, reuse=reuse) as scope:
            v_proj = tf.get_variable("w",
                    shape=(N_HIDDEN, n_vocab),
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.43))
            b_proj = tf.get_variable("b",
                    shape=(n_vocab,),
                    initializer=tf.constant_initializer(0))

            t_dec_state, _ = tf.nn.static_rnn(
                    cell, tf.unstack(t_emb_target, axis=1), initial_state=t_init, scope=scope)
            #t_dec_state, _ = tf.nn.dynamic_rnn(
            #        cell, t_emb_target, initial_state=t_init, scope=scope)
            #t_dec_state = tf.stack(t_dec_state, axis=1)
            t_pred = tf.einsum("ijk,kl->ijl", t_dec_state, v_proj) + b_proj

            t_dec_err = fake_log_loss(t_target[:, 1:], t_pred[:, :-1])
            #t_dec_err = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #        labels=t_target[:, 1:], logits=t_pred[:, :-1])
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

    def score(self, init, feed, session):
        scores, = session.run([self.t_scores], feed)
        return scores

    def decode(self, init, stop, feed, session, temp=None):
        last_hidden, = session.run([self.t_init], feed)
        last = init
        if self.multi:
            out = [[[w] for w in b] for b in init]
        else:
            out = [[w] for w in init]
        for t in range(20):
            next_hidden, next_pred = session.run(
                    [self.t_next_hidden, self.t_next_pred],
                    {self.t_last: last, self.t_last_hidden: last_hidden})
            if temp is None:
                next_out = np.argmax(next_pred, axis=-1)
            else:
                def sample(logits):
                    probs = np.exp(logits / temp)
                    probs /= np.sum(probs)
                    return random.choice(len(probs), p=probs)
                if self.multi:
                    next_out = [[sample(logits) for logits in batch] for batch in next_pred]
                else:
                    next_out = [sample(logits) for logits in next_pred]

            if self.multi:
                for batch, batch_so_far in zip(next_out, out):
                    for w, so_far in zip(batch, batch_so_far):
                        if so_far[-1] != stop:
                            so_far.append(w)
            else:
                for w, so_far in zip(next_out, out):
                    if so_far[-1] != stop:
                        so_far.append(w)
            last_hidden = next_hidden
            last = next_out
        return out

class ConvModel(object):
    def __init__(self, task):
        self.task = task

        #self.t_hint = tf.placeholder(tf.int32, (None, None), "hint")
        #self.t_hint_len = tf.placeholder(tf.int32, (None,), "hint_len")

        self.t_ex = tf.placeholder(
                tf.float32, (None, None, task.width, task.height, task.channels))

        self.t_input = tf.placeholder(
                tf.float32, (None, 1, task.width, task.height, task.channels))
        self.t_output = tf.placeholder(tf.float32, (None, 1))

        t_n_batch = tf.shape(self.t_input)[0]

        #self.t_last_hyp = tf.placeholder(tf.int32, (None,), "last_hyp")
        #self.t_last_hyp_hidden = tf.placeholder(tf.float32, (None, N_HIDDEN), "last_hyp_hidden")

        #t_hint_vecs = tf.get_variable(
        #        "hint_vec", shape=(len(task.hint_vocab), N_EMBED_WORD),
        #        initializer=tf.uniform_unit_scaling_initializer())

        params = _conv_params()
        params_tiled = []
        for param in params:
            tparam = tf.tile(tf.expand_dims(param, 0), (t_n_batch,) + (1,) * len(param.get_shape()))
            params_tiled.append(tparam)

        #print([p.get_shape() for p in params_tiled])

        t_enc_ex_all = _convolve("encode_ex", self.t_ex, params_tiled)
        #print(t_enc_ex_all.get_shape())

        t_ex_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones(tf.shape(t_enc_ex_all)), logits=t_enc_ex_all)
        #print(t_ex_loss_all.get_shape())
        t_ex_loss = tf.reduce_mean(t_ex_loss_all, axis=1)

        t_grads = tf.gradients(t_ex_loss, params_tiled)
        print(t_grads)
        params_stepped = []
        for param, grad in zip(params_tiled, t_grads):
            sparam = param - 0.1 * grad
            params_stepped.append(sparam)
        #params_stepped = params_tiled

        #t_enc_hint = _encode(
        #        "encode_hint", self.t_hint, self.t_hint_len, t_hint_vecs)
        #t_enc_ex = tf.reduce_mean(t_enc_ex_all, axis=1)

        #if FLAGS.infer_hyp:
        #    t_concept = t_enc_hint
        #else:
        #    t_concept = t_enc_ex

        t_enc_input = _convolve("encode_input", self.t_input, params_stepped)
        #t_enc_input, _ = _convolve("encode_input", self.t_input)
        #self.hyp_decoder = Decoder(
        #        "decode_hyp", t_enc_ex, self.t_hint, self.t_last_hyp,
        #        self.t_last_hyp_hidden, t_hint_vecs)

        #t_bc_concept = tf.expand_dims(t_concept, 1)
        # TODO bilinear?
        #self.t_score = tf.reduce_sum(t_bc_concept * t_enc_input, axis=2)
        self.t_score = t_enc_input

        t_err = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.t_output, logits=self.t_score)
        t_pred_loss = tf.reduce_mean(tf.reduce_sum(t_err, axis=1))

        #if FLAGS.predict_hyp:
        #    self.t_loss = t_pred_loss + self.hyp_decoder.t_loss
        #else:
        #    self.t_loss = t_pred_loss
        self.t_loss = t_pred_loss

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.o_train = optimizer.minimize(self.t_loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if FLAGS.restore is not None:
            self.restore(FLAGS.restore)

    def feed(self, batch, input_examples=False):
        width, height, channels = self.task.width, self.task.height, self.task.channels
        max_hint_len = max(len(d.hint) for d in batch)
        hint = np.zeros((len(batch), max_hint_len), dtype=np.int32)
        hint_len = np.zeros((len(batch),), dtype=np.int32)
        example = np.zeros((len(batch), N_CLS_EX, width, height, channels))
        if input_examples:
            n_input = N_CLS_EX
        else:
            n_input = 1
        inp = np.zeros((len(batch), n_input, width, height, channels))
        out = np.zeros((len(batch), n_input))

        for i, datum in enumerate(batch):
            hint[i, :len(datum.hint)] = datum.hint
            hint_len[i] = len(datum.hint)
            example[i, ...] = datum.ex_inputs
            if input_examples:
                inp[i, ...] = datum.ex_inputs
                out[i, ...] = 1
            else:
                inp[i, 0, ...] = datum.input
                out[i, 0] = datum.label

        return {
            #self.t_hint: hint,
            #self.t_hint_len: hint_len,
            self.t_ex: example,
            self.t_input: inp,
            self.t_output: out
        }

    def train(self, batch):
        feed = self.feed(batch)
        loss, _ = self.session.run([self.t_loss, self.o_train], feed)
        return loss

    def hypothesize(self, batch):
        hyp_init = [self.task.hint_vocab[self.task.START] for _ in batch]
        hyp_stop = self.task.hint_vocab[self.task.STOP]
        feed = self.feed(batch)

        best_score = [-1] * len(batch)
        best_hyps = [None] * len(batch)
        worst_score = [6] * len(batch)

        for i in range(FLAGS.n_sample_hyps):
            hyps = self.hyp_decoder.decode(
                    hyp_init, hyp_stop, feed, self.session,
                    temp=None if i == 0 else 1)
            hyp_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
            hyp_feed = self.feed(hyp_batch, input_examples=True)
            scores, = self.session.run([self.t_score], hyp_feed)
            #print(scores)
            preds = scores > 0

            for j in range(len(batch)):
                ex_here = hyp_feed[self.t_output][j, ...]
                score = np.sum(preds[j, :])
                if score > best_score[j]:
                    best_score[j] = score
                    best_hyps[j] = hyps[j]
                if score < worst_score[j]:
                    worst_score[j] = score

        hyps = best_hyps
        #print >>sys.stderr, best_score
        ##print >>sys.stderr, worst_score

        print("\n".join(" ".join(self.task.hint_vocab.get(c) for c in hyp) for hyp in hyps[:3]))
        print("\n".join(" ".join(self.task.hint_vocab.get(c) for c in hyp if c) for hyp in feed[self.t_hint][:3]))
        print()

        return hyps

    def predict(self, batch):
        #if FLAGS.infer_hyp and not FLAGS.use_true_hyp:
        #    hyps = self.hypothesize(batch)
        #    pred_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
        #else:
        #    pred_batch = batch

        pred_batch = batch
        pred_feed = self.feed(pred_batch)

        scores, = self.session.run([self.t_score], pred_feed)
        preds = scores.ravel() > 0
        labels = [d.label for d in batch]
        accs = (preds == labels)
        return np.mean(accs)

    def save(self):
        self.saver.save(self.session, "model.chk")

    def restore(self, path):
        self.saver.restore(self.session, path)

class TransducerModel(object):
    def __init__(self, task):
        self.task = task
        
        max_ilen = task.max_input_len
        max_olen = task.max_output_len

        self.t_hint = tf.placeholder(tf.int32, (None, None), "hint")
        self.t_hint_len = tf.placeholder(tf.int32, (None,), "hint_len")

        self.t_ex = tf.placeholder(tf.int32, (None, None, None), "ex")
        self.t_ex_len = tf.placeholder(tf.int32, (None, None), "ex_len")

        self.t_input = tf.placeholder(tf.int32, (None, None, max_ilen), "input")
        self.t_input_len = tf.placeholder(tf.int32, (None, None), "input_len")
        self.t_output = tf.placeholder(tf.int32, (None, None, max_olen), "output")

        self.t_last_out = tf.placeholder(tf.int32, (None, None), "last_out")
        self.t_last_out_hidden = tf.placeholder(tf.float32, (None, None, N_HIDDEN), "last_out_hidden")

        self.t_last_hyp = tf.placeholder(tf.int32, (None,), "last_hyp")
        self.t_last_hyp_hidden = tf.placeholder(tf.float32, (None, N_HIDDEN), "last_hyp_hidden")

        self.t_maml_input = tf.placeholder(tf.int32, (None, None, max_ilen), "maml_input")
        self.t_maml_input_len = tf.placeholder(tf.int32, (None, None), "maml_input_len")
        self.t_maml_output = tf.placeholder(tf.int32, (None, None, max_olen), "maml_output")
        self.t_maml_last_out = tf.placeholder(tf.int32, (None, None), "maml_last_out")
        self.t_maml_last_out_hidden = tf.placeholder(tf.float32, (None, None, N_HIDDEN), "maml_last_out_hidden")

        t_str_vecs = tf.get_variable(
                "str_vec", shape=(len(task.str_vocab), N_EMBED),
                initializer=tf.uniform_unit_scaling_initializer())
        t_hint_vecs = tf.get_variable(
                "hint_vec", shape=(len(task.hint_vocab), N_EMBED_WORD),
                initializer=tf.uniform_unit_scaling_initializer())

        #t_enc_ex_all = encode(
        #        "encode_ex", self.t_ex, self.t_ex_len, t_str_vecs)
        #t_enc_hint = _encode(
        #        "encode_hint", self.t_hint, self.t_hint_len, t_hint_vecs)
        #t_enc_ex = tf.reduce_mean(t_enc_ex_all, axis=1)
        t_enc_hint = None
        t_enc_ex = None

        t_maml_repr = tf.get_variable(
                "maml_repr", shape=(1, N_HIDDEN,),
                initializer=tf.uniform_unit_scaling_initializer())
        t_n_batch = tf.shape(self.t_maml_input)[0]
        t_maml_repr_tiled = tf.tile(t_maml_repr, (t_n_batch, 1))

        t_enc_maml_input = _encode(
                "encode_input", self.t_input, self.t_input_len, t_str_vecs,
                t_init=t_maml_repr_tiled)
        maml_decoder = Decoder(
                "decode_out", t_enc_maml_input, self.t_maml_output,
                self.t_maml_last_out, self.t_maml_last_out_hidden, t_str_vecs)
        t_maml_grads, = tf.gradients(maml_decoder.t_loss, t_maml_repr_tiled)
        # TODO
        t_maml_repr_step = t_maml_repr_tiled - 0.1 * t_maml_grads

        if FLAGS.infer_hyp:
            t_concept = t_enc_hint
        elif FLAGS.infer_maml:
            t_concept = t_maml_repr_step
        else:
            t_concept = t_enc_ex

        t_enc_input = _encode(
                "encode_input", self.t_input, self.t_input_len, t_str_vecs,
                t_init=t_concept, reuse=FLAGS.infer_maml)
        #self.hyp_decoder = Decoder(
        #        "decode_hyp", t_enc_ex, self.t_hint, self.t_last_hyp,
        #        self.t_last_hyp_hidden, t_hint_vecs)
        self.out_decoder = Decoder(
                "decode_out", t_enc_input, self.t_output, self.t_last_out,
                self.t_last_out_hidden, t_str_vecs, reuse=FLAGS.infer_maml)

        if FLAGS.predict_hyp:
            self.t_loss = self.out_decoder.t_loss + self.hyp_decoder.t_loss
        else:
            self.t_loss = self.out_decoder.t_loss

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.o_train = optimizer.minimize(self.t_loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if FLAGS.restore is not None:
            self.restore(FLAGS.restore)

    def feed(self, batch, input_examples=False):
        max_hint_len = max(len(d.hint) for d in batch)
        max_einp_len = max(len(e) for d in batch for e in d.ex_inputs)
        max_eout_len = max(len(e) for d in batch for e in d.ex_outputs)
        max_inp_len = max(len(d.input) for d in batch)
        max_inp_len = max(max_inp_len, max_einp_len)
        max_out_len = max(len(d.output) for d in batch)
        max_out_len = max(max_out_len, max_eout_len)
        max_ex_len = max_einp_len + max_eout_len + 1

        max_inp_len = self.task.max_input_len
        max_out_len = self.task.max_output_len

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
        minp = np.zeros(inp.shape)
        mout = np.zeros(out.shape)
        for i, datum in enumerate(batch):
            examples = zip(datum.ex_inputs, datum.ex_outputs)
            target_inp = datum.input
            target_out = datum.output
            if input_examples:
                for j, (e_inp, e_out) in enumerate(examples):
                    inp[i, j, :len(e_inp)] = e_inp
                    inp_len[i, j] = len(e_inp)
                    out[i, j, :len(e_out)] = e_out
                minp[i, j, :len(e_inp)] = e_inp
                mout[i, j, :len(e_out)] = e_out
            else:
                inp[i, 0, :len(target_inp)] = target_inp
                inp_len[i, 0] = len(target_inp)
                out[i, 0, :len(target_out)] = target_out

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
            self.t_output: out,

            self.t_maml_input: minp,
            self.t_maml_output: mout
        }

    def train(self, batch):
        feed = self.feed(batch)
        loss, _ = self.session.run([self.t_loss, self.o_train], feed)
        return loss

    def hypothesize(self, batch):
        hyp_init = [self.task.hint_vocab[self.task.START] for _ in batch]
        hyp_stop = self.task.hint_vocab[self.task.STOP]
        feed = self.feed(batch)

        best_score = [-np.inf] * len(batch)
        best_hyps = [None] * len(batch)
        worst_score = [np.inf] * len(batch)

        for i in range(FLAGS.n_sample_hyps):
            hyps = self.hyp_decoder.decode(
                    hyp_init, hyp_stop, feed, self.session,
                    temp=None if i == 0 else 1)
            hyp_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
            hyp_feed = self.feed(hyp_batch, input_examples=True)

            init = self.task.str_vocab[self.task.START] * np.ones(hyp_feed[self.t_ex_len].shape, dtype=np.int32)
            stop = self.task.str_vocab[self.task.STOP]
            preds = self.out_decoder.decode(init, stop, hyp_feed, self.session)
            scores = self.out_decoder.score(init, hyp_feed, self.session)

            for j in range(len(batch)):
                if FLAGS.infer_by_likelihood:
                    score = scores[j, :].sum()
                else:
                    ex_here = hyp_feed[self.t_output][j, ...]
                    score = 0
                    for ex, pred in zip(ex_here, preds[j]):
                        if np.all(ex[:len(pred)] == pred):
                            score += 1

                if score > best_score[j]:
                    best_score[j] = score
                    best_hyps[j] = hyps[j]
                if score < worst_score[j]:
                    worst_score[j] = score

        hyps = best_hyps
        print >>sys.stderr, best_score
        #print >>sys.stderr, worst_score

        print >>sys.stderr, "\n".join(" ".join(self.task.hint_vocab.get(c) for c in hyp) for hyp in hyps[:3])
        print >>sys.stderr, "\n".join(" ".join(self.task.hint_vocab.get(c) for c in hyp if c) for hyp in feed[self.t_hint][:3])
        print >>sys.stderr

        return hyps

    def predict(self, batch):
        if FLAGS.infer_hyp and not FLAGS.use_true_hyp:
            hyps = self.hypothesize(batch)
            pred_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
        else:
            pred_batch = batch

        init = [[self.task.str_vocab[self.task.START]] for _ in batch]
        stop = self.task.str_vocab[self.task.STOP]

        pred_feed = self.feed(pred_batch)
        preds = self.out_decoder.decode(init, stop, pred_feed, self.session)
        accs = []
        for i, (pred, gold) in enumerate(zip(preds, pred_feed[self.t_output])):
            pred = pred[0]
            gold = gold[0].tolist()
            gold = gold[:gold.index(self.task.str_vocab[self.task.STOP])+1]
            accs.append(pred == gold)
        return np.mean(accs)

    def save(self):
        self.saver.save(self.session, "model.chk")

    def restore(self, path):
        self.saver.restore(self.session, path)

