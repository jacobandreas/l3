from net import _mlp, _embed_dict, _linear
import util

import tensorflow as tf
import numpy as np
import sys

N_EMBED = 32
N_EMBED_WORD = 128
N_HIDDEN = 512
N_EX = 5

TRAIN_HYP = True
USE_HYP = True
N_HYPS = 5
USE_GOLD = False

random = util.next_random()

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
            t_n_batch, t_n_multi = t_shape[0], t_shape[1]
            t_last = tf.reshape(t_last, (t_n_batch*t_n_multi,))
            t_last_hidden = tf.reshape(t_last_hidden, (t_n_batch*t_n_multi, N_HIDDEN))

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

            scope.reuse_variables()

            t_next_hidden, _ = cell(t_emb_last, t_last_hidden)
            t_next_pred = tf.einsum("ij,jk->ik", t_next_hidden, v_proj) + b_proj

        if multi:
            t_next_hidden = tf.reshape(t_next_hidden, (t_n_batch, t_n_multi, N_HIDDEN))
            t_next_pred = tf.reshape(t_next_pred, (t_n_batch, t_n_multi, n_vocab))

        self.t_loss = t_dec_loss
        self.t_next_hidden = t_next_hidden
        self.t_next_pred = t_next_pred
        self.multi = multi

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

class Model(object):
    def __init__(self, dataset):
        self.dataset = dataset

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
                "str_vec", shape=(len(dataset.str_vocab), N_EMBED),
                initializer=tf.uniform_unit_scaling_initializer())
        t_hint_vecs = tf.get_variable(
                "hint_vec", shape=(len(dataset.hint_vocab), N_EMBED_WORD),
                initializer=tf.uniform_unit_scaling_initializer())

        t_enc_ex_all = _encode(
                "encode_ex", self.t_ex, self.t_ex_len, t_str_vecs)
        t_enc_hint = _encode(
                "encode_hint", self.t_hint, self.t_hint_len, t_hint_vecs)
        t_enc_ex = tf.reduce_mean(t_enc_ex_all, axis=1)

        if USE_HYP:
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

        if TRAIN_HYP:
            self.t_loss = self.out_decoder.t_loss + self.hyp_decoder.t_loss
        else:
            self.t_loss = self.out_decoder.t_loss

        optimizer = tf.train.AdamOptimizer(0.001)
        self.o_train = optimizer.minimize(self.t_loss)

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        #self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def feed(self, batch, input_examples=False):
        sep = self.dataset.str_vocab[self.dataset.SEP]
        assert sep is not None
        max_hint_len = max(len(d.hint) for d in batch)
        max_inp_len = max(len(i) for d in batch for i in d.inputs)
        max_out_len = max(len(o) for d in batch for o in d.outputs)
        max_ex_len = max_inp_len + 1 + max_out_len

        hint = np.zeros((len(batch), max_hint_len), dtype=np.int32)
        hint_len = np.zeros((len(batch),), dtype=np.int32)
        example = np.zeros((len(batch), N_EX, max_ex_len), dtype=np.int32)
        example_len = np.zeros((len(batch), N_EX), dtype=np.int32)
        if input_examples:
            n_input = N_EX
        else:
            n_input = 1
        inp = np.zeros((len(batch), n_input, max_inp_len), dtype=np.int32)
        inp_len = np.zeros((len(batch), n_input), dtype=np.int32)
        out = np.zeros((len(batch), n_input, max_out_len), dtype=np.int32)
        for i, datum in enumerate(batch):
            examples = list(zip(datum.inputs, datum.outputs))
            examples = examples[:N_EX+1]
            target_inp, target_out = examples.pop()
            if input_examples:
                for j, (e_inp, e_out) in enumerate(examples):
                    inp[i, j, :len(e_inp)] = e_inp
                    inp_len[i, j] = len(e_inp)
                    out[i, j, :len(e_out)] = e_out
            else:
                inp[i, 0, :len(target_inp)] = target_inp
                inp_len[i, 0] = len(target_inp)
                out[i, 0, :len(target_out)] = target_out

            hint[i, :len(datum.hint)] = datum.hint
            hint_len[i] = len(datum.hint)
            for j, (e_inp, e_out) in enumerate(examples):
                exp = e_inp + [sep] + e_out
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

    def train(self, batch):
        feed = self.feed(batch)
        loss, _ = self.session.run([self.t_loss, self.o_train], feed)
        return loss

    def hypothesize(self, batch):
        hyp_init = [self.dataset.hint_vocab[self.dataset.START] for _ in batch]
        hyp_stop = self.dataset.hint_vocab[self.dataset.STOP]
        feed = self.feed(batch)

        best_score = [-1] * len(batch)
        best_hyps = [None] * len(batch)
        worst_score = [6] * len(batch)

        for i in range(N_HYPS):
            hyps = self.hyp_decoder.decode(
                    hyp_init, hyp_stop, feed, self.session,
                    temp=None if i == 0 else 1)
            hyp_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
            hyp_feed = self.feed(hyp_batch, input_examples=True)

            init = self.dataset.str_vocab[self.dataset.START] * np.ones(hyp_feed[self.t_ex_len].shape, dtype=np.int32)
            stop = self.dataset.str_vocab[self.dataset.STOP]
            preds = self.out_decoder.decode(init, stop, hyp_feed, self.session)

            for j in range(len(batch)):
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

        print >>sys.stderr, "\n".join(" ".join(self.dataset.hint_vocab.get(c) for c in hyp) for hyp in hyps[:3])
        print >>sys.stderr, "\n".join(" ".join(self.dataset.hint_vocab.get(c) for c in hyp if c) for hyp in feed[self.t_hint][:3])
        print >>sys.stderr

        return hyps

    def predict(self, batch):
        if USE_HYP and not USE_GOLD:
            hyps = self.hypothesize(batch)
            pred_batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
        else:
            pred_batch = batch

        init = [[self.dataset.str_vocab[self.dataset.START]] for _ in batch]
        stop = self.dataset.str_vocab[self.dataset.STOP]

        pred_feed = self.feed(pred_batch)
        preds = self.out_decoder.decode(init, stop, pred_feed, self.session)
        accs = []
        for i, (pred, gold) in enumerate(zip(preds, pred_feed[self.t_output])):
            pred = pred[0]
            gold = gold[0].tolist()
            gold = gold[:gold.index(self.dataset.str_vocab[self.dataset.STOP])+1]
            accs.append(pred == gold)
        return np.mean(accs)

    def save(self):
        self.saver.save(self.session, "model")

