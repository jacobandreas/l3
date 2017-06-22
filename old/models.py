from net import _mlp, _embed, _linear
import util

import tensorflow as tf
import numpy as np

#N_EMBED = 64
#N_HIDDEN = 128

N_EMBED = 32
N_HIDDEN = 512

random = util.next_random()

class Model3(object):
    def __init__(self, name, dataset):
        self.dataset = dataset

        self.t_hint = tf.placeholder(tf.int32, (None, None))
        self.t_hint_len = tf.placeholder(tf.int32, (None,))
        self.t_ex = tf.placeholder(tf.int32, (None, None, None))
        self.t_ex_len = tf.placeholder(tf.int32, (None, None))
        t_ex_shape = tf.shape(self.t_ex)
        t_n_batch, t_n_ex, t_len_ex = t_ex_shape[0], t_ex_shape[1], t_ex_shape[2]
        self.t_input = tf.placeholder(tf.int32, (None, None))
        self.t_input_len = tf.placeholder(tf.int32, (None,))
        self.t_output = tf.placeholder(tf.int32, (None, None))

        self.t_last_out = tf.placeholder(tf.int32, (None,))
        self.t_last_out_hidden = tf.placeholder(tf.float32, (None, N_HIDDEN))

        self.t_last_hyp = tf.placeholder(tf.int32, (None,))
        self.t_last_hyp_hidden = tf.placeholder(tf.float32, (None, N_HIDDEN))

        with tf.variable_scope("enc_hint"):
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            t_emb_hint = _embed(self.t_hint, len(dataset.hint_vocab), N_EMBED)
            _, t_enc_hint = tf.nn.dynamic_rnn(
                    cell, t_emb_hint, self.t_hint_len, dtype=tf.float32)

        with tf.variable_scope("enc_ex"):
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            t_emb_ex = _embed(self.t_ex, len(dataset.str_vocab), N_EMBED)
            t_emb_ex = tf.reshape(t_emb_ex, (t_n_batch * t_n_ex, t_len_ex, N_EMBED))
            t_ex_len = tf.reshape(self.t_ex_len, (t_n_batch * t_n_ex,))
            _, t_enc_ex = tf.nn.dynamic_rnn(
                    cell, t_emb_ex, t_ex_len, dtype=tf.float32)
            t_enc_ex_all = tf.reshape(t_enc_ex, (t_n_batch, t_n_ex, N_HIDDEN))
            self.t_enc_ex = tf.reduce_mean(t_enc_ex_all, axis=1)

        #t_command = t_enc_hint
        t_command = self.t_enc_ex
        #t_mix = tf.expand_dims(tf.random_uniform(shape=(t_n_batch,)), 1)
        #t_mix = tf.multinomial(tf.zeros((t_n_batch, 2)), 1)
        #t_mix = tf.cast(t_mix, tf.float32)
        #t_command = t_mix * t_enc_hint + (1 - t_mix) * t_enc_ex_pool

        with tf.variable_scope("enc_input"):
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            t_emb_input = _embed(self.t_input, len(dataset.str_vocab), N_EMBED)
            #_, t_enc_input = tf.nn.dynamic_rnn(
            #        cell, t_emb_input, self.t_input_len, dtype=tf.float32)
            _, t_enc_input = tf.nn.dynamic_rnn(
                    cell, t_emb_input, self.t_input_len,
                    initial_state=t_command)

        #self.t_repr = t_enc_input + t_enc_hint # + tf.reduce_mean(t_enc_ex, axis=1)
        #self.t_repr = t_enc_input * tf.reduce_mean(t_enc_ex, axis=1)
        #self.t_repr = t_enc_input * t_enc_hint
        #self.t_repr = t_enc_input + t_command
        self.t_repr = t_enc_input

        with tf.variable_scope("dec_output") as scope:
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            #cell = tf.contrib.rnn.OutputProjectionWrapper(cell, len(dataset.str_vocab))

            t_emb_output = _embed(self.t_output, len(dataset.str_vocab), N_EMBED)
            t_dec_state, _ = tf.nn.dynamic_rnn(
                    cell, t_emb_output, initial_state=self.t_repr, scope=scope)
            # TODO why?
            v_proj = tf.get_variable("w",
                    shape=(N_HIDDEN, len(dataset.str_vocab)),
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.43))
            b_proj = tf.get_variable("b", shape=(len(dataset.str_vocab),),
                    initializer=tf.constant_initializer(0))
            t_pred = tf.einsum("ijk,kl->ijl", t_dec_state, v_proj) + b_proj
            self.t_dec_pred = t_pred

            t_dec_err = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.t_output[:, 1:], logits=t_pred[:, :-1])
            t_dec_output_loss = tf.reduce_mean(tf.reduce_sum(t_dec_err, axis=1))

            scope.reuse_variables()

            #cell = tf.contrib.rnn.GRUCell(N_HIDDEN, reuse=True)
            t_emb_last = _embed(self.t_last_out, len(dataset.str_vocab), N_EMBED)
            self.t_next_out_hidden = cell(t_emb_last, self.t_last_out_hidden)[0]
            t_next_logit = tf.einsum("ij,jk->ik", self.t_next_out_hidden, v_proj) + b_proj
            self.t_next_out = tf.nn.softmax(t_next_logit, 1)

        with tf.variable_scope("dec_hyp") as scope:
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            t_emb_hyp = _embed(self.t_hint, len(dataset.hint_vocab), N_EMBED)
            t_dec_state, _ = tf.nn.dynamic_rnn(
                    cell, t_emb_hyp, initial_state=self.t_enc_ex, scope=scope)

            v_proj = tf.get_variable("w",
                    shape=(N_HIDDEN, len(dataset.hint_vocab)),
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.43))
            b_proj = tf.get_variable("b", shape=(len(dataset.hint_vocab),),
                    initializer=tf.constant_initializer(0))
            t_pred = tf.einsum("ijk,kl->ijl", t_dec_state, v_proj) + b_proj

            t_dec_err = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.t_hint[:, 1:], logits=t_pred[:, :-1])
            t_dec_hyp_loss = tf.reduce_mean(tf.reduce_sum(t_dec_err, axis=1))

            scope.reuse_variables()

            t_emb_last = _embed(self.t_last_hyp, len(dataset.hint_vocab), N_EMBED)
            self.t_next_hyp_hidden = cell(t_emb_last, self.t_last_hyp_hidden)[0]
            t_next_logit = tf.einsum("ij,jk->ik", self.t_next_hyp_hidden, v_proj + b_proj)
            self.t_next_hyp = tf.nn.softmax(t_next_logit, 1)

        self.t_loss = t_dec_output_loss + t_dec_hyp_loss

    def feed(self, batch):
        sep = self.dataset.str_vocab[self.dataset.SEP]

        max_hint_len = max(len(d.hint) for d in batch)
        #max_ex_count = max(len(d.inputs) for d in batch)
        max_ex_count = 5
        max_inp_len = max(len(i) for d in batch for i in d.inputs)
        max_out_len = max(len(o) for d in batch for o in d.outputs)
        max_ex_len = max_inp_len + 1 + max_out_len

        hint = np.zeros((len(batch), max_hint_len), dtype=np.int32)
        hint_len = np.zeros((len(batch),), dtype=np.int32)
        example = np.zeros((len(batch), max_ex_count-1, max_ex_len), dtype=np.int32)
        example_len = np.zeros((len(batch), max_ex_count-1), dtype=np.int32)
        inp = np.zeros((len(batch), max_inp_len), dtype=np.int32)
        inp_len = np.zeros((len(batch),), dtype=np.int32)
        out = np.zeros((len(batch), max_out_len), dtype=np.int32)
        for i, datum in enumerate(batch):
            examples = list(zip(datum.inputs, datum.outputs))
            #random.shuffle(examples)
            examples = examples[:5]

            target_inp, target_out = examples.pop()
            inp[i, :len(target_inp)] = target_inp
            inp_len[i] = len(target_inp)
            out[i, :len(target_out)] = target_out

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

    def hypothesize(self, batch, session):
        last_hyp = [self.dataset.hint_vocab[self.dataset.START] for _ in batch]
        hyp = [[l] for l in last_hyp]
        feed = self.feed(batch)
        last_hidden, = session.run([self.t_enc_ex], feed)
        for t in range(20):
            next_hyp_prob, next_hidden = session.run(
                    [self.t_next_hyp, self.t_next_hyp_hidden],
                    {self.t_last_hyp: last_hyp, self.t_last_hyp_hidden: last_hidden})
            #next_hyp = np.argmax(next_hyp_prob, axis=1)
            next_hyp = []
            for row in next_hyp_prob:
                #row = np.exp(10 * row)
                #row /= np.sum(row)
                #next_hyp.append(random.choice(row.size, p=row))
                next_hyp.append(np.argmax(row))
            for w, ww in zip(next_hyp, hyp):
                if ww[-1] != self.dataset.hint_vocab[self.dataset.STOP]:
                    ww.append(w)
            last_hidden = next_hidden
            last_hyp = next_hyp
        return hyp

    def predict(self, batch, session):
        hyps = [None] * len(batch)
        hyp_accs = [-1] * len(batch)
        eos = self.dataset.hint_vocab[self.dataset.STOP]

        #for i in range(10):
        #    trial_hyps = self.hypothesize(batch, session)
        #    trial_batch = [d._replace(hint=h) for d, h in zip(batch, trial_hyps)]
        #    trial_feed = self.feed(trial_batch)
        #    trial_scores, = session.run([self.t_dec_pred], trial_feed)
        #    trial_pred = np.argmax(trial_scores, axis=2)
        #    agree = np.sum(trial_pred[:, :-1] == trial_feed[self.t_output][:, 1:], axis=1)
        #    for j in range(len(batch)):
        #        if agree[j] > hyp_accs[j]:
        #            hyp_accs[j] = agree[j]
        #            hyps[j] = trial_hyps[j]
        #    #hyp_accs += agree
        #print hyp_accs

        hyps = self.hypothesize(batch, session)

        orig_batch = batch
        batch = [d._replace(hint=h) for d, h in zip(batch, hyps)]
        feed = self.feed(batch)

        last_out = [self.dataset.str_vocab[self.dataset.START] for _ in batch]
        out = [[l] for l in last_out]
        last_hidden, = session.run([self.t_repr], feed)
        for t in range(20):
            next_out_prob, next_hidden = session.run(
                    [self.t_next_out, self.t_next_out_hidden],
                    {self.t_last_out: last_out, self.t_last_out_hidden: last_hidden})
            next_out = np.argmax(next_out_prob, axis=1)
            for w, ww in zip(next_out, out):
                if ww[-1] != self.dataset.str_vocab[self.dataset.STOP]:
                    ww.append(w)
            last_hidden = next_hidden
            last_out = next_out

        accs = []
        for i, (pred, gold) in enumerate(zip(out, feed[self.t_output])):
            gold = gold.tolist()
            gold = gold[:gold.index(self.dataset.str_vocab[self.dataset.STOP])]
            pred = pred[:-1]
            accs.append(pred == gold)
            if i < 3:
                print "".join(self.dataset.str_vocab.get(c) for c in pred[1:])
                print "".join(self.dataset.hint_vocab.get(c) for c in hyps[i])
                print "".join(self.dataset.str_vocab.get(c) for c in gold[1:])
                print "".join(self.dataset.hint_vocab.get(c) for c in orig_batch[i].hint)
                print
        return np.mean(accs)

        #descriptions = self.sample(batch, session)
        #random.shuffle(descriptions)
        #self_batch = [b._replace(hint=h) for b, h in zip(batch, descriptions)]
        #self_batch = batch
        #preds, = session.run([self.t_score], self.feed(batch))
        #preds, = session.run([self.t_score], self.feed(batch))
        #return (preds > 0).astype(int)


class Model2(object):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset

        self.t_positive = tf.placeholder(tf.int32, (None, None, None))
        t_pos_shape = tf.shape(self.t_positive)
        t_n_batch, t_n_pos, t_len_pos = t_pos_shape[0], t_pos_shape[1], t_pos_shape[2]
        self.t_positive_len = tf.placeholder(tf.int32, (None, None))
        self.t_negative = tf.placeholder(tf.int32, (None, None, None))
        t_neg_shape = tf.shape(self.t_negative)
        t_n_neg, t_len_neg = t_neg_shape[1], t_neg_shape[2]
        self.t_negative_len = tf.placeholder(tf.int32, (None, None))
        self.t_hint = tf.placeholder(tf.int32, (None, None))
        self.t_hint_len = tf.placeholder(tf.int32, (None,))
        self.t_target = tf.placeholder(tf.int32, (None, None))
        self.t_target_len = tf.placeholder(tf.int32, (None,))
        self.t_label = tf.placeholder(tf.float32, (None,))

        self.t_last_hidden = tf.placeholder(tf.float32, (None, N_HIDDEN))
        self.t_last_word = tf.placeholder(tf.int32, (None,))

        with tf.variable_scope("emb") as scope:
            t_emb_pos = _embed(self.t_positive, len(dataset.str_vocab), N_EMBED)
            t_emb_pos = tf.reshape(t_emb_pos, (t_n_batch * t_n_pos, t_len_pos, N_EMBED))
            t_positive_len = tf.reshape(self.t_positive_len, (t_n_batch * t_n_pos,))
            scope.reuse_variables()
            t_emb_neg = _embed(self.t_negative, len(dataset.str_vocab), N_EMBED)
            t_emb_neg = tf.reshape(t_emb_neg, (t_n_batch * t_n_neg, t_len_neg, N_EMBED))
            t_negative_len = tf.reshape(self.t_negative_len, (t_n_batch * t_n_neg,))
            scope.reuse_variables()
            t_emb_target = _embed(self.t_target, len(dataset.str_vocab), N_EMBED)

        with tf.variable_scope("enc_pos"):
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            _, t_hidden_pos = tf.nn.dynamic_rnn(
                    cell, t_emb_pos, t_positive_len, dtype=tf.float32)
            t_hidden_pos = tf.reshape(t_hidden_pos, (t_n_batch, t_n_pos, N_HIDDEN))
            t_red_pos = tf.reduce_mean(t_hidden_pos, axis=1)

        with tf.variable_scope("enc_neg"):
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            _, t_hidden_neg = tf.nn.dynamic_rnn(
                    cell, t_emb_neg, t_negative_len, dtype=tf.float32)
            t_hidden_neg = tf.reshape(t_hidden_neg, (t_n_batch, t_n_neg, N_HIDDEN))
            t_red_neg = tf.reduce_mean(t_hidden_neg, axis=1)

        self.t_example_repr = (t_red_pos + t_red_neg) / 2
        #self.t_example_repr = t_red_pos

        with tf.variable_scope("emb_hint") as scope:
            t_emb_hint = _embed(self.t_hint, len(dataset.nl_vocab), N_EMBED)

        with tf.variable_scope("emb_hint2", reuse=False) as scope:
            t_emb_hint2 = _embed(self.t_hint, len(dataset.nl_vocab), N_EMBED)
            #t_emb_hint2 = tf.stop_gradient(t_emb_hint2)
            scope.reuse_variables()
            t_emb_last = _embed(self.t_last_word, len(dataset.nl_vocab), N_EMBED)

        with tf.variable_scope("enc_hint"):
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            _, t_hint_repr = tf.nn.dynamic_rnn(
                    cell, t_emb_hint, self.t_hint_len, dtype=tf.float32)

        with tf.variable_scope("dec_hint") as scope:
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            #cell = tf.contrib.rnn.OutputProjectionWrapper(cell, len(dataset.nl_vocab))
            t_dec_state, _ = tf.nn.dynamic_rnn(
                    cell, t_emb_hint2, self.t_hint_len,
                    initial_state=tf.stop_gradient(self.t_example_repr), scope=scope)
            v_proj = tf.get_variable("w",
                    shape=(N_HIDDEN, len(dataset.nl_vocab)),
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.43))
            b_proj = tf.get_variable("b", shape=(len(dataset.nl_vocab),),
                    initializer=tf.constant_initializer(0))
            t_pred_hint = tf.einsum("ijk,kl->ijl", t_dec_state, v_proj) + b_proj

            scope.reuse_variables()

            self.t_next_hidden = cell(t_emb_last, self.t_last_hidden)[0]
            t_next_logit = tf.einsum("ij,jk->ik", self.t_next_hidden, v_proj) + b_proj
            self.t_next_word = tf.nn.softmax(t_next_logit, 1)

        with tf.variable_scope("enc_target"):
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            _, t_hidden_target = tf.nn.dynamic_rnn(
                    cell, t_emb_target, self.t_target_len, dtype=tf.float32)

        #t_hint_repr = tf.zeros(tf.shape(t_hint_repr))
        #t_full_repr = (self.t_example_repr + t_hint_repr) / 2
        #t_full_repr = self.t_example_repr
        t_full_repr = t_hint_repr + tf.get_variable("repr_bias",
                shape=(N_HIDDEN,), initializer=tf.constant_initializer(0))

        # hint_scorer
        with tf.variable_scope("hint_score"):
            self.t_hint_score = _mlp(t_hint_repr * t_hidden_target, (1,), (None,))
            self.t_hint_score = tf.reshape(self.t_hint_score, (-1,))
            t_hint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.t_label, logits=self.t_hint_score))

        # full scorer
        with tf.variable_scope("full_score"):
            self.t_full_score = _mlp(t_full_repr * t_hidden_target, (1,), (None,))
            self.t_full_score = tf.reshape(self.t_full_score, (-1,))
            t_full_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.t_label, logits=self.t_full_score))

        # describer
        with tf.variable_scope("describe"):
            t_desc_err = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.t_hint[:, 1:], logits=t_pred_hint[:, :-1])
            t_desc_loss = tf.reduce_mean(tf.reduce_sum(t_desc_err, axis=1))

        self.t_loss = t_full_loss + t_desc_loss
        self.t_score = self.t_full_score
        #self.t_loss = t_hint_loss + t_desc_loss
        #self.t_score = self.t_hint_score

    def sample(self, batch, session):
        # TODO
        last_word = [self.dataset.nl_vocab["lines"] for _ in batch]
        last_hidden, = session.run([self.t_example_repr], self.feed(batch))
        out = [[i] for i in last_word]
        for _ in range(20):
            next_word, next_hidden = session.run(
                    [self.t_next_word, self.t_next_hidden],
                    {self.t_last_word: last_word, self.t_last_hidden: last_hidden})
            chosen_word = []
            for o, w in zip(out, next_word):
                #chosen = random.choice(w.size, p=w)
                chosen = np.argmax(w)
                chosen_word.append(chosen)
                # TODO
                if o[-1] > 0:
                    o.append(chosen)
            last_word = chosen_word
            last_hidden = next_hidden
        return out

    def predict(self, batch, session):
        descriptions = self.sample(batch, session)
        #random.shuffle(descriptions)
        #self_batch = [b._replace(hint=h) for b, h in zip(batch, descriptions)]
        self_batch = batch
        preds, = session.run([self.t_score], self.feed(self_batch))
        #preds, = session.run([self.t_score], self.feed(batch))
        return (preds > 0).astype(int)

    def feed(self, batch):
        n_batch = len(batch)
        max_pos = max(len(d.pos) for d in batch)
        max_pos_len = max(max(len(p) for p in d.pos) for d in batch)
        max_neg = max(len(d.neg) for d in batch)
        max_neg_len = max(max(len(n) for n in d.neg) for d in batch if d.neg)
        max_hint_len = max(len(d.hint) for d in batch)
        max_target_len = max(len(d.target) for d in batch)

        positive = np.zeros((n_batch, max_pos, max_pos_len))
        positive_len = np.zeros((n_batch, max_pos))
        negative = np.zeros((n_batch, max_neg, max_neg_len))
        negative_len = np.zeros((n_batch, max_neg))
        for i_datum, datum in enumerate(batch):
            for i_pos, pos in enumerate(datum.pos):
                positive[i_datum, i_pos, :len(pos)] = pos
                positive_len[i_datum, i_pos] = len(pos)
            for i_neg, neg in enumerate(datum.neg):
                negative[i_datum, i_neg, :len(neg)] = neg
                negative_len[i_datum, i_neg] = len(neg)

        hint = np.zeros((n_batch, max_hint_len))
        hint_len = np.zeros((n_batch,))
        target = np.zeros((n_batch, max_target_len))
        target_len = np.zeros((n_batch,))
        label = np.zeros((n_batch,))
        for i_datum, datum in enumerate(batch):
            hint[i_datum, :len(datum.hint)] = datum.hint
            hint_len[i_datum] = len(datum.hint)
            target[i_datum, :len(datum.target)] = datum.target
            target_len[i_datum] = len(datum.target)
            label[i_datum] = datum.label

        return {
            self.t_positive: positive,
            self.t_positive_len: positive_len,
            self.t_negative: negative,
            self.t_negative_len: negative_len,
            self.t_hint: hint,
            self.t_hint_len: hint_len,
            self.t_target: target,
            self.t_target_len: target_len,
            self.t_label: label
        }

class Model1(object):
    def __init__(self, name):
        self.name = name

        self.t_features = tf.placeholder(tf.float32, (None, dataset.n_features))
        self.t_hint = tf.placeholder(tf.int32, (None, None))
        self.t_hint_len = tf.placeholder(tf.int32, (None,))
        self.t_label = tf.placeholder(tf.int32, (None,))

        with tf.variable_scope("enc"):
            t_embed_hint = _embed(self.t_hint, len(dataset.vocab), N_EMBED)
            cell = tf.contrib.rnn.GRUCell(N_HIDDEN)
            _, t_hidden = tf.nn.dynamic_rnn(cell, t_embed_hint, self.t_hint_len, dtype=tf.float32)
            self.t_proj = tf.nn.l2_normalize(_linear(t_hidden, dataset.n_features), 1)
            self.t_proc_features = tf.nn.l2_normalize(self.t_features, 1)
            #self.t_proj = _linear(t_hidden, dataset.n_features)
            #t_mul = self.t_proj * self.t_proc_features
            t_comb = (self.t_proj * self.t_proc_features)
            t_score = tf.reduce_sum(t_comb, axis=1)
            self.t_loss = 1-tf.reduce_mean(t_score)

    def feed(self, batch):
        features = [datum.features for datum in batch]
        max_hint_len = max(len(datum.hint) for datum in batch)
        hints = np.zeros((len(batch), max_hint_len), dtype=np.int32)
        hint_lens = []
        for i, datum in enumerate(batch):
            hints[i, :len(datum.hint)] = datum.hint
            hint_lens.append(len(datum.hint))
        labels = [datum.label for datum in batch]
        return {
            self.t_features: features,
            self.t_hint: hints,
            self.t_hint_len: hint_lens,
            self.t_label: labels
        }

    #def fit(self, test_exemplars):
    #    self.exemplars = {}
    #    for ex in test_exemplars:
    #        self.exemplars[ex.label] = ex.features

    def fit(self, test_exemplars, session):
        proj, feats = session.run([self.t_proj, self.t_proc_features, ], self.feed(test_exemplars))
        #self.exemplars = {ex.label: np.zeros(ex.features.size) for ex in test_exemplars}
        self.exemplars = {}
        self.counts = {ex.label: 0 for ex in test_exemplars}
        for i_ex, ex in enumerate(test_exemplars):
            if ex.label not in self.exemplars:
                self.exemplars[ex.label] = 100 * feats[i_ex, :]
                self.counts[ex.label] += 100
            self.exemplars[ex.label] += proj[i_ex, :]
            self.counts[ex.label] += 1
        for label in self.counts:
            self.exemplars[label] /= self.counts[label]

    def predict(self, test_data):
        out = []
        for datum in test_data:
            scores = [
                    (l, 
                        np.dot(datum.features, f) 
                        / (np.linalg.norm(datum.features) * np.linalg.norm(f)))
                    for l, f in self.exemplars.items()]
            label = max(scores, key=lambda p: p[1])[0]
            out.append(label)
        return out
