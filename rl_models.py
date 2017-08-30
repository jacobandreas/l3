from models import _encode, Decoder
from models import N_HIDDEN as N_DEC_HIDDEN
from net import _mlp, _linear, _embed_dict
from misc import util

from collections import defaultdict
import gflags
import numpy as np
import tensorflow as tf

FLAGS = gflags.FLAGS
def _set_flags():
    gflags.DEFINE_boolean("predict_hyp", False, "train to predict hypotheses")
    gflags.DEFINE_boolean("infer_hyp", False, "use hypotheses at test time")
    gflags.DEFINE_string("restore", None, "model to restore")
    gflags.DEFINE_boolean("concept_prior", False, "place a standard normal prior on concept representations")

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
        self.t_task = tf.placeholder(tf.int32, (None,))

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

        t_task_vecs = tf.get_variable(
                "task_vec", (len(task.vocab), N_EMBED),
                initializer=tf.uniform_unit_scaling_initializer())
        t_task_repr = _embed_dict(self.t_task, t_task_vecs)

        if FLAGS.infer_hyp:
            self.t_concept = t_hint_repr
        else:
            self.t_concept = t_task_repr

        with tf.variable_scope("features"):
            t_features = _mlp(self.t_state, (N_HIDDEN, N_HIDDEN), (tf.nn.tanh, tf.nn.tanh))
        with tf.variable_scope("param"):
            t_concept_param = _linear(self.t_concept, N_HIDDEN * task.n_actions)
            t_concept_mat = tf.reshape(t_concept_param, (-1, N_HIDDEN, task.n_actions))
        self.t_score = tf.einsum("ij,ijk->ik", t_features, t_concept_mat)

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

        self.t_rl_loss = t_loss_surrogate + t_baseline_err - 0.001 * t_entropy
        self.t_dagger_loss = -tf.reduce_mean(t_chosen_logprob)

        if FLAGS.predict_hyp:
            self.t_loss = self.t_rl_loss + self.hyp_decoder.t_loss
            self.t_dagger_loss = self.t_dagger_loss + self.hyp_decoder.t_loss

        if FLAGS.concept_prior:
            def normal(x):
                return tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=1))
            self.t_loss += normal(self.t_concept)
            self.t_dagger_loss += normal(self.t_concept)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.o_train = optimizer.minimize(self.t_loss)
        self.o_rl_train = optimizer.minimize(self.t_rl_loss)
        self.o_dagger_train = optimizer.minimize(self.t_dagger_loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if FLAGS.restore is not None:
            self.restore(FLAGS.restore)

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
            self.t_hint_len: hint_len,
            self.t_task: [s.task_id for s in states]
        }
        # TODO naming
        if states[0].task_repr is not None:
            concepts = [s.task_repr for s in states]
            feed_dict[self.t_concept] = concepts
        logprobs, = self.session.run([self.t_logprob], feed_dict)
        probs = np.exp(logprobs)
        actions = []
        for i in range(len(states)):
            action = random.choice(self.task.n_actions, p=probs[i, :])
            actions.append(action)
        return actions

    def train(self, transitions, ignore_hyp=False):
        states, actions, _, rewards = zip(*transitions)
        features = [s.features for s in states]
        hint, hint_len = self.load_hint(states)
        feed_dict = {
            self.t_state: features,
            self.t_action: actions,
            self.t_reward: rewards,
            self.t_hint: hint,
            self.t_hint_len: hint_len,
            self.t_task: [s.task_id for s in states]
        }
        t_loss, o_train = (self.t_rl_loss, self.o_rl_train) if ignore_hyp else (self.t_loss, self.o_train)
        loss, _ = self.session.run([t_loss, o_train], feed_dict)
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
            self.t_hint_len: hint_len,
            self.t_task: [s.task_id for s in states]
        }
        loss, _ = self.session.run([self.t_dagger_loss, self.o_dagger_train], feed_dict)
        return loss

    def sample_hyps(self, n):
        init = [self.task.vocab[self.task.START] for _ in range(n)]
        hyps = self.hyp_decoder.decode(
                init,
                self.task.vocab[self.task.STOP], 
                {self.t_n_batch: len(init)},
                self.session,
                temp=1)
        return hyps
        #words = [" ".join(self.task.vocab.get(w) for w in hyp) for hyp in hyps]
        #return words

        #raw = [
        #    "reach diamond", "reach circle", "reach star",
        #    "reach triangle", "reach heart", "reach spade"
        #]
        #encoded = [
        #    [self.task.vocab[self.task.START]]
        #        + [self.task.vocab[w] for w in s.split()]
        #        + [self.task.vocab[self.task.STOP]]
        #    for s in raw]
        #print self.task.vocab.contents
        #print encoded
        #assert not any(None in s for s in encoded)
        #return [encoded[random.randint(len(encoded))] for _ in range(n)]

    def reset(self):
        self._adapt_scores = defaultdict(lambda: 0.)
        self._adapt_counts = defaultdict(lambda: 0)
        self._adapt_all = defaultdict(list)
        self._adapt_total = 0

        if FLAGS.infer_hyp:
            adapt_reprs = self.sample_hyps(50)
            self._adapt_reprs = list(set(tuple(r) for r in adapt_reprs))
            self._n_ad = len(self._adapt_reprs)
            self._ann_counter = 0
        else:
            self._adapt_reprs = random.normal(size=(50, N_EMBED))
            self._n_ad = 20
            self._ann_counter = 0

        self.ready = False

    def adapt(self, transitions, episodes):
        #self.train(transitions)
        #return

        if self._adapt_total < 500:
            for ss, r in episodes:
                s = ss[0][0]
                self._adapt_scores[s.task_id, s.meta] += r
                self._adapt_counts[s.task_id, s.meta] += 1
                self._adapt_all[s.task_id, s.meta].append(r)
                self._adapt_total += 1
        else:
            self.train(transitions)

    def annotate(self, states):
        #return states
        if self._adapt_total < 500:
            n_ad = self._n_ad
            repr_ids = [random.randint(n_ad) for state in states]
            #print repr_ids
            #repr_ids = [self._ann_counter for state in states]
            #self._ann_counter += 1
            #print self._ann_counter
            reprs = [self._adapt_reprs[i] for i in repr_ids]
            new_states = []
            for i, state in enumerate(states):
                if FLAGS.infer_hyp:
                    new_states.append(state.annotate_instruction(reprs[i], repr_ids[i]))
                else:
                    new_states.append(state.annotate_task_repr(reprs[i], repr_ids[i]))
            return new_states
        else:
            # TODO outside
            adapt_means = {}
            for k in self._adapt_scores:
                adapt_means[k] = self._adapt_scores[k] / self._adapt_counts[k]

            if not self.ready:
                print "TRANSITION"
                self.ready = True

                for k, v in sorted(self._adapt_counts.items(), 
                        key=lambda x: adapt_means[x[0]]):
                    print (
                            k,
                            v,
                            " ".join(self.task.vocab.get(w) for w in self._adapt_reprs[k[1]]),
                            adapt_means[k]
                            #self._adapt_all[k])
                            )

            new_states = []
            for state in states:
                valid_keys = [k for k in adapt_means if k[0] == state.task_id]
                best_key = max(valid_keys, key=lambda x: adapt_means[x])
                #print best_key, adapt_means[best_key]
                choose_repr = self._adapt_reprs[best_key[1]]
                if FLAGS.infer_hyp:
                    new_states.append(state.annotate_instruction(choose_repr, best_key[1]))
                else:
                    new_states.append(state.annotate_task_repr(choose_repr, best_key[1]))
            return new_states


    #def hypothesize(self, states):
    #    if not hasattr(self, "_adapt_total") or self._adapt_total < 5000:
    #        return self.sample(states)

    #    adapt_means = {}
    #    for k in self._adapt_scores:
    #        adapt_means[k] = self._adapt_scores[k] / self._adapt_counts[k]

    #    hyps = []
    #    #print self._adapt_counts
    #    for state in states:
    #        valid_keys = [k for k in adapt_means if k[0] == state.task_id]
    #        best_key = max(valid_keys, key=lambda x: adapt_means[x])
    #        hyps.append(best_key[1])

    #    return hyps

    ## TODO copycat
    #def sample_reprs(self, states):
    #    if not hasattr(self, "_adapt_repr_total"):
    #        self._adapt_repr_candidates = random.normal(size=(100, N_EMBED))
    #        self._adapt_repr_total = 0
    #        self._adapt_repr_scores = defaultdict(lambda: 0)
    #        self._adapt_repr_counts = defaultdict(lambda: 0)

    #    if self._adapt_repr_total < 5000:
    #        return [self._adapt_repr_candidates[i, random.randint(100)] 
    #                for _ in range(len(states)]

    #    adapt_means = {}
    #    for k in self._adapt_repr_scores:
    #        adapt_means[k] = self._adapt_repr_scores[k] / self._adapt_counts[k]

    #    reprs = []
    #    for state in states:
    #        valid_keys = [k for k in adapt_means if k[0] == state.task_id]
    #        best_key = max(valid_keys, key=lambda x: adapt_means[x])
    #        reprs.append(best_key[1])

    #    return hyps

    def save(self):
        self.saver.save(self.session, "model.chk")

    def restore(self, path):
        self.saver.restore(self.session, tf.train.latest_checkpoint(path))
