from misc import util

from collections import namedtuple
import gflags
import json
import re
import sys
import os

FLAGS = gflags.FLAGS
gflags.DEFINE_string("hint_type", None, "hint format")

SEP = "@"
START = "<"
STOP = ">"
N_EX = 5

FullDatum = namedtuple("FullDatum", ["hints", "pairs"])
Datum = namedtuple("Datum", ["hint", "ex_inputs", "ex_outputs", "input", "output"])

random = util.next_random()

class RegexTask():
    def __init__(self):
        assert FLAGS.hint_type in ("re", "nl", "none")

        with open(os.path.join(sys.path[0], "data/re2/corpus.json")) as corpus_f:
            corpus = json.load(corpus_f)

        self.hint_vocab = util.Index()
        self.str_vocab = util.Index()
        self.str_vocab.index(SEP)
        self.SEP = SEP
        self.START = START
        self.STOP = STOP

        data = {}
        for fold in ["train", "val", "test"]:
            data[fold] = []
            for example in corpus[fold]:
                if FLAGS.hint_type == "re":
                    hint = example["re"]
                    hint = [self.hint_vocab.index(c) for c in hint]
                    hints = [hint]
                elif FLAGS.hint_type == "nl":
                    hints = []
                    for hint in example["hints_aug"]:
                        hint = [self.hint_vocab.index(w) for w in hint]
                        hints.append(hint)
                elif FLAGS.hint_type == "none":
                    hints = [[]]

                pairs = []
                for inp, out in example["examples"]:
                    inp = [self.str_vocab.index(c) for c in inp]
                    out = [self.str_vocab.index(c) for c in out]
                    pairs.append((inp, out))

                datum = FullDatum(hints, pairs)
                data[fold].append(datum)

        self.train_data = data["train"]
        self.val_data = data["val"]
        self.test_data = data["test"]

    def sample_train(self, n_batch):
        batch = []
        n_train = len(self.train_data)
        for _ in range(n_batch):
            full_datum = self.train_data[random.randint(n_train)]
            pairs = list(full_datum.pairs)
            random.shuffle(pairs)
            hint = full_datum.hints[random.randint(len(full_datum.hints))]
            inp, out = pairs.pop()
            pairs = pairs[:N_EX]
            ex_inputs, ex_outputs = zip(*pairs)
            assert len(ex_inputs) == N_EX
            datum = Datum(hint, ex_inputs, ex_outputs, inp, out)
            batch.append(datum)
        return batch

    def sample_val(self):
        batch = []
        for full_datum in self.val_data:
            pairs = list(full_datum.pairs)
            inp, out = pairs.pop()
            ex_inputs, ex_outputs = zip(*pairs)
            assert len(ex_inputs) == N_EX
            if FLAGS.use_true_hyp or FLAGS.use_true_eval or FLAGS.vis:
                hint = full_datum.hints[random.randint(len(full_datum.hints))]
            else:
                hint = []
            batch.append(Datum(hint, ex_inputs, ex_outputs, inp, out))
        return batch

    def sample_test(self):
        batch = []
        for full_datum in self.test_data:
            pairs = list(full_datum.pairs)
            inp, out = pairs.pop()
            ex_inputs, ex_outputs = zip(*pairs)
            assert len(ex_inputs) == N_EX
            batch.append(Datum([], ex_inputs, ex_outputs, inp, out))
        return batch

    def execute(self, hint, inps, outs):
        if 0 in hint:
            return [-1 for _ in inps], ["" for _ in inps]
        hint = "".join(self.hint_vocab.get(t) for t in hint)
        hint = hint[1:-1]
        if hint.count("@") != 1:
            return [-1 for _ in inps], ["" for _ in inps]
        before, after = hint.split("@")
        before = before.replace("C", "[^aeiou]").replace("V", "[aeiou]")

        inps = [inp[1:-1] for inp in inps]
        inps = ["".join(self.str_vocab.get(w) for w in inp) for inp in inps]
        outs = [out[1:-1] for out in outs]
        outs = ["".join(self.str_vocab.get(w) for w in out) for out in outs]

        try:
            predicted_out_strs = [re.sub(before, after, inp) for inp in inps]
        except Exception as e:
            return [-1 for _ in inps], ["" for _ in inps]
        score = [float(p == o) for p, o in zip(outs, predicted_out_strs)]

        predicted_outs = []
        for s in predicted_out_strs:
            predicted_outs.append(
                    [self.str_vocab[START]] 
                    + [self.str_vocab[w] for w in s]
                    + [self.str_vocab[STOP]])

        return score, predicted_outs

    def visualize(self, datum, hyp, pred):
        for i in range(N_EX):
            print "".join(self.str_vocab.get(w) for w in datum.ex_inputs[i][1:-1]),
            print "".join(self.str_vocab.get(w) for w in datum.ex_outputs[i][1:-1])
        print
        print "gold:", " ".join(self.hint_vocab.get(w) for w in datum.hint[1:-1])
        print "pred:", " ".join(self.hint_vocab.get(w) for w in hyp[1:-1])
        print
        print "".join(self.str_vocab.get(w) for w in datum.input)
        print "gold:", "".join(self.str_vocab.get(w) for w in datum.output)
        print "pred:", "".join(self.str_vocab.get(w) for w in pred)
        print "==="
        print
