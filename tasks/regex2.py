import util

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

random = util.next_random()

FullDatum = namedtuple("FullDatum", ["hints", "pairs"])
Datum = namedtuple("Datum", ["hint", "ex_inputs", "ex_outputs", "input", "output"])

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
            batch.append(Datum([], ex_inputs, ex_outputs, inp, out))
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
