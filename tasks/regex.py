import util

from collections import namedtuple
import exrex
import itertools
import logging
import os
import pickle
import re

N_SAMPLES = 5
REP_LIMIT = 3

N_VAL = 200
N_TEST = 200

Dataset = namedtuple("Dataset", ["data", "nl_vocab", "str_vocab"])
Datum = namedtuple("Datum", ["hint", "re", "pos", "neg", "target", "label"])

random = util.next_random()

if not os.path.exists("data/nlrx/cached.pkl"):
    logging.info("rebuilding cache")

    nl_vocab = util.Index()
    str_vocab = util.Index()

    nls = []
    with open("data/nlrx/src.txt") as nl_f:
        for line in nl_f:
            nls.append([nl_vocab.index(w) for w in line.strip().split()])

    exs = []
    with open("data/nlrx/targ.txt") as re_f:
        for line in re_f:
            exs.append(line.strip())

    pairs = []
    for nl, ex in zip(nls, exs):
        if "~" in ex or "&" in ex:
            continue
        pairs.append((nl, ex))

    pos_samples = []
    distractors = []
    for _, ex in pairs:
        pos_samp = [exrex.getone(ex, REP_LIMIT) for _ in range(N_SAMPLES)]
        pos_samp = [tuple(str_vocab.index(c) for c in s) for s in pos_samp]
        pos_samples.append(pos_samp)
        distractors += [exrex.getone(ex, REP_LIMIT) for _ in range(N_SAMPLES)]

    neg_samples = []
    for (_, ex), pos in zip(pairs, pos_samples):
        samples = []
        max_pos_len = max(len(e) for e in pos)
        min_pos_len = min(len(e) for e in pos)
        i = 0
        while len(samples) < N_SAMPLES:
            i += 1
            samp = random.choice(distractors)
            if i < 1000 and len(samp) > max_pos_len or len(samp) < min_pos_len:
                continue
            if re.match(ex, samp):
                if i > 100:
                    break
                continue
            samples.append(samp)

        samples = [tuple(str_vocab.index(c) for c in s) for s in samples]
        neg_samples.append(samples)
        print len(samples), ex

    data = []
    for (nl, ex), pos, neg in zip(pairs, pos_samples, neg_samples):
        if len(neg) > 0 and random.rand() < 0.5:
            target = neg.pop()
            label = 0
        else:
            target = pos.pop()
            label = 1
        data.append(Datum(nl, ex, pos, neg, target, label))

    dataset = Dataset(data, nl_vocab, str_vocab)

    with open("data/nlrx/cached.pkl", "wb") as pkl_f:
        pickle.dump(dataset, pkl_f, pickle.HIGHEST_PROTOCOL)

with open("data/nlrx/cached.pkl", "rb") as pkl_f:
    dataset = pickle.load(pkl_f)

str_vocab = dataset.str_vocab
nl_vocab = dataset.nl_vocab

train_data = dataset.data[:-(N_VAL+N_TEST)]
val_data = dataset.data[-(N_VAL+N_TEST):-N_TEST]
test_data = dataset.data[-N_TEST:]

for datum in val_data:
    assert datum not in train_data

def sample_train():
    batch = []
    for _ in range(100):
        datum = train_data[random.randint(len(train_data))]
        pos = datum.pos[random.randint(len(datum.pos))]
        neg = datum.neg[random.randint(len(datum.neg))] if datum.neg else []
        new_datum = datum._replace(pos=[pos], neg=[neg])
        batch.append(new_datum)
    return batch

    #hints = [d.hint for d in batch]
    ##hints = [random.randint(len(nl_vocab), size=10) for d in batch]
    ##random.shuffle(hints)
    ##hint = [nl_vocab[w] for w in "lines with a number before a vowel".split()]
    ##hints = [hint for d in batch]
    #batch = [d._replace(hint=h) for d, h in zip(batch, hints)]
    #batch = [d._replace(pos=d.pos[:1], neg=d.neg[:1]) for d in batch]
    #return batch

def sample_test():
    batch = []
    for datum in val_data:
        batch.append(datum._replace(pos=datum.pos[:1], neg=datum.neg[:1]))
    return batch
