from misc import util

from collections import namedtuple
import numpy as np
import os
from PIL import Image
import sys
import json
import scipy
import gflags

FLAGS = gflags.FLAGS

USE_IMAGES = False
#N_EX = 6
N_EX = 4

sw_path = os.path.join(sys.path[0], "data/shapeworld")

Fold = namedtuple("Fold", ["hints", "examples", "inputs", "labels"])
Datum = namedtuple("Datum", ["hint", "ex_inputs", "input", "label", "task_id"])
VisDatum = namedtuple("VisDatum", ["hint", "ex_inputs", "input", "label", "vis_ex_inputs", "vis_input", "task_id"])

START = "<s>"
STOP = "</s>"

random = util.next_random()

class ShapeworldTask():
    def __init__(self):
        self.hint_vocab = util.Index()
        self.feature_index = util.Index()
        self.task_index = util.Index()
        self.START = START
        self.STOP = STOP

        #with open(os.path.join(sw_path, "train", "examples.struct.json")) as feature_f:
        #    feature_data = json.load(feature_f)
        #    for datum in feature_data:
        #        for example in datum:
        #            for feature in example:
        #                self.feature_index.index(tuple(feature))

        data = {}
        for fold in ("train", "val", "test", "val_same", "test_same"):
            examples = np.load(os.path.join(sw_path, fold, "examples.npy"))
            inputs = np.load(os.path.join(sw_path, fold, "inputs.npy"))
            labels = np.load(os.path.join(sw_path, fold, "labels.npy"))

            with open(os.path.join(sw_path, fold, "hints.json")) as hint_f:
                hints = json.load(hint_f)

            #new_hints = []
            #for hint in hints:
            #    hint = hint.split()
            #    new_hint = []
            #    for i in range(len(hint) - 1):
            #        new_hint.append(hint[i] + "/" + hint[i+1])
            #    new_hints.append(" ".join(new_hint))
            #hints = new_hints

            indexed_hints = []
            for hint in hints:
                hint = [START] + hint.split() + [STOP]
                indexed_hint = [self.hint_vocab.index(w) for w in hint]
                indexed_hints.append(indexed_hint)
            hints = indexed_hints

            tasks = []
            for hint in hints:
                tasks.append(self.task_index.index(tuple(hint)))

            #ex_features = np.zeros((examples.shape[0], examples.shape[1], len(self.feature_index)))
            #inp_features = np.zeros((examples.shape[0], len(self.feature_index)))
            #with open(os.path.join(sw_path, fold, "examples.struct.json")) as ex_struct_f:
            #    examples_struct = json.load(ex_struct_f)
            #    for i_datum, expls in enumerate(examples_struct):
            #        for i_ex, example in enumerate(expls):
            #            for feature in example:
            #                i_feat = self.feature_index[tuple(feature)]
            #                if i_feat:
            #                    ex_features[i_datum, i_ex, i_feat] = 1
            #with open(os.path.join(sw_path, fold, "inputs.struct.json")) as in_struct_f:
            #    inputs_struct = json.load(in_struct_f)
            #    for i_datum, example in enumerate(inputs_struct):
            #        for feature in example:
            #            i_feat = self.feature_index[tuple(feature)]
            #            if i_feat is not None:
            #                inp_features[i_datum, i_feat] = 1
            ex_features = np.load(os.path.join(sw_path, fold, "examples.feats.npy"))
            inp_features = np.load(os.path.join(sw_path, fold, "inputs.feats.npy"))

            fold_data = []

            for i in range(len(hints)):
                if USE_IMAGES:
                    fold_data.append(Datum(
                        hints[i], examples[i, ...], inputs[i, ...], labels[i], tasks[i]))
                else:
                    fold_data.append(Datum(
                        hints[i], ex_features[i, ...], inp_features[i, ...], labels[i], tasks[i]))
                    if FLAGS.vis:
                        # TODO this is so dirty!
                        datum = fold_data[-1]
                        fold_data[-1] = VisDatum(
                            datum.hint, datum.ex_inputs, datum.input,
                            datum.label, examples[i, ...], inputs[i, ...], tasks[i])
            data[fold] = fold_data

        self.train_data = data["train"]
        self.val_data = data["val"]
        self.test_data = data["test"]
        self.val_same_data = data["val_same"]
        self.test_same_data = data["test_same"]

        #self.train_data = data["train"][:8000]
        #self.val_data = data["train"][8000:8500]
        #self.test_data = data["train"][8500:9000]

        if USE_IMAGES:
            self.width, self.height, self.channels = self.train_data[0].input.shape
        else:
            #self.n_features = len(self.feature_index)
            self.n_features = inp_features.shape[1]

    def sample_train(self, n_batch, augment):
        n_train = len(self.train_data)
        batch = []

        #for _ in range(n_batch):
        #    datum = self.train_data[random.randint(n_train)]
        #    batch.append(datum)

        for _ in range(n_batch):
            datum = self.train_data[random.randint(n_train)]
            if not augment:
                batch.append(datum)
                continue

            label = random.randint(2)
            if label == 0:
                alt_datum = self.train_data[random.randint(n_train)]
                swap = random.randint(N_EX + 1)
                if swap == N_EX:
                    feats = alt_datum.input
                else:
                    feats = alt_datum.ex_inputs[swap, ...]
                datum = datum._replace(input=feats, label=0)

            elif label == 1:
                swap = random.randint((N_EX + 1 if datum.label == 1 else N_EX))
                if swap != N_EX:
                    examples = datum.ex_inputs.copy()
                    feats = examples[swap, ...]
                    if datum.label == 1:
                        examples[swap, ...] = datum.input
                    else:
                        examples[swap, ...] = examples[random.randint(N_EX), ...]
                    datum = datum._replace(input=feats, ex_inputs=examples, label=1)

            batch.append(datum)

            #if datum.label == 0:
            #    batch.append(datum)
            #    continue
            #swap = random.randint(N_EX + 1)
            #if swap == N_EX:
            #    batch.append(datum)
            #    continue
            #examples = datum.ex_inputs.copy()
            #tmp = examples[swap, ...]
            #examples[swap, ...] = datum.input
            #datum = datum._replace(ex_inputs=examples, input=tmp)
            #batch.append(datum)

        #for _ in range(n_batch):
        #    datum = self.train_data[random.randint(n_train)]
        #    in_examples = datum.ex_inputs
        #    out_examples = []
        #    #for i_ex in range(N_EX):
        #    #    out_examples.append(
        #    #            in_examples[random.randint(in_examples.shape[0]), ...])
        #    indices = list(range(in_examples.shape[0]))
        #    random.shuffle(indices)
        #    indices = indices[:N_EX]
        #    out_examples = [in_examples[i, ...] for i in indices]
        #    #out_examples = in_examples[:N_EX, ...]
        #    datum = datum._replace(ex_inputs=np.asarray(out_examples))
        #    batch.append(datum)

        return batch

    def sample_val(self, same=False):
        if same:
            return self.val_same_data
        else:
            return self.val_data

    def sample_test(self, same=False):
        if same:
            return self.test_same_data
        else:
            return self.test_data

    def sample_safe(self, data):
        out = []
        for datum in data:
            swap = random.randint(N_EX)
            swap_with = random.choice([i for i in range(N_EX) if i != swap])
            examples = datum.ex_inputs.copy()
            examples[swap, ...] = examples[swap_with, ...]
            datum = datum._replace(input=swap, label=1, ex_inputs=examples)
            out.append(datum)
        return out

    def sample_test_safe(self, same=False):
        return self.sample_safe(self.test_same_data if same else self.test_data)

    def sample_val_safe(self, same=False):
        return self.sample_safe(self.val_same_data if same else self.val_data)

    def visualize(self, datum, hyp, pred, dest):
        hint = " ".join(self.hint_vocab.get(w) for w in datum.hint[1:-1])
        hyp = " ".join(self.hint_vocab.get(w) for w in hyp[1:-1])
        os.mkdir(dest)
        with open(os.path.join(dest, "desc.txt"), "w") as desc_f:
            print >>desc_f, "gold desc:", hint
            print >>desc_f, "pred desc:", hyp
            print >>desc_f, "gold label:", bool(datum.label)
            print >>desc_f, "pred label:", bool(pred)
        for i in range(datum.ex_inputs.shape[0]):
            scipy.misc.imsave(
                    os.path.join(dest, "ex_%d.png" % i),
                    datum.vis_ex_inputs[i, ...])
        scipy.misc.imsave(
                os.path.join(dest, "input.png"),
                datum.vis_input)
