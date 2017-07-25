import util

from collections import namedtuple
import numpy as np
import os
from PIL import Image
import sys
import json

USE_IMAGES = False
N_EX = 6

sw_path = os.path.join(sys.path[0], "data/shapeworld")

Fold = namedtuple("Fold", ["hints", "examples", "inputs", "labels"])
Datum = namedtuple("Datum", ["hint", "ex_inputs", "input", "label"])

START = "<s>"
STOP = "</s>"

random = util.next_random()

class ShapeworldTask():
    def __init__(self):
        self.hint_vocab = util.Index()
        self.feature_index = util.Index()
        self.START = START
        self.STOP = STOP

        with open(os.path.join(sw_path, "train", "examples.struct.json")) as feature_f:
            feature_data = json.load(feature_f)
            for datum in feature_data:
                for example in datum:
                    for feature in example:
                        self.feature_index.index(tuple(feature))

        data = {}
        for fold in ("train", "validation", "test"):
            examples = np.load(os.path.join(sw_path, fold, "examples.npy"))
            inputs = np.load(os.path.join(sw_path, fold, "inputs.npy"))
            labels = np.load(os.path.join(sw_path, fold, "labels.npy"))

            with open(os.path.join(sw_path, fold, "hints.json")) as hint_f:
                hints = json.load(hint_f)
            indexed_hints = []
            for hint in hints:
                hint = [START] + hint.split() + [STOP]
                indexed_hint = [self.hint_vocab.index(w) for w in hint]
                indexed_hints.append(indexed_hint)
            hints = indexed_hints

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
                        hints[i], examples[i, ...], inputs[i, ...], labels[i]))
                else:
                    fold_data.append(Datum(
                        hints[i], ex_features[i, ...], inp_features[i, ...], labels[i]))
            data[fold] = fold_data

        self.train_data = data["train"]
        self.val_data = data["validation"]
        self.test_data = data["test"]

        if USE_IMAGES:
            self.width, self.height, self.channels = self.train_data[0].input.shape
        else:
            #self.n_features = len(self.feature_index)
            self.n_features = inp_features.shape[1]

    def sample_train(self, n_batch):
        batch = []
        n_train = len(self.train_data)
        for _ in range(n_batch):
            datum = self.train_data[random.randint(n_train)]
            in_examples = datum.ex_inputs
            out_examples = []
            #for i_ex in range(N_EX):
            #    out_examples.append(
            #            in_examples[random.randint(in_examples.shape[0]), ...])
            indices = list(range(in_examples.shape[0]))
            random.shuffle(indices)
            indices = indices[:N_EX]
            out_examples = [in_examples[i, ...] for i in indices]
            #out_examples = in_examples[:N_EX, ...]
            datum = datum._replace(ex_inputs=np.asarray(out_examples))
            batch.append(datum)
        return batch

    def sample_val(self):
        batch = []
        for datum in self.val_data:
            datum = datum._replace(ex_inputs=datum.ex_inputs[:N_EX, ...])
            batch.append(datum)
        return batch

    def sample_test(self):
        batch = []
        for datum in self.test_data:
            datum = datum._replace(ex_inputs=datum.ex_inputs[:N_EX, ...])
            batch.append(datum)
        return batch
