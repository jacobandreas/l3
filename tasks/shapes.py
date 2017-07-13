import util

from collections import namedtuple
import numpy as np
import os
from PIL import Image
import sys
import json

sw_path = os.path.join(sys.path[0], "data/shapeworld")

Fold = namedtuple("Fold", ["hints", "examples", "inputs", "labels"])
Datum = namedtuple("Datum", ["hint", "ex_inputs", "input", "label"])

START = "<s>"
STOP = "</s>"

random = util.next_random()

class ShapeworldTask():
    def __init__(self):
        self.hint_vocab = util.Index()
        self.START = START
        self.STOP = STOP

        #data = {}
        #for fold in ("train", "val", "test"):
        #    images = []
        #    for i_image in range(100):
        #        image = Image.open(os.path.join(sw_path, fold, "world-%d.bmp" % i_image))
        #        im_data = np.asarray(image)
        #        images.append(im_data)
        #    hints = []
        #    with open(os.path.join(sw_path, fold, "caption.txt")) as caption_f:
        #        for line in caption_f:
        #            hint = line.strip().split()
        #            hint = [self.hint_vocab.index(w) for w in line]
        #            hints.append(tuple(hint))
        #    labels = []
        #    with open(os.path.join(sw_path, fold, "agreement.txt")) as label_f:
        #        for line in label_f:
        #            label = int(float(line))
        #            labels.append(label)

        #    fold_data = []
        #    for image, hint, label in zip(images, hints, labels):
        #        fold_data.append(Datum(hint, [image], image, label))

        #    data[fold] = fold_data


        #data = self.dataset.generate(1, mode="train")
        #_, self.width, self.height, self.channels = data["world"].shape

        #self.sample_batch("train", 100)
        #exit()

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
            fold_data = []
            for i in range(len(hints)):
                fold_data.append(Datum(
                    hints[i], examples[i, ...], inputs[i, ...], labels[i]))
            data[fold] = fold_data

        self.train_data = data["train"]
        self.val_data = data["validation"]
        self.test_data = data["test"]

        self.width, self.height, self.channels = self.train_data[0].input.shape

    def sample_train(self, n_batch):
        batch = []
        n_train = len(self.train_data)
        for _ in range(n_batch):
            datum = self.train_data[random.randint(n_train)]
            batch.append(datum)
        return batch

    def sample_val(self):
        return self.val_data

    def sample_test(self):
        return self.test_data
