from misc import util

from collections import namedtuple
import csv
import numpy as np
import os
import pickle
import sys

N_EX = 4

Datum = namedtuple("Datum", ["hint", "ex_inputs", "input", "label"])

START = "<s>"
STOP = "</s>"

random = util.next_random()

birds_path = os.path.join(sys.path[0], "data/birds")

def choose_except(options, reject, random=random):
    choice = None
    while choice is None:
        choice = random.choice(options)
        if choice in reject:
            choice = None
    return choice

class BirdsTask():
    def __init__(self):
        self.hint_vocab = util.Index()
        self.START = START
        self.STOP = STOP

        with open(os.path.join(birds_path, "hendricks_data", "CUB_feature_dict.pkl")) as feat_f:
            self.features = pickle.load(feat_f)
            #file_to_full = {k.split("/")[1]: k for k in self.features}

        #self.captions = {}
        #for fname in os.listdir(os.path.join(birds_path, "captions")):
        #    name = file_to_full[fname[:-4] + ".jpg"]
        #    inst_capts = []
        #    with open(os.path.join(birds_path, "captions", fname)) as capt_f:
        #        for line in capt_f:
        #            line = line.strip().replace(".", " .").replace(",", " ,")
        #            toks = [START] + line.split() + [STOP]
        #            toks = [self.hint_vocab.index(w) for w in toks]
        #            inst_capts.append(tuple(toks))
        #    self.captions[name] = tuple(inst_capts)
        self.captions = {}
        with open(os.path.join(birds_path, "hendricks_data", "captions.tsv")) as capt_f:
            reader = csv.DictReader(capt_f, delimiter="\t")
            for row in reader:
                caption = row["Description"].lower().replace(".", " .").replace(",", " ,")
                toks = [START] + caption.split() + [STOP]
                toks = [self.hint_vocab.index(w) for w in toks]
                url = row["Input.image_url"]
                inst = "/".join(url.split("/")[-2:])
                if inst not in self.captions:
                    self.captions[inst] = []
                self.captions[inst].append(toks)

        classes = sorted(list(set(k.split("/")[0] for k in self.captions)))
        classes.remove("cub_missing")
        shuf_random = np.random.RandomState(999)
        shuf_random.shuffle(classes)
        assert len(classes) == 200
        data_classes = {
            "train": classes[:100],
            "val": classes[100:110],
            "test": classes[100:200]
        }

        data_insts = {}
        for fold in ("train", "val", "test"):
            classes = data_classes[fold]
            data_classes[fold] = classes

            instances = {cls: [] for cls in classes}
            for key in self.features.keys():
                cls, inst = key.split("/")
                if cls in instances:
                    instances[cls].append(key)
            data_insts[fold] = instances

        #    print fold
        #    for cls in classes:
        #        print cls, len(instances[cls])
        #    print
        #exit()

        self.train_classes = data_classes["train"]
        self.val_classes = data_classes["val"]
        self.test_classes = data_classes["test"]

        self.train_insts = data_insts["train"]
        self.val_insts = data_insts["val"]
        self.test_insts = data_insts["test"]

        self.n_features = self.features[self.features.keys()[0]].size

    def sample_train(self, n_batch, augment):
        assert not augment

        batch = []
        for _ in range(n_batch):
            cls = random.choice(self.train_classes)
            insts = [random.choice(self.train_insts[cls]) for _ in range(N_EX)]
            captions = self.captions[insts[0]]
            caption = captions[random.choice(len(captions))]
            feats = np.asarray([self.features[inst] for inst in insts])

            label = random.randint(2)
            if label == 0:
                other_cls = choose_except(self.train_classes, [cls])
                other_inst = random.choice(self.train_insts[other_cls])
            else:
                other_inst = choose_except(self.train_insts[cls], insts)

            other_feats = self.features[other_inst]
            datum = Datum(caption, feats, other_feats, label)
            batch.append(datum)
        return batch

    def sample_heldout(self, classes, insts):
        batch = []
        local_random = np.random.RandomState(0)
        for i, cls in enumerate(classes):
            datum_insts = insts[cls][:N_EX]
            caption = self.captions[datum_insts[0]][0]
            feats = np.asarray([self.features[inst] for inst in datum_insts])
            label = i % 2
            if label == 0:
                other_cls = choose_except(classes, [cls], local_random)
                other_inst = insts[other_cls][N_EX]
            else:
                other_inst = insts[cls][N_EX]

            other_feats = self.features[other_inst]
            datum = Datum(caption, feats, other_feats, label)
            batch.append(datum)
        return batch

    def sample_val(self, same=False):
        return self.sample_heldout(self.val_classes, self.val_insts)

    def sample_test(self, same=False):
        return self.sample_heldout(self.test_classes, self.test_insts)
