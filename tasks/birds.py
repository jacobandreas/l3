import util

from collections import defaultdict, namedtuple
import csv
import pickle

images = {}
rev_images = {}
with open("data/birds/CUB_200_2011/images.txt") as image_f:
    for line in image_f:
        image_id, image_path = line.strip().split()
        image_id = int(image_id)
        images[image_id] = image_path
        rev_images[image_path] = image_id


#train_ids = set()
#test_ids = set()
#with open("data/birds/CUB_200_2011/train_test_split.txt") as split_f:
#    for line in split_f:
#        image_id, is_train = (int(i) for i in line.strip().split())
#        if is_train:
#            train_ids.add(image_id)
#        else:
#            test_ids.add(image_id)
#train_ids = sorted(list(train_ids))
#test_ids = sorted(list(test_ids))

labels = {}
with open("data/birds/CUB_200_2011/image_class_labels.txt") as label_f:
    for line in label_f:
        image_id, image_label = (int(i) for i in line.strip().split())
        labels[image_id] = image_label

train_labels = list(range(100))
val_labels = list(range(100, 150))
test_labels = list(range(150, 200))
train_ids = []
val_ids = []
test_ids = []
for image_id in images:
    label = labels[image_id]
    if label in train_labels:
        train_ids.append(image_id)
    elif label in val_labels:
        val_ids.append(image_id)
    elif label in test_labels:
        test_ids.append(image_id)

features = {}
with open("data/birds/labels.p", "rb") as feature_f:
    orig_features = pickle.load(feature_f)
    for image_id, image_path in images.items():
        features[image_id] = orig_features[image_path]

vocab = util.Index()
hints = defaultdict(list)
with open("data/birds/cub_0917_5cap.tsv") as ann_f:
    reader = csv.reader(ann_f, delimiter="\t")
    header = reader.next()
    i_url = header.index("Input.image_url")
    i_desc = header.index("Answer.Description")
    for line in reader:
        url = line[i_url]
        desc = line[i_desc]
        key = "/".join(url.split("/")[-2:])
        desc = desc.lower().replace(".", "").replace(",", "")
        desc = "<s> " + desc + " </s>"
        if key not in rev_images:
            continue
        image_id = rev_images[key]
        hint = tuple(vocab.index(w) for w in desc.split())
        hints[image_id].append(hint)

for image_id in images:
    assert image_id in hints

BirdsDatum = namedtuple("BirdsDatum", ["features", "label", "hint"])
random = util.next_random()

def sample_train():
    out = []
    for _ in range(100):
        chosen_id = random.choice(train_ids)
        feats = features[chosen_id]
        label = labels[chosen_id]
        hint_options = hints[chosen_id]
        hint = hint_options[random.randint(len(hint_options))]
        out.append(BirdsDatum(feats, label, hint))
    return out

###

use_ids = val_ids
use_labels = val_labels
by_label = [(l, sorted([i for i in use_ids if labels[i] == l])) for l in use_labels]
exemplars = [(l, i) for l, ii in by_label for i in ii]
heldout = [i for l, ii in by_label for i in ii]
#exemplars = [(l, i) for l, ii in by_label for i in ii[:10]]
#heldout = [i for l, ii in by_label for i in ii[10:]]
n_features = features.values()[0].size

def sample_test_exemplars():
    out = []
    for l, i in exemplars:
        for hint in hints[i]:
            out.append(BirdsDatum(features[i], l, hint))
    return out

def sample_test_data():
    out = []
    for i in heldout:
        out.append(BirdsDatum(features[i], labels[i], None))
    return out
