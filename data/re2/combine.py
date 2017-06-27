#!/usr/bin/env python2

import json
import re
import numpy as np

START = "<"
STOP = ">"
SEP = "@"

random = np.random.RandomState(0)

N_EX = 5

with open("data.json") as data_f:
    data = json.load(data_f)

with open("templates.json") as template_f:
    templates = json.load(template_f)

with open("hints.json") as hint_f:
    hints = json.load(hint_f)
    hints = {int(k): v for k, v in hints.items()}

annotations = []

for i, example in enumerate(data):
    t_before = example["before"]
    t_before = t_before.replace("[aeiou]", "V").replace("[^aeiou]", "C")
    re_before = t_before
    letters_before = t_before.split(")(")[1].replace(".", "").replace("V", "").replace("C", "")
    letters_before = " ".join(letters_before)
    t_before = re.sub("[a-z]+", "l", t_before)
    t_after = example["after"][2:-2]
    re_after = t_after
    letters_after = t_after.replace("\\2", "")
    letters_after = " ".join(letters_after)
    t_after = re.sub("[a-z]+", "l", t_after)
    template_key = t_before + SEP + t_after

    if template_key not in templates:
        continue

    aug_hints = []
    for template in templates[template_key]:
        aug_hint = template.replace("BEFORE", letters_before).replace("AFTER", letters_after)
        aug_hint = [START] + aug_hint.split() + [STOP]
        aug_hints.append(aug_hint)

    if i in hints:
        hint = hints[i]
    else:
        hint = ""
    hint = [START] + hint.split() + [STOP]

    re_hint = START + re_before + SEP + re_after + STOP

    ex = []
    for inp, out in example["examples"]:
        inp = START + inp + STOP
        out = START + out + STOP
        ex.append((inp, out))

    annotations.append({
        "examples": ex,
        "re": re_hint,
        "hint": hint,
        "hints_aug": aug_hints
    })

train = annotations[:3000]
val = annotations[3000:3500]
test = annotations[3500:4000]

for datum in val:
    del datum["examples"][N_EX+1:]

for datum in test:
    del datum["examples"][N_EX+1:]

corpus = {
    "train": train,
    "val": val,
    "test": test
}

with open("corpus.json", "w") as corpus_f:
    json.dump(corpus, corpus_f)
