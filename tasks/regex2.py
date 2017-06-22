import util

from collections import namedtuple
import json
import re

N_VAL = 200
N_TEST = 200
START = "<"
STOP = ">"
SEP = "@"

random = util.next_random()

hint_vocab = util.Index()
str_vocab = util.Index()
str_vocab.index(SEP)

#Dataset = namedtuple("Dataset", ["data", "hint_vocab", "str_vocab"])
Datum = namedtuple("Datum", ["hints_", "hint", "inputs", "outputs"])

with open("data/re2/data.json") as data_f:
    annotations = json.load(data_f)

with open("data/re2/templates.json") as template_f:
    templates = json.load(template_f)

patterns_1k = set()
patterns_all = set()

data = {}
for fold in ["train", "val", "test"]:
    data[fold] = []
    for i, example in enumerate(annotations[fold]):
        #before = example["before"].replace("[aeiou]", "V").replace("[^aeiou]", "C").replace("\\1", "").replace("\\3", "")
        #after = example["after"].replace("[aeiou]", "V").replace("[^aeiou]", "C").replace("\\1", "").replace("\\3", "")

        ##re = example["before"] + SEP + example["after"]
        #re = START + before + SEP + after + STOP
        #hint = [hint_vocab.index(c) for c in re]
        if fold == "train":
            t_before = example["before"]
            t_before = t_before.replace("[aeiou]", "V").replace("[^aeiou]", "C")
            letters_before = t_before.split(")(")[1].replace(".", "").replace("V", "").replace("C", "")
            letters_before = " ".join(letters_before)
            t_before = re.sub("[a-z]+", "l", t_before)
            t_after = example["after"][2:-2]
            letters_after = t_after.replace("\\2", "")
            letters_after = " ".join(letters_after)
            t_after = re.sub("[a-z]+", "l", t_after)
            template_key = t_before + "@" + t_after
            patterns_all.add(template_key)
            if i < 1000:
                patterns_1k.add(template_key)
            if template_key not in templates:
                continue
                #if template_key not in unknown:
                #    print "unknown:", template_key
                #template = ""

            #template = random.choice(templates[template_key])
            hints = []
            for template in templates[template_key]:
                hint = template.replace("BEFORE", letters_before).replace("AFTER", letters_after)
                hint = [START] + hint.split() + [STOP]
                hints.append(hint)


            #if i in hints:
            #    hint = [START] + hints[i].split() + [STOP]
            #else:
            #    continue
        else:
            hints = [[]]
        hints = [[hint_vocab.index(c) for c in hint] for hint in hints]

        inputs = []
        outputs = []
        for inp, out in example["examples"]:
            inp = START + inp + STOP
            out = START + out + STOP
            inp_indexed = [str_vocab.index(c) for c in inp]
            out_indexed = [str_vocab.index(c) for c in out]
            inputs.append(inp_indexed)
            outputs.append(out_indexed)

        data[fold].append(Datum(hints, None, inputs, outputs))
    print fold, len(data[fold])

train_data = data["train"]
val_data = data["val"]
test_data = data["test"]

#train_data = train_data[:1000]
train_data = data["train"][:3000]
val_data = data["train"][3000:3500]
test_data = data["train"][3500:]

def sample_train():
    batch = []
    for _ in range(100):
        datum = train_data[random.randint(len(train_data))]
        pairs = list(zip(datum.inputs, datum.outputs))
        random.shuffle(pairs)
        new_inputs, new_outputs = zip(*pairs)
        datum = datum._replace(inputs=new_inputs, outputs=new_outputs)
        hint = datum.hints_[random.randint(len(datum.hints_))]
        datum = datum._replace(hint=hint, hints_=None)
        batch.append(datum)
    return batch

def sample_test():
    batch = []
    for datum in val_data:
        pairs = list(zip(datum.inputs, datum.outputs))
        #random.shuffle(pairs)
        new_inputs, new_outputs = zip(*pairs)
        datum = datum._replace(inputs=new_inputs, outputs=new_outputs)

        #datum = datum._replace(hint=[])
        hint = datum.hints_[0]
        datum = datum._replace(hint=hint, hints_=None)

        batch.append(datum)
    return batch
