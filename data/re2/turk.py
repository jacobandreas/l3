#!/usr/bin/env python2

import json
import re

N_EX = 8

with open("data.json") as data_f:
    data = json.load(data_f)

train_data = data["train"]
#label_data = train_data[:10]
#label_data = train_data[:1000]
label_data = train_data

with open("turk_data_3.csv", "w") as turk_f:
    header_labels = ["id"]
    header_labels += ["before_%d" % i for i in range(N_EX)]
    header_labels += ["after_%d" % i for i in range(N_EX)]
    header_labels += ["test_before", "test_after"]
    header_labels += ["hint"]
    print >>turk_f, ",".join(header_labels)

    #for i, datum in enumerate(label_data):
    for i in range(2000, 3000):
        datum = label_data[i]
        ref_before, ref_after = zip(*datum["examples"][:N_EX])
        test_before, test_after = datum["examples"][N_EX]
        hints = []
        pat = datum["before"]
        sub = datum["after"]
        before = [re.sub(pat, "<span class='from'>\\1\\2\\3</span>", b) for b in ref_before]
        after = [re.sub(pat, "<span class='to'>" + sub + "</span>", b) for b in ref_before]
        after_check = [re.sub(pat, sub, b) for b in ref_before]
        assert after_check == list(ref_after)
        if "(^)" in pat:
            hints.append("look at the <b>beginning</b> of the word")
        if "$" in pat:
            hints.append("look at the <b>end</b> of the word")
        if "[aeiou]" in pat:
            hints.append("look at <b>vowels</b>")
        if "[^aeiou]" in pat:
            hints.append("look at <b>consonants</b>")
        if "\\2\\2" in sub:
            hints.append("look for <b>repitition</b>")
        if len(hints) > 0:
            hint = "</li><li>".join(hints)
        else:
            hint = "(no hint for this problem)"

        line = ",".join([str(i)] + before + after + [test_before, test_after, hint])
        print >>turk_f, line
