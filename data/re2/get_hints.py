#!/usr/bin/env python2

from collections import defaultdict
import csv
import json
from nltk.tokenize import word_tokenize
import os
import re

templates = defaultdict(list)
hints = {}

exclude = ["'", '"', "`", ".", "\\"]

template_to_pat = defaultdict(set)

with open("data.json") as data_f:
    data = json.load(data_f)

for name in os.listdir("turk"):
    path = os.path.join("turk", name)
    with open(path) as turk_f:
        reader = csv.DictReader(turk_f)
        for row in reader:
            datum_id = int(row["Input.id"])
            hint = row["Answer.hint_text"]
            words = word_tokenize(hint)
            hint = " ".join(words)
            hint = hint.lower()
            for e in exclude:
                hint = hint.replace(e, "")
            hint = hint.replace("-", " - ")
            hint = hint.replace("+", " + ")
            hint = hint.replace("/", " / ")
            hint = re.sub(r"\s+", " ", hint)

            datum = data[datum_id]
            special_b = datum["before"].split(")(")[1]
            special_b = special_b.replace(".", "").replace("[^aeiou]", "").replace("[aeiou]", "")
            special_a = datum["after"][2:-2]
            special_a = special_a.replace("\\2", "")

            a_exploded = " ".join(special_a)
            b_exploded = " ".join(special_b)

            pattern_before = datum["before"].replace("[aeiou]", "V").replace("[^aeiou]", "C")
            pattern_before = re.sub("[a-z]", "l", pattern_before)
            pattern_after = datum["after"][2:-2]
            pattern_after = re.sub("[a-z]+", "l", pattern_after)

            template = hint

            if special_a:
                template = re.sub(r"\b%s\b" % special_a, "AFTER", template)
                template = re.sub(r"\b%s\b" % a_exploded, "AFTER", template)
                hint = re.sub(r"\b%s\b" % special_a, a_exploded, hint)
            if special_b:
                template = re.sub(r"\b%s\b" % special_b, "BEFORE", template)
                template = re.sub(r"\b%s\b" % b_exploded, "BEFORE", template)
                hint = re.sub(r"\b%s\b" % special_b, b_exploded, hint)

            assert datum_id not in hints
            hints[datum_id] = hint

            if re.search(r"\b[b-r,t-z]\b", template):
                # bad template
                continue

            pattern = pattern_before + "@" + pattern_after
            templates[pattern].append(template)
            template_to_pat[template].add(pattern)

            #print (pattern_before, pattern_after), hint

with open("hints.json", "w") as hint_f:
    json.dump(hints, hint_f)

with open("templates.json", "w") as template_f:
    json.dump(templates, template_f)

#for template, pats in template_to_pat.items():
#    print len(pats), template, pats
