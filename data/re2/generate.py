#!/usr/bin/env python2

import json
import numpy as np
import re

N_VAL = 500
N_TEST = 500

random = np.random.RandomState(0)

word_re = re.compile("^[a-z]+$")
words = []
with open("/usr/share/dict/words") as words_f:
    for line in words_f:
        word = line.strip()
        if word_re.match(word):
            words.append(word)

chars = "."
vowels = "[aeiou]"
consonants = "[^aeiou]"
letters = [chr(i) for i in range(ord("a"), ord("z"))]
classes = [chars, vowels, consonants]

N_EX = 30

def sample_block(size):
    out = ""
    for i in range(size):
        use_letter = random.randint(2)
        if use_letter:
            out += random.choice(letters)
        else:
            out += random.choice(classes)
    return out

def sample_replace(size):
    out = ""
    for i in range(size):
        use_letter = random.randint(3)
        if use_letter:
            out += random.choice(letters)
        else:
            out += "\\2"
    return out

def sample():
    anchor_start = random.randint(2)
    anchor_end = not anchor_start and random.randint(2)

    start_len = random.randint(1)
    match_len = random.randint(2) + 1
    rep_len = random.randint(3)
    end_len = random.randint(1)

    start = "^" if anchor_start else ""
    start += sample_block(start_len)
    
    match = sample_block(match_len)
    replace = sample_replace(rep_len)

    end = sample_block(end_len)
    if anchor_end:
        end += "$"

    before = "(%s)(%s)(%s)" % (start, match, end)
    after = "\\1%s\\3" % replace

    return before, after

seen = set()
data = []
while len(data) < 5000:
    before, after = sample()
    if (before, after) in seen:
        continue
    seen.add((before, after))
    matches = []
    order = list(range(len(words)))
    random.shuffle(order)
    attempts = 0
    success = True
    while len(matches) < N_EX:
        attempts += 1
        if attempts > 100:
            success = False
            break
        j = order.pop()
        word = words[j]
        sub = re.sub(before, after, word)
        if sub == word and len(matches) < N_EX / 2:
            continue
        #if sub == word:
        #    continue
        matches.append((word, sub))

    random.shuffle(matches)

    if not success:
        continue

    print len(data)

    data.append({"before": before, "after": after, "examples": matches})

    #print before, after
    #for match in matches:
    #    print " ", match

random.shuffle(data)

#train_data = data[:-(N_VAL+N_TEST)]
#val_data = data[-(N_VAL+N_TEST):-N_TEST]
#test_data = data[-N_TEST:]
#
#dataset = {
#    "train": train_data,
#    "val": val_data,
#    "test": test_data
#}

dataset = data

with open("data.json", "w") as data_f:
    json.dump(dataset, data_f)
