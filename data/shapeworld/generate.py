#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
import os
from shapeworld import dataset
import json
import itertools

#N_CAPTIONS = 5000
#N_TRAIN = 4000
#N_VAL = 500
#N_TEST = 500

N_CAPTIONS = 3000
N_TRAIN = 2000
N_VAL = 500
N_TEST = 500

#N_CAPTIONS = 100
#N_TRAIN = 50
#N_VAL = 25
#N_TEST = 25

assert N_TRAIN + N_VAL + N_TEST == N_CAPTIONS

WIDTH = 64
HEIGHT = 64
CHANNELS = 3
EXAMPLES = 4

DATASET = dataset(dtype="agreement", name="spatial_jda")
random = np.random.RandomState(0)

all_captions = {}
while len(all_captions) < N_CAPTIONS:
    if len(all_captions) % 500 == 0:
        print("%d / %d captions" % (len(all_captions), N_CAPTIONS))

    DATASET.world_generator.sample_values(mode="train")
    DATASET.world_captioner.sample_values(mode="train", correct=True)
    while True:
        world = DATASET.world_generator()
        if world is None:
            continue
        caption = DATASET.world_captioner(entities=world.entities)
        if caption is None:
            continue
        break
    realized, = DATASET.caption_realizer.realize(captions=[caption])
    realized = tuple(realized)
    all_captions[realized] = caption

captions = list(sorted(all_captions.keys()))
random.shuffle(captions)
train_captions = captions[:N_TRAIN]
val_captions = captions[N_TRAIN:N_TRAIN+N_VAL]
test_captions = captions[N_TRAIN+N_VAL:]
caption_data = all_captions

def generate(name, captions, n_examples):
    mappings = defaultdict(list)
    for i_scene in range(n_examples * 20):
    #for _ in range(n_examples * 5):
        DATASET.world_generator.sample_values(mode="train")
        world = DATASET.world_generator()
        if world is None:
            continue
        for key in captions:
            caption = caption_data[key]
            agree = caption.agreement(entities=world.entities) > 0
            #print(agree)
            #print(world.entities)
            if not agree:
                continue
            mappings[key].append(world)

        if i_scene % 1000 == 0:
            print("%d / %d scenes" % (i_scene, n_examples * 20))

    for key in mappings:
        print(" ".join(key), len(mappings[key]))

    total_scenes = sum(len(l) for l in mappings.values())
    assert total_scenes > n_examples * 6

    examples = np.zeros((n_examples, EXAMPLES, WIDTH, HEIGHT, CHANNELS))
    inputs = np.zeros((n_examples, WIDTH, HEIGHT, CHANNELS))
    labels = np.zeros((n_examples,))
    hints = []

    i_example = 0
    while i_example < n_examples:
        key = captions[random.randint(len(captions))]

        worlds = mappings[key]
        if len(worlds) < EXAMPLES + 1:
            continue
        for i_world in range(EXAMPLES):
            world = worlds.pop()
            examples[i_example, i_world, ...] = world.get_array()

        if random.randint(2) == 0:
            world = worlds.pop()
            inputs[i_example, ...] = world.get_array()
            labels[i_example] = 1
        else:
            while True:
                other_key = captions[random.randint(len(captions))]
                if len(mappings[other_key]) > 0:
                    other_world = mappings[other_key].pop()
                    break
            inputs[i_example, ...] = other_world.get_array()
            labels[i_example] = 0
        hints.append(" ".join(key))

        if i_example % 500 == 0:
            print("%d / %d examples" % (i_example, n_examples))

        i_example += 1

    print("\n\n")

    np.save(os.path.join(name, "examples.npy"), examples)
    np.save(os.path.join(name, "inputs.npy"), inputs)
    np.save(os.path.join(name, "labels.npy"), labels)
    with open(os.path.join(name, "hints.json"), "w") as hint_f:
        json.dump(hints, hint_f)

generate("train", train_captions, 9000)
generate("val", val_captions, 500)
generate("test", test_captions, 500)
generate("val_same", train_captions, 500)
generate("test_same", train_captions, 500)

#generate("train", train_captions, 100)
#generate("val", val_captions, 100)
#generate("test", test_captions, 100)
