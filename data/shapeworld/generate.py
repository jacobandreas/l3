import numpy as np
import os
from shapeworld import dataset
import json
import itertools

DATASET = dataset(dtype="agreement", name="multishape")
WIDTH = 64
HEIGHT = 64
CHANNELS = 3
EXAMPLES = 8
ATTEMPTS = 100

random = np.random.RandomState()

all_captions = {}
for _ in range(500):
    inp_world = DATASET.world_generator("train")
    caption = DATASET.world_captioner(
            world=inp_world.model(), correct=bool(random.randint(2)),
            mode="train")
    realized, = DATASET.caption_realizer.realize(captions=[caption])
    realized = tuple(realized)
    all_captions[realized] = caption

captions = list(sorted(all_captions.keys()))
random.shuffle(captions)
train_captions = captions[:-20]
val_captions = captions[-20:-10]
test_captions = captions[-10:]
print(len(train_captions), len(val_captions), len(test_captions))

def generate(name, mode, captions, n):
    print()
    print(name)
    examples = np.zeros((n, EXAMPLES, WIDTH, HEIGHT, CHANNELS))
    example_structs = []
    inputs = np.zeros((n, WIDTH, HEIGHT, CHANNELS))
    input_structs = []
    labels = np.zeros((n,))
    hints = []

    i = 0
    running = 0
    while i < n:
        get_true = random.randint(2)
        capt_key = captions[random.randint(len(captions))]
        caption = all_captions[capt_key]
        success = False
        for _ in range(100 if name == "train" else 100):
            inp_world = DATASET.world_generator(mode)
            agree = caption.agreement(world=inp_world.model())
            if get_true == agree:
                success = True
                break

        ex = []
        ex_struct = []
        while len(ex) < EXAMPLES:
            world = DATASET.world_generator(mode)
            model = world.model()
            if caption.agreement(world=model) != 1.0:
                continue
            ex.append(world.get_array())
            struct = set()
            for entity in world.entities:
                shape = str(entity.shape)
                color = str(entity.color)
                struct.update({(shape,), (color,), (shape, color)})
            ex_struct.append(list(struct))

        label = caption.agreement(world=inp_world.model())
        caption_text, = DATASET.caption_realizer.realize(captions=[caption])
        caption_text = " ".join(caption_text)

        examples[i, ...] = ex
        inputs[i, ...] = inp_world.get_array()
        labels[i] = label
        hints.append(caption_text)

        inp_struct = set()
        for entity in inp_world.entities:
            shape = str(entity.shape)
            color = str(entity.color)
            inp_struct.update({(shape,), (color,), (shape, color)})
        example_structs.append(ex_struct)
        input_structs.append(list(inp_struct))

        running += label

        i += 1
        if i % 10 == 0:
            print(i, running / i)

    np.save(os.path.join(name, "examples.npy"), examples)
    np.save(os.path.join(name, "inputs.npy"), inputs)
    np.save(os.path.join(name, "labels.npy"), labels)
    with open(os.path.join(name, "hints.json"), "w") as hint_f:
        json.dump(hints, hint_f)
    with open(os.path.join(name, "examples.struct.json"), "w") as ex_f:
        json.dump(example_structs, ex_f)
    with open(os.path.join(name, "inputs.struct.json"), "w") as inp_f:
        json.dump(input_structs, inp_f)

generate("train", "train", train_captions, 5000)
generate("validation", "train", val_captions, 500)
generate("test", "train", test_captions, 500)

