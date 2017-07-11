import numpy as np
import os
from shapeworld import dataset
import json

DATASET = dataset(dtype="agreement", name="multishape")
WIDTH = 64
HEIGHT = 64
CHANNELS = 3
EXAMPLES = 3
ATTEMPTS = 100

def generate(mode, n):
    print(mode)
    examples = np.zeros((n, EXAMPLES, WIDTH, HEIGHT, CHANNELS))
    inputs = np.zeros((n, WIDTH, HEIGHT, CHANNELS))
    labels = np.zeros((n,))
    hints = []

    i = 0
    while i < n:
        init_world = DATASET.world_generator(mode)
        model = init_world.model()
        caption = DATASET.world_captioner(world=model, correct=True, mode=mode)
        if not caption:
            continue
        assert caption.agreement(world=model) == 1.0

        ex = [init_world.get_array()]
        #for _ in range(ATTEMPTS):
        while len(ex) < EXAMPLES:
            world = DATASET.world_generator(mode)
            model = world.model()
            if caption.agreement(world=model) != 1.0:
                continue
            ex.append(world.get_array())
            if len(ex) == EXAMPLES:
                break
        #if len(ex) < EXAMPLES:
        #    continue

        inp = DATASET.world_generator(mode)
        label = int(caption.agreement(world=inp.model()))

        caption_text, = DATASET.caption_realizer.realize(captions=[caption])
        caption_text = " ".join(caption_text)

        examples[i, ...] = ex
        inputs[i, ...] = inp.get_array()
        labels[i] = label
        hints.append(caption_text)

        i += 1
        if i % 10 == 0:
            print(i)

    np.save(os.path.join(mode, "examples.npy"), examples)
    np.save(os.path.join(mode, "inputs.npy"), inputs)
    np.save(os.path.join(mode, "labels.npy"), labels)
    with open(os.path.join(mode, "hints.json"), "w") as hint_f:
        json.dump(hints, hint_f)

generate("train", 10000)
generate("validation", 1000)
generate("test", 1000)
