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

def generate(mode, n):
    print()
    print(mode)
    examples = np.zeros((n, EXAMPLES, WIDTH, HEIGHT, CHANNELS))
    example_structs = []
    inputs = np.zeros((n, WIDTH, HEIGHT, CHANNELS))
    input_structs = []
    labels = np.zeros((n,))
    hints = []

    i = 0
    running = 0
    while i < n:
        inp_world = DATASET.world_generator(mode)
        caption = DATASET.world_captioner(
                world=inp_world.model(), correct=bool(random.randint(2)), mode=mode)
        if not caption:
            continue
        #assert caption.agreement(world=model) == 1.0

        ex = []
        ex_struct = []
        #for _ in range(ATTEMPTS):
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
        #if len(ex) < EXAMPLES:
        #    continue

        #inp = DATASET.world_generator(mode)
        #label = int(caption.agreement(world=inp.model()))
        label = caption.agreement(world=inp_world.model())

        caption_text, = DATASET.caption_realizer.realize(captions=[caption])
        caption_text = " ".join(caption_text)

        print(inp_world.get_array())
        print(inp_world.get_array().max())
        print(inp_world.get_array().shape)
        exit()

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

    np.save(os.path.join(mode, "examples.npy"), examples)
    np.save(os.path.join(mode, "inputs.npy"), inputs)
    np.save(os.path.join(mode, "labels.npy"), labels)
    with open(os.path.join(mode, "hints.json"), "w") as hint_f:
        json.dump(hints, hint_f)
    with open(os.path.join(mode, "examples.struct.json"), "w") as ex_f:
        json.dump(example_structs, ex_f)
    with open(os.path.join(mode, "inputs.struct.json"), "w") as inp_f:
        json.dump(input_structs, inp_f)

generate("train", 10000)
generate("validation", 1000)
generate("test", 1000)

#generate("train", 20)
