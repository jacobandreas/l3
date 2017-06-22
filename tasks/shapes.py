import util

import cairocffi as cairo
from collections import namedtuple
import numpy as np

WIDTH = 30
HEIGHT = 30

random = util.next_random()

N_SHAPES = 3
SHAPES = [
    [(5, 15), (25, 5), (25, 25)],
    [(5, 5), (5, 25), (25, 25), (25, 5)],
    [(5, 10), (5, 20), (15, 25), (25, 20), (25, 10), (15, 5)]
]

N_COLORS = 3
COLORS = [
    (1., 0., 0., 1),
    (0., 1., 0., 1),
    (0., 0., 1., 1)
]

N_NDOTS = 3
NDOTS = [1, 2, 3]

N_ATTRS = 3
N_ATTR_COUNTS = [N_SHAPES, N_COLORS, N_NDOTS]

ATTR_NAMES = ["shape", "color", "ndots"]
VAL_NAMES = [
    ["triangle", "square", "hexagon"],
    ["red", "green", "blue"],
    ["one", "two", "three"]
]

counter = [0]

vocab = util.Index()
for attr in range(N_ATTRS):
    for val in range(N_ATTR_COUNTS[attr]):
        vocab.index((attr, val))
for name in ["and", "or"]:
    vocab.index(name)
#n_vocab = len(vocab)
n_vocab = 100

ClassifyDatum = namedtuple("ClassifyDatum", ["image", "concept", "label", "hint"])

image_shape = (WIDTH, HEIGHT, 3)
n_concepts = N_SHAPES + N_COLORS + N_NDOTS

def sample_primitive_concept():
    attr = random.randint(N_ATTRS)
    val = random.randint(N_ATTR_COUNTS[attr]-1)
    #return "%s:%s" % (ATTR_NAMES[attr], VAL_NAMES[attr][val])
    return (attr, val)

def sample_concept():
    kind = random.randint(3)
    if kind == 0:
        return sample_primitive_concept()
    if kind == 1:
        concept1 = sample_primitive_concept()
        concept2 = concept1
        while concept2 == concept1:
            concept2 = sample_primitive_concept()
        return ("or", concept1, concept2)
    if kind == 2:
        concept1 = sample_primitive_concept()
        concept2 = concept1
        while concept2[0] == concept1[0]:
            concept2 = sample_primitive_concept()
        return ("and", concept1, concept2)
    assert False

def validate(concept, attrs):
    if concept[0] == "and":
        return all(validate(child, attrs) for child in concept[1:])
    if concept[0] == "or":
        return any(validate(child, attrs) for child in concept[1:])
    return attrs[concept[0]] == concept[1]

concepts = sorted(list(set(sample_concept() for _ in range(10000))))
random.shuffle(concepts)
assert len(concepts) == 60
n_train = 30
train_concepts = concepts[:n_train]
test_concepts = concepts[n_train:]

def sample(test=False):
    #concept = sample_concept()
    if test:
        concept = test_concepts[random.randint(len(test_concepts))]
    else:
        concept = train_concepts[random.randint(len(train_concepts))]
    acc = bool(random.randint(2))
    while True:
        shape = random.randint(N_SHAPES)
        color = random.randint(N_COLORS)
        ndots = random.randint(N_NDOTS)
        attrs = [shape, color, ndots]
        if validate(concept, attrs) == acc:
            break

    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    surf = cairo.ImageSurface.create_for_data(data.data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgba(1., 1., 1., 1.)
    ctx.paint()

    ctx.set_source_rgba(*COLORS[color])
    points = SHAPES[shape]
    ctx.move_to(*points[-1])
    for point in points:
        ctx.line_to(*point)
    ctx.fill()

    dot_corners = list(SHAPES[shape])
    random.shuffle(dot_corners)
    dot_corners = dot_corners[:NDOTS[ndots]]
    for cx, cy in dot_corners:
        ctx.arc(cx, cy, 2, 0, 2*np.pi)
        ctx.fill()

    img = data[:, :, :3] / 255

    #label_attr = random.randint(N_ATTRS)
    #if random.rand() < 0.5: # positive example
    #    label = (label_attr, attrs[label_attr])
    #    acc = True
    #else: # negative example
    #    val = random.randint(N_ATTR_COUNTS[label_attr]-1)
    #    if val >= attrs[label_attr]:
    #        val += 1
    #    assert val != attrs[label_attr]
    #    label = (label_attr, val)
    #    acc = False

    if True or test:
        hint = (vocab.index(concept),)
    else:
        if isinstance(concept[0], str):
            hint = tuple(vocab[w] for w in concept)
        else:
            hint = (vocab[concept],)
    assert None not in hint
    return ClassifyDatum(img, concept, acc, hint)
