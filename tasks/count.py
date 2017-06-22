import util

import cairocffi as cairo
import numpy as np

WIDTH = 30
HEIGHT = 30

TOTAL_AREA = 0.2 * 15 ** 2

MAX_COUNT = 10

random = util.next_random()
counter = [0]

def sample():
    count = random.randint(MAX_COUNT)

    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    surf = cairo.ImageSurface.create_for_data(data.data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgba(0., 0., 0., 1.)
    ctx.paint()

    if count == 0:
        return

    max_radius = max(int(np.sqrt(TOTAL_AREA / count)), 4)
    min_radius = max(3, max_radius / 2)

    for _ in range(count):
        radius = random.randint(min_radius, max_radius)
        while True:
            x = radius + random.randint(WIDTH - 2 * radius)
            y = radius + random.randint(HEIGHT - 2 * radius)
            if data[y, x, 0] > 0:
                continue
            center = (x, y)
            break
        cx, cy = center
        ctx.set_source_rgba(1., 1., 1., 0.01)
        ctx.arc(cx, cy, radius*2, 0, 2*np.pi)
        ctx.fill()
        ctx.set_source_rgba(1., 1., 1., 1.)
        ctx.arc(cx, cy, radius, 0, 2*np.pi)
        ctx.fill()

    img = data[:, :, 0] / 255.
    if random.rand() < 0.5:
        label, acc = count, True
    else:
        label = random.randint(MAX_COUNT)
        acc = label == count

    return img, acc, label

    #surf.write_to_png("%d.png" % counter[0])
    #counter[0] += 1
