import util

import numpy as np

START = "<s>"
STOP = "</s>"

KINDS = 9
ROWS = 10
COLS = 10
ACTS = 6

CRAFTS = dict()
for c1 in range(KINDS):
    for c2 in range(c1+1, KINDS):
        CRAFTS[c1, c2] = len(CRAFTS)

NORTH, EAST, SOUTH, WEST, GRAB, CRAFT = range(6)

random = util.next_random()

def neighbor(point, direction):
    r, c = point
    if direction == NORTH:
        r -= 1
    elif direction == EAST:
        c += 1
    elif direction == SOUTH:
        r += 1
    elif direction == WEST:
        c -= 1
    else:
        assert False
    return r, c

class CraftDatum(object):
    def __init__(self, world, goal, hint, task):
        self.world = world
        self.goal = goal
        self.hint = hint
        self.task = task

    def init(self):
        pos = None
        while pos is None:
            r, c = random.randint(ROWS), random.randint(COLS)
            if self.world[r, c, :].any():
                continue
            pos = r, c
            orientation = random.randint(4)
        return CraftState(pos, orientation, self)

class CraftState(object):
    def __init__(self, position, orientation, inventory, datum):
        self.position = position
        self.orientation = orientation
        self.datum = datum

    def _make_features(self, raw):
        r, c = self.position
        padded = np.zeros((2*ROWS-1, 2*COLS-1, OBJS))
        sr = ROWS - r - 1
        sc = COLS - c - 1
        assert 0 <= sr < 2 * ROWS - 1
        assert 0 <= sc < 2 * COLS - 1
        padded[sr:sr+ROWS, sc:sc+COLS, :] = raw
        return padded

    def step(self, action):
        assert action < ACTS
        r, c = self.position
        o = self.orientation
        new_inventory = self.inventory
        reward = 0
        stop = False

        if action < GRAB:
            r, c = neighbor(action)
            o = action
        elif action == GRAB:
            nr, nc = neighbor(o)
            assert self.datum.world[r, c, :].sum() == 1
            kind = np.argmax(self.datum.world[r, c, :])
            new_inventory = self.inventory.copy()
            new_inventory[kind] += 1
        elif action == CRAFT:
            assert self.inventory.max() <= 1
            new_inventory = self.inventory.copy()
            inv_probs1 = new_inventory / new_inventory.sum()
            thing1 = random.choice(KINDS, p=inv_probs1)
            new_inventory[thing1] -= 1
            inv_probs2 = new_inventory / new_inventory.sum()
            thing2 = random.choice(KINDS, p=inv_probs2)
            new_inventory[thing2] -= 1
            thing1, thing2 = sorted([thing1, thing2])
            assert thing1 < thing2
            if CRAFTS[thing1, thing2] == self.datum.goal:
                reward = 1
                stop = True
        else:
            assert False

        return CraftState((r, c), o, new_inventory, self.datum)

class CraftTask(object):
    def __init__(self):
        goals = list(CRAFTS.keys())
        random.shuffle(goals)
        print len(goals)
        exit()
