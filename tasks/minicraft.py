from misc import util

import numpy as np

START = "<s>"
STOP = "</s>"

KINDS = 10
ROWS = 10
COLS = 10
ACTS = 6

VOCAB = util.Index()
CRAFTS = dict()
HINTS = dict()
for c1 in range(1, KINDS):
    for c2 in range(c1+1, KINDS):
        thing1 = "obj%d" % c1
        thing2 = "obj%d" % c2
        v1 = VOCAB.index(thing1)
        v2 = VOCAB.index(thing2)
        assert v1 == c1
        assert v2 == c2
        craft = len(CRAFTS)
        CRAFTS[v1, v2] = craft

for comps, craft in CRAFTS.items():
    HINTS[craft] = (VOCAB.index(START),) + comps + (VOCAB.index(STOP),)

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
        world = self.world.copy()
        while pos is None:
            r, c = random.randint(ROWS), random.randint(COLS)
            if world[r, c, :].any():
                continue
            pos = r, c
        orientation = random.randint(4)
        inventory = np.zeros((KINDS,))
        return CraftState(pos, orientation, inventory, world, self)

class CraftState(object):
    def __init__(self, position, orientation, inventory, world, datum):
        self.position = position
        self.orientation = orientation
        self.inventory = inventory
        self.world = world
        self.datum = datum
        self.features = self._make_features(orientation, world, inventory)
        self.instruction = self.datum.hint

    def _make_features(self, orientation, world, inventory):
        r, c = self.position
        padded = np.zeros((2*ROWS-1, 2*COLS-1, KINDS))
        sr = ROWS - r - 1
        sc = COLS - c - 1
        assert 0 <= sr < 2 * ROWS - 1
        assert 0 <= sc < 2 * COLS - 1
        padded[sr:sr+ROWS, sc:sc+COLS, :] = world
        ofeats = np.zeros(4)
        ofeats[orientation] = 1
        return np.concatenate((padded.ravel(), inventory, ofeats))

    def step(self, action):
        assert action < ACTS
        r, c = self.position
        o = self.orientation
        new_inventory = self.inventory
        new_world = self.world
        reward = 0
        stop = False

        if action < GRAB:
            new_r, new_c = neighbor((r, c), action)
            new_r = min(max(new_r, 0), ROWS-1)
            new_c = min(max(new_c, 0), COLS-1)
            if new_world[new_r, new_c, :].any():
                new_r, new_c = r, c
            r, c = new_r, new_c
            o = action
        elif action == GRAB:
            nr, nc = neighbor((r, c), o)
            nr = min(max(nr, 0), ROWS-1)
            nc = min(max(nc, 0), COLS-1)
            assert self.world[nr, nc, :].sum() <= 1
            if self.world[nr, nc, :].sum() == 1:
                kind = np.argmax(self.world[nr, nc, :])
                new_inventory = self.inventory.copy()
                new_inventory[kind] += 1
                new_world = self.world.copy()
                new_world[nr, nc, :] = 0
        elif action == CRAFT:
            if self.inventory.sum() >= 2:
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

        new_state = CraftState((r, c), o, new_inventory, new_world, self.datum)
        return new_state, reward, stop

class CraftTask(object):
    def __init__(self):
        goals = list(CRAFTS.values())
        assert len(goals) == 36
        train_goals = goals[:24]
        test_goals = goals[24:]

        folds = dict()
        for fold, goals in (("train", train_goals), ("test", test_goals)):
            fold_data = []
            for goal in goals:
                hint = HINTS[goal]
                for _ in range(20):
                    world = self._sample_world()
                    datum = CraftDatum(world, goal, hint, self)
                    fold_data.append(datum)
            folds[fold] = fold_data

        self.train_data = folds["train"]
        self.test_data = folds["test"]

        #self.feature_shape = self.train_data[0].features.shape
        self.n_features = self.train_data[0].init().features.size
        self.n_actions = ACTS
        self.vocab = VOCAB

    def _sample_world(self):
        world = np.zeros((ROWS, COLS, KINDS))
        for kind in range(1, KINDS):
            while True:
                r, c = random.randint(ROWS), random.randint(COLS)
                if world[r, c, :].any():
                    continue
                pos = r, c
                break
            world[r, c, kind] = 1
        return world

    def sample_train(self):
        data = self.train_data
        return data[random.choice(len(data))].init()

    def sample_test(self):
        data = self.test_data
        return data[random.choice(len(data))].init()
