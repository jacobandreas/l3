from misc import util
from misc import array

from collections import namedtuple
import numpy as np
import pickle
import re
from skimage.measure import block_reduce

SIZE = 7
WINDOW_SIZE = 5

#WOOD_VARIANTS = ["oak", "pine", "birch"]
#ORE_VARIANTS = ["copper", "iron", "nickel"]
#STONE_VARIANTS = ["granite", "quartz", "slate"]
WOOD_VARIANTS = ["oak", "pine"]
ORE_VARIANTS = ["copper", "iron"]
STONE_VARIANTS = ["granite", "quartz"]
#WOOD_VARIANTS = ["oak"]
#ORE_VARIANTS = ["copper"]
#STONE_VARIANTS = ["granite"]

INGREDIENTS = (
    ["wood_%s" % v for v in WOOD_VARIANTS]
    + ["ore_%s" % v for v in ORE_VARIANTS]
    + ["grass"]
    + ["stone_%s" % v for v in STONE_VARIANTS]
)

CRAFTS = {
    #"stick_?": ["stick_%s" % v for v in WOOD_VARIANTS],
    #"metal_?": ["metal_%s" % v for v in ORE_VARIANTS],
    #"rope": ["rope"],
    "shovel_?_?": ["shovel_%s_%s" % (v1, v2) for v1 in WOOD_VARIANTS for v2 in ORE_VARIANTS],
    "ladder_?": ["ladder_%s" % v for v in WOOD_VARIANTS],
    "axe_?_?": ["axe_%s_%s" % (v1, v2) for v1 in WOOD_VARIANTS for v2 in STONE_VARIANTS],
    "trap_?": ["trap_%s" % v for v in ORE_VARIANTS],
    "sword_?_?": ["sword_%s_%s" % (v1, v2) for v1 in ORE_VARIANTS for v2 in STONE_VARIANTS],
    "bridge_?": ["bridge_%s" % v for v in STONE_VARIANTS]
}

TEMPLATES = {
    #"stick_?": ["wood_?"],
    #"metal_?": ["ore_?"],
    #"rope": ["grass"],
    #"axe_?_?": ["stick_?", "stone_?"],
    #"trap_?": ["metal_?", "rope"],
    #"sword_?_?": ["metal_?", "stone_?"],
    #"bridge_?": ["rope", "stone_?"]

    "shovel_?_?": ["wood_?", "ore_?"],
    "ladder_?": ["wood_?", "grass"],
    "axe_?_?": ["wood_?", "stone_?"],
    "trap_?": ["ore_?", "grass"],
    "sword_?_?": ["ore_?", "stone_?"],
    "bridge_?": ["grass", "stone_?"]
}

RECIPES = {}
HINTS = []
for group in CRAFTS.values():
    for goal in group:
        variants = goal.split("_")[1:]
        abstracted = re.sub(r"_[a-z]+", "_?", goal)
        ingredients = TEMPLATES[abstracted]

        spec_parents = []
        var_parents = list(variants)
        for ingredient in ingredients:
            if "?" in ingredient:
                spec_parents.append(ingredient.replace("?", var_parents.pop(0)))
            else:
                spec_parents.append(ingredient)
        RECIPES[goal] = spec_parents

        base = False
        while not base:
            base = True
            new_ingredients = []
            for ing in ingredients:
                if ing in TEMPLATES:
                    new_ingredients += TEMPLATES[ing]
                    base = False
                else:
                    new_ingredients.append(ing)
            ingredients = new_ingredients
        specialized = []
        for ingredient in ingredients:
            if "?" in ingredient:
                specialized.append(ingredient.replace("?", variants.pop(0)))
            else:
                specialized.append(ingredient)
        #print goal, specialized
        HINTS.append((goal, specialized))

for ingredient in INGREDIENTS:
    HINTS.append((ingredient, [ingredient]))

util.next_random().shuffle(HINTS)

for x in RECIPES.items():
    print x

print

for x in HINTS:
    print x

#for i, (k, v) in enumerate(HINTS):
#    print i, k, v

#HINTS = [
#    ("wood", ["wood"]),
#    ("ore", ["ore"]),
#    ("grass", ["grass"]),
#    ("stone", ["stone"]),
#    ("stick", ["wood", "craft"]),
#    ("metal", ["ore", "craft"]),
#    ("rope", ["grass", "craft"]),
#    ("shovel", ["wood", "metal", "craft"]),
#    ("ladder", ["wood", "grass", "craft"]),
#    ("axe", ["wood", "stone", "craft"]),
#    ("trap", ["ore", "grass", "craft"]),
#    ("sword", ["ore", "stone", "craft"]),
#    ("bridge", ["grass", "stone", "craft"]),
#    #("shovel", ["stick", "metal", "craft"]),
#    #("ladder", ["stick", "rope", "craft"]),
#    #("axe", ["stick", "stone", "craft"]),
#    #("trap", ["metal", "rope", "craft"]),
#    #("sword", ["metal", "stone", "craft"]),
#    #("bridge", ["rope", "stone", "craft"]),
#]

TEST_IDS = list(range(len(HINTS))[::4])
#TEST_IDS = list(range(len(HINTS))[::2])
TRAIN_IDS = [i for i in range(len(HINTS)) if i not in TEST_IDS]

#TRAIN_IDS = [TEST_IDS[3]]
#TEST_IDS = [TEST_IDS[0]]

#TEST_IDS = list(range(len(HINTS)))

#TEST_IDS = [len(HINTS)-1]
#TRAIN_IDS = [len(HINTS)-1]

N_ACTIONS = 6
UP, DOWN, LEFT, RIGHT, USE, CRAFT = range(N_ACTIONS)

Minicraft2Task = namedtuple("Minicraft2Task", ["id", "goal", "hint"])

def random_free(grid, random):
    pos = None
    while pos is None:
        (x, y) = (random.randint(SIZE), random.randint(SIZE))
        if grid[x, y, :].any():
            continue
        pos = (x, y)
    return pos

def neighbors(pos, dir=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dir is None or dir == LEFT):
        neighbors.append((x-1, y))
    if y > 0 and (dir is None or dir == DOWN):
        neighbors.append((x, y-1))
    if x < SIZE - 1 and (dir is None or dir == RIGHT):
        neighbors.append((x+1, y))
    if y < SIZE - 1 and (dir is None or dir == UP):
        neighbors.append((x, y+1))
    return neighbors

class Minicraft2Instance(object):
    def __init__(self, task, init_state, demo=None):
        self.task = task
        self.state = init_state
        self.demo = demo
        self.instruction = task.hint

class Minicraft2World(object):
    def __init__(self):
        self.tasks = []
        self.index = util.Index()
        self.vocab = util.Index()
        self.ingredients = [self.index.index(k) for k in INGREDIENTS]
        self.recipes = {
            self.index.index(k): set(self.index.index(vv) for vv in v)
            for k, v in RECIPES.items()
        }
        print self.recipes
        #self.hints = [
        #    (self.index.index(k), tuple(self.vocab.index(vv) for vv in v))
        #    for k, v in HINTS
        #]
        self.hints = []
        for k, v in HINTS:
            self.hints.append((self.index.index(k), v))
            for w in v:
                self.vocab.index(w)

        self.kind_to_obs = {}
        self.obs_to_kind = {}
        for k in self.ingredients:
            self.kind_to_obs[k] = len(self.kind_to_obs)
            self.obs_to_kind[self.kind_to_obs[k]] = k

        self.n_obs = (
            2 * WINDOW_SIZE * WINDOW_SIZE * len(self.kind_to_obs)
            + len(self.index)
            + 4)
        self.n_act = N_ACTIONS
        self.n_actions = self.n_act
        self.n_features = self.n_obs
        self.is_discrete = True

        self.max_hint_len = 3
        self.n_vocab = len(self.vocab)
        self.random = util.next_random()

        self.tasks = []
        for i, (goal, steps) in enumerate(self.hints):
            self.tasks.append(Minicraft2Task(i, goal, None))
        self.n_tasks = len(self.tasks)
        self.n_train = len(TRAIN_IDS)
        self.n_val = 0
        self.n_test = len(TEST_IDS)

        self.demos = {}
        #for i in range(len(TRAIN_IDS)):
        #    with open("data/minicraft/paths.%d.pkl" % i, "rb") as pf:
        #        paths = pickle.load(pf)[-1000:]
        #        task = paths[0][0]._replace(hint=None)
        #        assert task == self.tasks[TRAIN_IDS[i]]
        #        self.demos[TRAIN_IDS[i]] = [v for k, v in paths]

    def sample_instance(self, task_id):
        task = self.tasks[task_id]
        _, steps = self.hints[task_id]
        indexed_steps = [self.vocab[w] for w in steps]
        task = task._replace(hint=tuple(indexed_steps))
        demo = None
        if task_id in self.demos:
            episodes = self.demos[task_id]
            episode = episodes[self.random.randint(len(episodes))]
            demo = []
            for transition in episode:
                demo.append((
                    FakeMinicraft2State(transition.s1),
                    transition.a[0],
                    FakeMinicraft2State(transition.s2)))
        state = self.sample_state(task)
        return Minicraft2Instance(task, state, demo)

    def sample_state(self, task):
        grid = np.zeros((SIZE, SIZE, len(self.kind_to_obs)))
        for k in self.ingredients:
            obs = self.kind_to_obs[k]
            for _ in range(1):
                x, y = random_free(grid, self.random)
                grid[x, y, obs] = 1

        init_pos = random_free(grid, self.random)
        init_dir = self.random.randint(4)
        return Minicraft2State(self, grid, init_pos, init_dir, np.zeros(len(self.index)), task)

    #def sample_train(self, p=None):
    #    return self.sample_instance(self.random.choice(TRAIN_IDS, p=p))

    #def sample_val(self, p=None):
    #    assert False
    #    return self.sample_instance(self.random.choice(TRAIN_IDS, p=p))

    #def sample_test(self, p=None):
    #    return self.sample_instance(self.random.choice(TEST_IDS, p=p))

    def sample_train(self):
        return self.sample_instance(self.random.choice(TRAIN_IDS)).state

    def sample_test(self):
        return self.sample_instance(self.random.choice(TEST_IDS)).state

    def reset(self, insts):
        return [inst.state.features() for inst in insts]

    def step(self, actions, insts):
        features, rewards, stops = [], [], []
        for action, inst in zip(actions, insts):
            reward, new_state, stop = inst.state.step(action)
            inst.state = new_state
            features.append(new_state.features())
            rewards.append(reward)
            stops.append(stop)
        return features, rewards, stops

class FakeMinicraft2State(object):
    def __init__(self, features):
        self._features = features

    def features(self):
        return self._features

    def step(self, action):
        assert False

class Minicraft2State(object):
    def __init__(self, world, grid, pos, dir, inventory, task):
        self.world = world
        self.grid = grid
        self.pos = pos
        self.dir = dir
        self.inventory = inventory
        self.task = task
        self._cached_features = None
        self.instruction = self.task.hint
        self.features = self._features()

    def _features(self):
        if self._cached_features is not None:
            return self._cached_features

        x, y = self.pos
        hs = WINDOW_SIZE / 2
        bhs = WINDOW_SIZE * WINDOW_SIZE / 2

        grid_feats = array.pad_slice(self.grid, (x-hs, x+hs+1), (y-hs, y+hs+1))
        grid_feats_big = array.pad_slice(self.grid, (x-bhs, x+bhs+1), (y-bhs, y+bhs+1))
        grid_feats_red = block_reduce(grid_feats_big, (WINDOW_SIZE, WINDOW_SIZE, 1), func=np.max)

        pos_feats = np.asarray(self.pos, dtype=np.float32) / SIZE
        dir_feats = np.zeros(4)
        dir_feats[self.dir] = 1

        features = np.concatenate((
            grid_feats.ravel(),
            grid_feats_red.ravel(),
            self.inventory,
            dir_feats))
        self._cached_features = features
        return features

    def step(self, action):
        x, y = self.pos
        new_dir = self.dir
        new_inventory = self.inventory
        new_grid = self.grid
        reward = 0
        stop = False

        dx, dy = 0, 0
        if action == UP:
            dx, dy = 0, 1
            new_dir = UP
        elif action == DOWN:
            dx, dy = 0, -1
            new_dir = DOWN
        elif action == LEFT:
            dx, dy = -1, 0
            new_dir = LEFT
        elif action == RIGHT:
            dx, dy = 1, 0
            new_dir = RIGHT
        elif action == USE:
            for nx, ny in neighbors(self.pos, self.dir):
                here = self.grid[nx, ny, :]
                if not here.any():
                    continue
                if here.sum() > 1:
                    assert False
                assert here.sum() == 1
                obs = here.argmax()
                new_inventory = self.inventory.copy()
                new_grid = self.grid.copy()
                new_inventory[self.world.obs_to_kind[obs]] += 1
                new_grid[nx, ny, obs] = 0
        elif action == CRAFT:
            new_inventory = self.inventory.copy()
            more = True
            while more:
                more = False
                for product, ingredients in self.world.recipes.items():
                    if all(new_inventory[ing] > 0 for ing in ingredients):
                        new_inventory[product] += 1
                        for ing in ingredients:
                            new_inventory[ing] -= 1
                        #more = True
                        break

        n_x = x + dx
        n_y = y + dy
        if n_x < 0 or n_x >= SIZE:
            n_x = x
        if n_y < 0 or n_y >= SIZE:
            n_y = y
        if self.grid[n_x, n_y, :].any():
            n_x, n_y = x, y

        if new_inventory[self.task.goal] > 0:
            reward = 1
            stop = True

        new_state = Minicraft2State(self.world, new_grid, (n_x, n_y), new_dir, new_inventory, self.task)
        #return reward, new_state, stop
        return new_state, reward, stop
