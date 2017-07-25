import util

import numpy as np
import loading

from skimage.measure import block_reduce

ROWS = 10
COLS = 10
OBJS = 11
WINDOW_SIZE = 5
ACTS = 4

START = "<s>"
STOP = "</s>"
UNK = "UNK"

random = util.next_random()

class NavDatum(object):
    def __init__(self, features, reward, terminal, instructions, task):
        self.features = features
        self.reward = reward
        self.terminal = terminal
        self.instructions = instructions
        self.task = task

    def init(self):
        pos = None
        while pos is None:
            r, c = random.randint(ROWS), random.randint(COLS)
            if True: #self.features[r, c, 0] == 1:
                pos = r, c
        instruction = self.instructions[random.choice(len(self.instructions))]
        return NavState(pos, instruction, self)

class NavState(object):
    def __init__(self, position, instruction, datum):
        self.position = position
        self.instruction = instruction
        self.datum = datum
        self.features = self._make_features(datum.features)

    def _make_features(self, raw):
        r, c = self.position
        padded = np.zeros((2*ROWS-1, 2*COLS-1, OBJS+3))
        sr = ROWS - r - 1
        sc = COLS - c - 1
        assert 0 <= sr < 2 * ROWS - 1
        assert 0 <= sc < 2 * COLS - 1
        padded[sr:sr+ROWS, sc:sc+COLS, :] = raw
        padded_red = block_reduce(padded, (WINDOW_SIZE, WINDOW_SIZE, 1), func=np.max)
        print padded.shape
        print padded_red.shape
        exit()
        return padded

    def step(self, action):
        assert action < ACTS
        r, c = self.position
        if action == 0:
            r -= 1
        elif action == 1:
            c -= 1
        elif action == 2:
            r += 1
        elif action == 3:
            c += 1
        r = min(max(r, 0), ROWS-1)
        c = min(max(c, 0), COLS-1)

        rew = self.datum.reward[r, c]
        term = self.datum.terminal[r, c]
        if rew > 0:
            assert term

        if rew > 0:
            rew = 1
        elif rew < 0:
            rew = -.1

        #term = False
        #rew = min(self.datum.reward[r, c], 0)

        return NavState((r, c), self.instruction, self.datum), rew, term

    def render(self):
        rows = []
        rows.append(" ".join([self.datum.task.vocab.get(w) for w in self.instruction]))
        for r in range(ROWS):
            row = []
            for c in range(COLS):
                if (r, c) == self.position:
                    row.append("@")
                elif self.datum.reward[r, c] > 0:
                    row.append("*")
                elif self.datum.reward[r, c] < 0:
                    row.append("^")
                else:
                    row.append(".")
            rows.append("".join(str(s) for s in row))
        return "\n".join(rows)

class NavTask(object):
    def __init__(self):
        train_local, test_local = loading.load("local", "human", 2000, 500)
        train_global, test_global = loading.load("global", "human", 1500, 500)
        #train_local, test_local = loading.load("local", "human", 20, 20)
        #train_global, test_global = loading.load("global", "human", 20, 20)
        self.vocab = util.Index()
        self.vocab.index("UNK")

        loaded_splits = {
            ("train", "local"): train_local,
            ("train", "global"): train_global,
            ("test", "local"): test_local,
            ("test", "global"): test_global
        }

        splits = {}
        for fold in ("train", "test"):
            for mode in ("local", "global"):
                insts = loaded_splits[fold, mode]
                terrains, objects, rewards, terminals, instructions, _, _ = insts
                data = []
                for i in range(terrains.shape[0]):
                    terr = terrains[i, 0, :, :].astype(int)
                    obj = objects[i, 0, :, :].astype(int)
                    rew = rewards[i, 0, :, :]
                    term = terminals[i, 0, :, :].astype(int)
                    instr = instructions[i]

                    features = np.zeros((ROWS, COLS, OBJS+3))
                    for r in range(ROWS):
                        for c in range(COLS):
                            #features[r, c, :] = rew[r, c]
                            if terr[r, c] == 0:
                                assert obj[r, c] == 0
                                features[r, c, OBJS] = 1
                            else:
                                features[r, c, obj[r, c]] = 1

                            features[r, c, OBJS+1] = 1. * r / ROWS
                            features[r, c, OBJS+2] = 1. * c / COLS

                    indexed_instr = []
                    for instruction in instructions:
                        p_instruction = [START] + instruction.split() + [STOP]
                        if fold == "train":
                            toks = [self.vocab.index(w) for w in p_instruction]
                        else:
                            toks = [self.vocab[w] or self.vocab[UNK] for w in p_instruction]
                        indexed_instr.append(toks)

                    datum = NavDatum(features, rew, term, indexed_instr, self)
                    data.append(datum)
                splits[fold, mode] = data

        self.train_local = splits["train", "local"]
        self.train_global = splits["train", "global"]
        #self.train = self.train_local + self.train_global
        self.train = self.train_local
        self.test_local = splits["test", "local"]
        self.test_global = splits["test", "global"]
        #self.test = self.test_local + self.test_global
        self.train = self.train_local
        self.test = self.test_local

        samp = self.sample_train()
        #self.n_features = samp.features.size
        self.feature_shape = samp.features.shape
        self.n_actions = ACTS

    def sample_train(self):
        data = self.train
        datum = data[random.choice(len(data))]
        return datum.init()

    def sample_test(self):
        data = self.test
        datum = data[random.choice(len(data))]
        return datum.init()
