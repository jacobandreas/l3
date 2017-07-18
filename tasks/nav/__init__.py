import util

import numpy as np
import loading

ROWS = 10
COLS = 10
OBJS = 11
ACTS = 4

START = "<s>"
STOP = "</s>"
UNK = "UNK"

random = util.next_random()

class NavDatum(object):
    def __init__(self, features, reward, terminal, instructions):
        self.features = features
        self.reward = reward
        self.terminal = terminal
        self.instructions = instructions

    def init(self):
        pos = None
        while pos is None:
            r, c = random.randint(ROWS), random.randint(COLS)
            if self.features[r, c, 0] == 1:
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
        padded = np.zeros((2*ROWS-1, 2*COLS-1, OBJS+1))
        sr = ROWS - r - 1
        sc = COLS - c - 1
        assert 0 <= sr < 2 * ROWS - 1
        assert 0 <= sc < 2 * COLS - 1
        padded[sr:sr+ROWS, sc:sc+COLS, :] = raw
        return padded.ravel()

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
        #term = False
        #rew = min(self.datum.reward[r, c], 0)

        return NavState((r, c), self.instruction, self.datum), rew, term

class NavTask(object):
    def __init__(self):
        train_local, test_local = loading.load("local", "human", 2000, 500)
        train_global, test_global = loading.load("global", "human", 1500, 500)
        #train_local, test_local = loading.load("local", "human", 20, 20)
        #train_global, test_global = loading.load("global", "human", 20, 20)
        self.vocab = util.Index()

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

                    features = np.zeros((ROWS, COLS, OBJS+1))
                    for r in range(ROWS):
                        for c in range(COLS):
                            if terr[r, c] == 0:
                                assert obj[r, c] == 0
                                features[r, c, OBJS] = 1
                            else:
                                features[r, c, obj[r, c]] = 1

                    indexed_instr = []
                    for instruction in instructions:
                        p_instruction = [START] + instruction.split() + [STOP]
                        if fold == "train":
                            toks = [self.vocab.index(w) for w in p_instruction]
                        else:
                            toks = [self.vocab[w] or self.vocab[UNK] for w in p_instruction]
                        indexed_instr.append(toks)

                    datum = NavDatum(features, rew, term, indexed_instr)
                    data.append(datum)
                splits[fold, mode] = data

        self.train_local = splits["train", "local"]
        self.train_global = splits["train", "global"]
        self.train = self.train_local + self.train_global
        self.test_local = splits["test", "local"]
        self.test_global = splits["test", "global"]
        self.test = self.test_local + self.test_global

        samp = self.sample_train()
        self.n_features = samp.features.size
        self.n_actions = ACTS

    def sample_train(self):
        data = self.train
        datum = data[random.choice(len(data))]
        return datum.init()
