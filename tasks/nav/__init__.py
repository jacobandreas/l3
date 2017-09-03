from misc import util

from collections import defaultdict
from gflags import FLAGS
import numpy as np
import loading

from skimage.measure import block_reduce

ROWS = 10
COLS = 10
OBJS = 11
WINDOW_SIZE = 5
ACTS = 5

MAPS_PER_TASK = 20

START = "<s>"
STOP = "</s>"
UNK = "UNK"

MAX_BREATH = 3

random = util.next_random()

FEATURE_NAMES = {
  1: "star",
  2: "circle",
  3: "triangle",
  4: "heart",
  5: "spade",
  6: "diamond",
}

ALL_FEATURE_NAMES = {"star", "circle", "triangle", "heart", "spade",
"diamond", "tree", "horse", "house", "rock"}

class NavDatum(object):
    def __init__(self, goal, features, reward, terminal, instructions, values, task, task_id):
        self.goal = goal
        self.features = features
        self.reward = reward
        self.terminal = terminal
        self.instructions = instructions
        self.values = values
        self.task = task
        self.task_id = task_id

    def init(self):
        pos = None
        while pos is None:
            r, c = random.randint(ROWS), random.randint(COLS)
            if self.features[r, c, 0] == 1:
                pos = r, c
        instruction = self.instructions[random.choice(len(self.instructions))]
        return NavState(pos, MAX_BREATH, instruction, self, self.task_id)

class NavState(object):
    def __init__(self, position, breath, instruction, datum, task_id, meta=None):
        self.position = position
        self.instruction = instruction
        self.breath = breath
        self.datum = datum
        self.features = self._make_features(datum.features)
        self.expert_a = self._make_expert()
        self.task_id = task_id
        self.meta = meta

    def annotate_instruction(self, new_instruction, meta):
        return NavState(
                self.position, self.breath, new_instruction, self.datum,
                self.task_id, meta=meta)

    def annotate_task(self, new_task, meta):
        return NavState(
                self.position, self.breath, self.instruction, self.datum,
                new_task, meta=meta)

    def _make_expert(self):
        best_action = -1
        best_value = -np.inf
        for action in range(ACTS):
            r, c = self.position
            if action == 0:
                r -= 1
            elif action == 1:
                c -= 1
            elif action == 2:
                r += 1
            elif action == 3:
                c += 1
            else:
                assert action == 4

            if r < 0 or r > ROWS-1:
                continue
            if c < 0 or c > COLS-1:
                continue

            value = self.datum.values[r, c]
            if value > best_value:
                best_action = action
                best_value = value

        assert best_action >= 0
        return best_action

    def _make_features(self, raw):
        r, c = self.position
        padded = np.zeros((2*ROWS-1, 2*COLS-1, OBJS+3))
        sr = ROWS - r - 1
        sc = COLS - c - 1
        assert 0 <= sr < 2 * ROWS - 1
        assert 0 <= sc < 2 * COLS - 1
        padded[sr:sr+ROWS, sc:sc+COLS, :] = raw
        padded_red = block_reduce(padded, (4, 4, 1), func=np.max)
        assert padded.shape[0] == padded.shape[1]
        mid = padded.shape[0] / 2
        padded_slice = padded[mid-2:mid+3, mid-2:mid+3, :]
        return np.concatenate((
            padded_red.ravel(),
            padded_slice.ravel()
        ))

    def step(self, action):
        assert action < ACTS
        pr, pc = self.position
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
        term = action == 4

        nbreath = self.breath
        if rew < 0:
            nbreath -= 1
        else:
            nbreath = MAX_BREATH
        if nbreath == 0:
            term = True

        if (r, c) == self.datum.goal and action == 4:
            term = True
            rew = 3

        return NavState((r, c), nbreath, self.instruction, self.datum,
                self.task_id, self.meta), rew, term

    def render(self):
        rows = []
        rows.append(" ".join([self.datum.task.vocab.get(w) for w in self.instruction]))
        for r in range(0):
            row = []
            for c in range(COLS):
                if (r, c) == self.position:
                    row.append("@")
                elif 0 < self.datum.features[r, c, :OBJS].argmax():
                    row.append(str(self.datum.features[r, c, :OBJS].argmax())[-1:])
                elif self.datum.features[r, c, OBJS] == 1:
                    assert self.datum.reward[r, c] < 0
                    row.append("^")
                else:
                    row.append(".")
            rows.append("".join(str(s) for s in row))
        return "\n".join(rows)

class NavTask(object):
    def __init__(self):
        train_local_raw, test_local_raw = loading.load("local", "human", 2000, 500)
        train_global_raw, test_global_raw = loading.load("global", "human", 1500, 500)
        #train_local, test_local = loading.load("local", "human", 200, 200)
        #train_global, test_global = loading.load("global", "human", 200, 200)
        self.vocab = util.Index()
        self.vocab.index("UNK")
        self.START = START
        self.STOP = STOP

        raw_splits = {
            ("train", "local"): train_local_raw,
            ("train", "global"): train_global_raw,
            ("test", "local"): test_local_raw,
            ("test", "global"): test_global_raw
        }

        templates = defaultdict(list)

        splits = {}
        task_id = 0
        for fold in ("train", "test"):
            for mode in ("local",):
                data, task_id = self._format_data(fold, raw_splits[fold, mode], templates, task_id)
                splits[fold, mode] = data

        final_templates = defaultdict(list)
        for k, vv in templates.items():
            counts = defaultdict(lambda: 0)
            for v in vv:
                counts[v] += 1
            for v, c in counts.items():
                if c > 1:
                    final_templates[k].append(v)
        final_templates = dict(final_templates)

        train_local_fmt = splits["train", "local"]
        test_local_fmt = splits["test", "local"]

        task_counter = 0
        self.train, n_train_tasks = self._build_examples(
                train_local_fmt, 5000, final_templates, task_counter,
                group_tasks=False)
        self.test, n_total_tasks = self._build_examples(
                test_local_fmt, 50, final_templates, n_train_tasks,
                group_tasks=True)
        self.test_ids = sorted(list(set(d.task_id for d in self.test)))

        print "[n_train]", len(self.train)
        print "[n_test]", len(self.test), "test"

        samp = self.sample_train()
        self.n_features = samp.features.size
        self.n_tasks = n_train_tasks
        self.n_actions = ACTS

        print "[task_ids]", n_train_tasks, "->", n_total_tasks
        print "[n_vocab]", len(self.vocab)

    def _format_data(self, fold, insts, templates, task_id):
        terrains, objects, rewards, terminals, instructions, values, goals = insts
        data = []
        for i in range(terrains.shape[0]):
            terr = terrains[i, 0, :, :].astype(int)
            obj = objects[i, 0, :, :].astype(int)
            rew = rewards[i, 0, :, :]
            term = terminals[i, 0, :, :].astype(int)
            goal = goals[i]

            formatted_instr = []
            for off_r, off_c in [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]:
                gr, gc = goal
                tr, tc = gr + off_r, gc + off_c
                if tr < 0 or tr >= ROWS or tc < 0 or tc >= COLS:
                    continue
                gobj = obj[tr, tc]
                instruction = instructions[i].replace(".", "")
                toks = instruction.split()
                good_names = [w for w in instruction.split() 
                        if gobj in FEATURE_NAMES and w == FEATURE_NAMES[gobj]]
                names = [w for w in instruction.split() 
                        if w in ALL_FEATURE_NAMES]
                if (
                        gobj > 0 and 
                        gobj < OBJS - 4 and
                        len(good_names) == 1 and
                        len(names) == 1):
                    template = instruction.replace(good_names[0], "%s")
                    templates[(off_r, off_c)].append(template)
                    formatted_instr.append(instruction)
            if len(formatted_instr) == 0:
                formatted_instr = [""]

            features = np.zeros((ROWS, COLS, OBJS+3))
            for r in range(ROWS):
                for c in range(COLS):
                    if terr[r, c] == 0:
                        assert obj[r, c] == 0
                        features[r, c, OBJS] = 1
                    else:
                        features[r, c, obj[r, c]] = 1

                    features[r, c, OBJS+1] = 1. * r / ROWS
                    features[r, c, OBJS+2] = 1. * c / COLS

            indexed_instr = []
            for instruction in formatted_instr:
                p_instruction = [START] + instruction.split() + [STOP]
                if fold == "train":
                    toks = [self.vocab.index(w) for w in p_instruction]
                else:
                    toks = [self.vocab[w] or self.vocab[UNK] for w in p_instruction]
                indexed_instr.append(toks)

            datum = NavDatum(goal, features, rew, term, indexed_instr, values[i, ...], self, task_id)
            data.append(datum)
            task_id += 1

        return data, task_id

    def _compute_values(self, rew, term):
        def neighbors(r, c):
            out = []
            if r > 0:
                out.append((r-1, c))
            if r < ROWS - 1:
                out.append((r+1, c))
            if c > 0:
                out.append((r, c-1))
            if c < COLS - 1:
                out.append((r, c+1))
            return out

        v_curr = rew
        for _ in range(ROWS + COLS):
            v_next = rew.copy()
            for r in range(ROWS):
                for c in range(COLS):
                    if term[r, c]:
                        continue
                    prev = max(v_curr[pos] for pos in neighbors(r, c))
                    v_next[r, c] += FLAGS.discount * prev
            v_curr = v_next
        return v_curr

    def _randomize_map(self, datum):
        features = datum.features
        reward = datum.reward
        terminal = datum.terminal
        values = datum.values
        rot = random.randint(4)
        for _ in range(rot):
            features = np.rot90(features)
            reward = np.rot90(reward)
            terminal = np.rot90(terminal)
            values = np.rot90(values)
        return NavDatum(None, features, reward, terminal, None, values, datum.task, datum.task_id)

    def _build_examples(self, in_data, n_goals, templates, task_counter,
            group_tasks):
        offsets = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
        data = []
        for i_goal in range(n_goals):
            obj = random.choice(list(range(1, OBJS-4)))
            offset_r, offset_c = offsets[random.randint(len(offsets))]
            i_map = 0

            n_maps = MAPS_PER_TASK if group_tasks else 1

            while i_map < n_maps:
                base_datum = in_data[random.randint(len(in_data))]
                base_datum = self._randomize_map(base_datum)
                ((obj_r, obj_c),) = np.argwhere(base_datum.features[:, :, obj])
                goal_r = obj_r - offset_r
                goal_c = obj_c - offset_c
                if goal_r < 0 or goal_c < 0 or goal_r >= ROWS or goal_c >= COLS:
                    continue
                if base_datum.reward[goal_r, goal_c] < 0:
                    continue

                new_rew = np.minimum(base_datum.reward, 0)
                new_term = np.zeros((ROWS, COLS), dtype=np.int32)

                valid_templates = templates[(offset_r, offset_c)]
                chosen_template = random.choice(valid_templates)
                filled_template = chosen_template % FEATURE_NAMES[obj]
                new_instr = [self.vocab[START]] + [self.vocab[w] for w in
                        filled_template.split()] + [self.vocab[STOP]]
                assert None not in new_instr

                new_feats = base_datum.features

                fake_rew = new_rew.copy()
                fake_rew[goal_r, goal_c] = 3
                new_values = self._compute_values(fake_rew, new_term)

                new_goal = (goal_r, goal_c)

                datum = NavDatum(new_goal, new_feats, new_rew, new_term,
                        [new_instr],
                        new_values,
                        base_datum.task, 
                        task_id=task_counter
                        )
                data.append(datum)

            task_counter += 1

        return data, task_counter

    def sample_train(self):
        data = self.train
        datum = data[random.choice(len(data))]
        return datum.init()

    def sample_test(self, i_datum, erase_hint=False):
        data = self.test
        available = [d for d in data if d.task_id == i_datum]
        assert len(available) > 0
        datum = available[random.choice(len(available))]
        if erase_hint:
            datum.instructions = [[self.vocab[START], self.vocab[STOP]]]
        return datum.init()
