import numpy as np
import loading

train_local, test_local = loading.load("local", "human", 1000, 1000)
train_local = zip(*train_local)
test_local = zip(*test_local)
train_global, test_global = loading.load("global", "human", 1000, 1000)
train_global = zip(*train_global)
test_global = zip(*test_global)

random = np.random.RandomState(3493)

def sample_train():
    if random.rand() < 0.5:
        datum = train_local[random.choice(len(train_local))]
    else:
        datum = train_global[random.choice(len(train_global))]
