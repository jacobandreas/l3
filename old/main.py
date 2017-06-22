#!/usr/bin/env python2

from models import Model3
#from tasks import count, shapes, birds, regex, regex2
from tasks import regex2
import util

import numpy as np
import tensorflow as tf

dataset = regex2

model = Model3("model", dataset)
print "built model"

optimizer = tf.train.AdamOptimizer(0.001)
o_train = optimizer.minimize(model.t_loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

random = util.next_random()

for i_epoch in range(1000):
    e_loss = 0.
    for i_batch in range(100):
        batch = dataset.sample_train()
        feed_dict = model.feed(batch)
        loss, _ = session.run(
                [model.t_loss, o_train], feed_dict)
        e_loss += loss

    batch = dataset.sample_train()
    e_acc = model.predict(batch, session)

    test_data = dataset.sample_test()
    t_acc = model.predict(test_data, session)
    
    #e_acc = np.mean([p == d.label for p, d in zip(preds, batch)])
    #e_acc = 0.

    print "%01.3f %01.3f" % (e_loss, e_acc)
    print "%01.3f" % t_acc
    print

    #test_data = dataset.sample_test()
    #samp = model.sample(test_data, session)
    #idx = random.randint(len(test_data))
    #print " ".join(dataset.nl_vocab.get(i) for i in samp[idx])
    #print "  " + " ".join(dataset.nl_vocab.get(i) for i in test_data[idx].hint)
    #preds = model.predict(test_data, session)
    #acc = np.mean([p == d.label for p, d in zip(preds, test_data)])
    #print acc
    #print


#test_exemplars = dataset.sample_test_exemplars()
#model.fit(test_exemplars, session)
#
#test_data = dataset.sample_test_data()
#preds = model.predict(test_data)
#acc = np.mean([p == d.label for p, d in zip(preds, test_data)])
#print acc
