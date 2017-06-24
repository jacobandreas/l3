#!/usr/bin/env python2

from models import Model
from tasks import regex2

dataset = regex2
model = Model(dataset)

for i_epoch in range(1000):
    e_loss = 0.
    for i_batch in range(100):
        batch = dataset.sample_train()
        b_loss = model.train(batch)
        e_loss += b_loss

    batch = dataset.sample_train()
    e_acc = model.predict(batch)

    t_batch = dataset.sample_test()
    e_t_acc = model.predict(t_batch)

    print "[loss] %01.4f" % e_loss
    print "[acc]  %01.4f" % e_acc
    print "[tacc] %01.4f" % e_t_acc

    if i_epoch % 10 == 0:
        model.save()
