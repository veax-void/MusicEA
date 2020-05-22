#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST 
"""
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import random
import numpy as np
from HMM import HMM

np.random.seed(18)
random.seed(18)

observations = ['3L', '2M', '1S', '2M', '1S', '3L', '3L', '3L']

states = ['1H', '2C']
observables = ['1S', '2M', '3L']

#observations = []
#for _ in range(50000):
#	observations.append(random.choice(observables))

model = HMM(states, observables)
model.fit(observations, epochs=100)

# generate new sequence
#chain = model.generate(len(observations))
#print(chain)

# probability of this observation to occur
score = model.score(observations)
print('Score: {}'.format(score))


'''
Get -> return numpy array
Set -> take numpy array
'''
# Test get interface
print('A\n',model.get_A())
print('B\n',model.get_B())
print('Pi\n',model.get_pi())

# Test set interface
a = model.get_A()
b = model.get_B()
pi = model.get_pi()

a[0] = [0.1, 0.9]
b[1] = [0.8, 0.1, 0.1]
pi = [0.5, 0.5]

model.set_A(a)
model.set_B(b)
model.set_pi(pi)

print('new_A\n',model.get_A())
print('new_B\n',model.get_B())
print('new_Pi\n',model.get_pi())







