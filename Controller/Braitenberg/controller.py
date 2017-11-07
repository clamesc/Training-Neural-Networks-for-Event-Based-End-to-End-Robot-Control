#!/usr/bin/env python

import numpy as np
import network
from environment import *
from parameters import *
import h5py

# Initialize static SNN
snn = network.SpikingNeuralNetwork()

# Initialize environment
env = VrepEnvironment()

# Reset robot
s = env.reset()

distance = []
position = []

for i in range(10000):
	# Simulate SNN for 50 ms
	n_l, n_r = snn.simulate(s)
	# Feed network output into steering wheel model
	s, d, p = env.step(len(n_l), len(n_r))
	# Break, if robot reaches starting position again
	if p < 0.49:
		break
	# Record distance-position
	distance.append(d)
	position.append(p)

# Save performance data
h5f = h5py.File(path + '/braitenberg_performance_data.h5', 'w')
h5f.create_dataset('distance', data=distance)
h5f.create_dataset('position', data=position)
h5f.close()