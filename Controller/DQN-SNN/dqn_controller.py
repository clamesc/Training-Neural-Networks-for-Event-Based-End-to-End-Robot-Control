#!/usr/bin/env python

import os
import numpy as np
import h5py
from environment import VrepEnvironment
from parameters import *

class QNetwork():
    def __init__(self, path):
        
        # Get weights from trained network
        h5f = h5py.File(path+'/dqn_data.h5', 'r')
        W1 = np.array(h5f['W1'])
        W2 = np.array(h5f['W2'])
        W3 = np.array(h5f['W3'])
        b1 = np.array(h5f['b1'])
        b2 = np.array(h5f['b2'])
        b3 = np.array(h5f['b3'])
        self.weights = []
        self.bias = []
        self.weights.extend([W1,W2,W3])
        self.bias.extend([b1,b2,b3])

        # Initialize layer arrays
        self.activations = []
        for layer in self.weights:
            self.activations.append(np.zeros(layer.shape[0]))
        self.activations.append(np.zeros(self.weights[-1].shape[1]))
    
    def simulate(self, input_data):
        # Reset layer arrays
        for l in range(len(self.weights)):
            self.activations[l] = np.zeros(self.weights[l].shape[0])
        self.activations[-1] = np.zeros(self.weights[-1].shape[1])
        # Propagate input through network
        self.activations[0] = input_data
        for l in range(len(self.weights)):
            inp = np.dot(self.weights[l].T, self.activations[l])
            if self.bias[l] is not None:
                inp += self.bias[l]
            self.activations[l+1] = inp * (inp > 0)
        return self.activations[-1]

# Initialize
QNN = QNetwork(path)
env = VrepEnvironment(motor_speed, turn_speed, resolution, reset_distance, pub_rate, dvs_queue, resize_factor, crop)
distance = []
position = []

# Reset robot
s = env.reset()
while True:
	# Use network to compute action from state
    a = np.argmax(QNN.simulate(np.where(s>0, 1, 0)))
    # Perform action, get state, reward, termination, distance, position
    s,r,t,d,p = env.step(a)
    if p < 0.49 or t:
    	break
    distance.append(d)
    position.append(p)

# Save performance data
h5f = h5py.File(path + '/dqn_performance_data.h5', 'w')
h5f.create_dataset('distance', data=distance)
h5f.create_dataset('position', data=position)
h5f.close()