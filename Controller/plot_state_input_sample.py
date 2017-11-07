#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt

# show image from state-action dataset
# e.g. Fig. 4.6

class dataset():
	def __init__(self, path):
		h5f = h5py.File(path+'dqn_data.h5', 'r')
		self.states = np.array(h5f['states'], dtype=float)
		self.actions = np.array(h5f['actions'])
		h5f.close()
		for i in range(self.states.shape[0]):
			if self.states[-i].any():
				break
		self.states = self.states[:-i+1]/np.amax(self.states)
		self.actions = self.actions[:-i+1]
	def next_batch(self, batchsize):
		idx = np.random.choice(self.states.shape[0],size=batchsize,replace=False)
		return [self.states[idx], self.actions[idx]]

path = '/data/dqn/scenario x/'
data = dataset(path)

for i in range(100):
	plt.imshow(np.flipud(data.states[90000+i].reshape((16,32))))
	plt.axis('off')
	plt.show()