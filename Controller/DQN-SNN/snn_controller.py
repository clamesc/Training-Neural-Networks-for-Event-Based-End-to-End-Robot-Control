#!/usr/bin/env python

import numpy as np
import h5py
from environment import VrepEnvironment
from parameters import *

class weighted_output():
	def __init__(self, size, factor=0.8):
		self.output = np.zeros(size)
		self.factor = factor
	def step(self, input_spikes):
		self.output *= self.factor
		self.output += input_spikes
		return np.argmax(self.output)

class SpikingQN():
	def __init__(self, weights, bias):
		
		# Get weights from trained network
		self.weights = weights
		self.bias = bias

		# Normalize weights such that maximum positive input is 1 in each layer
		for l in range(len(self.weights)):
			max_pos_input = 0
			# Find maximum input for this layer
			for o in range(self.weights[l].shape[1]):
				input_sum = 0
				for i in range(self.weights[l].shape[0]):
					input_sum += max(0, self.weights[l][i,o])
				if self.bias[l] is not None:
					input_sum += max(0, self.bias[l][o])
				max_pos_input = max(max_pos_input, input_sum)
			# Rescale all weights
			self.weights[l] = self.weights[l] / max_pos_input
			if self.bias[l] is not None:
				self.bias[l] = self.bias[l] / max_pos_input

		# Initialize layer arrays
		self.spikes = []
		self.spike_stats = []
		self.potential = []
		for layer in self.weights:
			self.potential.append(np.zeros(layer.shape[1]))
			self.spikes.append(np.zeros((layer.shape[0]), dtype=bool))
			self.spike_stats.append(np.zeros((layer.shape[0]), dtype=int))
		self.spikes.append(np.zeros((self.weights[-1].shape[1]), dtype=bool))
		self.spike_stats.append(np.zeros((self.weights[-1].shape[1]), dtype=int))

		# Output object keeps track of action traces and return action
		# with the highest action trace value at each step
		self.output = weighted_output(action_size)
	
	def simulate(self, input_data, duration=1.0, max_rate=200, dt=0.001, threshold=0.5):
		# Reset layer arrays
		for l in range(len(self.weights)):
			self.potential[l] = np.zeros(self.weights[l].shape[1])
			self.spikes[l] = np.zeros((self.weights[l].shape[0]), dtype=bool)
			self.spike_stats[l] = np.zeros((self.weights[l].shape[0]), dtype=int)
		self.spikes[-1] = np.zeros((self.weights[-1].shape[1]), dtype=bool)
		self.spike_stats[-1] = np.zeros((self.weights[-1].shape[1]), dtype=int)
		
		for t in range(int(duration/dt)):
			
			# Create poisson distributed spikes from input
			rescale_fac = 1/(dt*max_rate)
			spike_snapshot = np.random.uniform(size=input_data.shape) * rescale_fac
			input_spikes = spike_snapshot <= input_data
			self.spikes[0] = input_spikes
			self.spike_stats[0] += input_spikes
			
			for l in range(len(self.weights)):
				# Get input impulse from incoming spikes
				impulse = np.dot(self.weights[l].T, self.spikes[l])
				# Add bias impulse
				if self.bias[l] is not None:
					impulse += self.bias[l]/rescale_fac
				# Add input to membrane potential
				self.potential[l] += impulse
				self.potential[l] *= self.potential[l] > 0
				# Check for spiking
				self.spikes[l+1] = self.potential[l] >= threshold
				self.spike_stats[l+1] += self.spikes[l+1]
				# Reset
				self.potential[l][self.spikes[l+1]] = 0
		
		return self.spike_stats

	def step(self, input_data, duration=0.01, max_rate=1000, dt=0.001, threshold=1.0):
		# Step function does not reset the whole network, only the output spikes
		self.spike_stats[-1] = np.zeros((self.weights[-1].shape[1]), dtype=int)

		for t in range(int(duration/dt)):

			# Create poisson distributed spikes from input
			rescale_fac = 1/(dt*max_rate)
			spike_snapshot = np.random.uniform(size=input_data.shape) * rescale_fac
			self.spikes[0] = spike_snapshot <= input_data

			for l in range(len(self.weights)):
				# Get input impulse from incoming spikes
				impulse = np.dot(self.weights[l].T, self.spikes[l])
				# Add bias impulse
				if self.bias[l] is not None:
					impulse += self.bias[l]/rescale_fac
				# Add input to membrane potential
				self.potential[l] += impulse
				self.potential[l] *= self.potential[l] > 0
				# Check for spiking
				self.spikes[l+1] = self.potential[l] >= threshold
				# Reset
				self.potential[l][self.spikes[l+1]] = 0

			self.spike_stats[-1] += self.spikes[-1]

		print "Output Spikes for each action: " + str(self.spike_stats[-1])
		return self.output.step(self.spike_stats[-1])

distance = []
position = []

# Get weights from pre-trained ANN
h5f = h5py.File(path + '/snn_data.h5', 'r')
W1 = np.array(h5f['W1'], dtype=float)
W2 = np.array(h5f['W2'], dtype=float)
W3 = np.array(h5f['W3'], dtype=float)
b1 = np.array(h5f['b1'], dtype=float)
weights = [W1,W2,W3]
bias = [b1,None,None]

# Initialization
sNN = SpikingQN(weights,bias)
env = VrepEnvironment(motor_speed, turn_speed, resolution, reset_distance, snn_pub_rate, snn_dvs_queue, resize_factor, crop)
s = env.reset()*snn_spike_factor

while True:
	# Simulate network and get action
	a = sNN.step(s*snn_spike_factor)
	# Perform action and get state, reward, termination, distance, position
	s,r,t,d,p = env.step(a)
	# End lap, if robot reaches starting position again
	if p < 0.49:
		break
	# Save position and distance for performance evaluation
	distance.append(d)
	position.append(p)

# Save performance data
h5f = h5py.File(path + '/snn_performance_data.h5', 'w')
h5f.create_dataset('distance', data=distance)
h5f.create_dataset('position', data=position)
h5f.close()