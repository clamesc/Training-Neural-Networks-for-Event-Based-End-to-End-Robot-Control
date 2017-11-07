#!/usr/bin/env python

import numpy as np
import h5py
import tensorflow as tf
from parameters import *

class ClassificationNN():
	def __init__(self, in_size, hidden_size, out_size):
		self.x = tf.placeholder(tf.float32, shape=[None, in_size])
		self.y_ = tf.placeholder(tf.uint8, shape=[None])
		self.keep_prob = tf.placeholder(tf.float32)

		self.W_fc1 = tf.Variable(tf.truncated_normal([in_size, hidden_size], stddev=0.1))
		self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
		self.h_fc1 = tf.nn.relu(tf.matmul(self.x, self.W_fc1) + self.b_fc1)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		self.W_fc2 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=0.1))
		self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1_drop, self.W_fc2))
		self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

		self.W_fc3 = tf.Variable(tf.truncated_normal([hidden_size, out_size], stddev=0.1))
		self.y_fc = tf.nn.relu(tf.matmul(self.h_fc2_drop, self.W_fc3))

		self.y_onehot = tf.one_hot(self.y_,out_size,dtype=tf.float32)

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_onehot, logits=self.y_fc))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		self.correct_prediction = tf.equal(tf.argmax(self.y_fc,1), tf.argmax(self.y_onehot,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class dataset():
	def __init__(self, path):
		# Read DQN data
		h5f = h5py.File(path + '/dqn_data.h5', 'r')
		self.states = np.array(h5f['states'], dtype=float)
		self.actions = np.array(h5f['actions'])
		h5f.close()
		# Delete empty states at the end of array
		for i in range(self.states.shape[0]):
			if self.states[-i].any():
				break
		# Normalize states
		self.states = self.states[:-i+1]/np.amax(self.states)
		self.actions = self.actions[:-i+1]
	def next_batch(self, batchsize):
		# Return batch of state-action samples for training
		idx = np.random.choice(self.states.shape[0],size=batchsize,replace=False)
		return [self.states[idx], self.actions[idx]]

net = ClassificationNN(sensor_size,h_size,action_size)
policy = dataset(path)
with tf.Session() as sess:
	# Train Neural Network on dataset
	sess.run(tf.global_variables_initializer())
	for i in range(10000):
		batch = policy.next_batch(50)
		if i%100 == 0:
			train_accuracy = net.accuracy.eval(feed_dict={net.x:batch[0], net.y_: batch[1], net.keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		net.train_step.run(feed_dict={net.x: batch[0], net.y_: batch[1], net.keep_prob: 1.0})
	print("test accuracy %g"%net.accuracy.eval(feed_dict={net.x: policy.states, net.y_: policy.actions, net.keep_prob: 1.0}))
	# Save learned weights
	W1, b1, W2, W3 = sess.run([net.W_fc1, net.b_fc1, net.W_fc2, net.W_fc3])
	h5f = h5py.File(path + '/snn_data.h5', 'w')
	h5f.create_dataset('W1', data=W1)
	h5f.create_dataset('W2', data=W2)
	h5f.create_dataset('W3', data=W3)
	h5f.create_dataset('b1', data=b1)
	h5f.close()