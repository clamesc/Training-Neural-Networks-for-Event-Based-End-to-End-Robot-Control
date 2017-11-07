#!/usr/bin/env python
import tensorflow as tf

class QNetwork():
    def __init__(self,in_size, out_size, hidden_size, learn_rate, init_val):
        
        with tf.name_scope('Sensor_Input'):
            self.scalarInput = tf.placeholder("float", [None, in_size])

        # Hidden layer with RELU activation
        with tf.name_scope('1_Hidden_Layer'):
        	self.W1 = tf.Variable(tf.random_uniform([in_size, hidden_size], minval=-init_val, maxval=init_val))
        	self.b1 = tf.Variable(tf.random_uniform([hidden_size], minval=-init_val, maxval=init_val))
        	self.layer_1 = tf.add(tf.matmul(self.scalarInput, self.W1), self.b1)
        	self.layer_1 = tf.nn.relu(self.layer_1)

        # Hidden layer with RELU activation
        with tf.name_scope('2_Hidden_Layer'):
            self.W2 = tf.Variable(tf.random_uniform([hidden_size, hidden_size], minval=-init_val, maxval=init_val))
            self.b2 = tf.Variable(tf.random_uniform([hidden_size], minval=-init_val, maxval=init_val))
            self.layer_2 = tf.add(tf.matmul(self.layer_1, self.W2), self.b2)
            self.layer_2 = tf.nn.relu(self.layer_2)
        
        # Output layer
        with tf.name_scope('Q_Values'):
            self.Wout = tf.Variable(tf.random_uniform([hidden_size, out_size], minval=-init_val, maxval=init_val))
            self.bout = tf.Variable(tf.random_uniform([out_size], minval=-init_val, maxval=init_val))
            self.Qout = tf.matmul(self.layer_2, self.Wout) + self.bout

        # Action prediction
        with tf.name_scope('Action'):
            self.predict = tf.argmax(self.Qout,1)

        # Loss
        with tf.name_scope('Loss'):
            self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,out_size,dtype=tf.float32)
            self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
        
        # Training
        with tf.name_scope('Training'):
            self.trainer = tf.train.AdamOptimizer(learning_rate=learn_rate)
            self.updateModel = self.trainer.minimize(self.loss)

        self.summary_op = tf.summary.scalar("td_error", self.loss)