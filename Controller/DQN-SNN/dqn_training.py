#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
import h5py
from environment import VrepEnvironment
from qnetwork import QNetwork
from experience_buffer import ExperienceBuffer
from parameters import *
import matplotlib.pyplot as plt

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()
mainQN = QNetwork(sensor_size, action_size, h_size, l_rate, init_value)
targetQN = QNetwork(sensor_size, action_size, h_size, l_rate, init_value)
init = tf.initialize_all_variables()
saver = tf.train.Saver()
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)
myBuffer = ExperienceBuffer(total_size,resolution,buffersize)
env = VrepEnvironment(motor_speed, turn_speed, resolution, reset_distance, pub_rate, dvs_queue, resize_factor, crop)

# Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/anneling_steps

jList = []
rList = []
total_steps = 0

# Make a path for model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:

    writer = tf.summary.FileWriter(path, graph=tf.get_default_graph())

    # Set tensorflow network
    sess.run(init)
    if load_model == True:
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    updateTarget(targetOps,sess) # Set the target network to be equal to the primary network.

    for i in range(num_episodes): # Loop over episodes
        # Reset environment
        s = env.reset()		# Get first observation
        d = False			# Terminal state
        rAll = 0 			# Reward in an episode
        j = 0				# Episode steps
        
        while j < max_epLength: # Loop over episode steps
            
            # Choose action (epsilon-greedy policy)
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,action_size)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[np.where(s>0, 1, 0)]})[0]
            
            # Perform step
            s1,r,d = env.step(a)[:3]
            j += 1
            total_steps += 1

            #Save the experience to episode buffer
            myBuffer.add([s,a,r,s1,d]) 
            
            if total_steps > pre_train_steps:
            	# Decrease random action probability
                if e > endE:
                    e -= stepDrop
                
                # Perform gradient descent step on action network
                # Update target network weights towards action network weights
                if total_steps % (update_freq) == 0:

                	#Get a random batch of experiences.
                    trainBatch = myBuffer.sample(batch_size)
                    
                    # Calculate Target Q values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[3])})
                    end_multiplier = -(trainBatch[4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[2] + (y*doubleQ*end_multiplier)
                    
                    # Perform gradient descent step
                    _, summary = sess.run([mainQN.updateModel, mainQN.summary_op], \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[1]})
                    writer.add_summary(summary, total_steps)

                    # Update target network
                    updateTarget(targetOps,sess)

            rAll += r 	# Add episode reward
            s = s1		# Set state for next step

            if d == True: # Break episode if terminal state
                break
        
        jList.append(j)
        rList.append(rAll)

        # Print episode and total steps after each episode
        print "Episode: " + str(i) + "    Total steps: " + str(total_steps)
        r_summary = tf.Summary(value=[tf.Summary.Value(tag="r_ep", simple_value=rAll),])
        j_summary = tf.Summary(value=[tf.Summary.Value(tag="j_ep", simple_value=j),])
        writer.add_summary(r_summary, i)
        writer.add_summary(j_summary, i)

        if total_steps > max_steps: # End training after max number of steps
            break

    # Generate action labels for each state
    actions = np.zeros(total_size, dtype=int)
    for step in range(total_size):
        actions[step] = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[np.where(myBuffer.pre_s[step]>0, 1, 0)]})[0]

    # Get network weights
    W1, b1, W2, b2, W3, b3 = sess.run([mainQN.W1, mainQN.b1, mainQN.W2, mainQN.b2, mainQN.Wout, mainQN.bout])
    
    # Save state-action dataset
    saver.save(sess,path + '/model-'+str(i)+'.cptk')
    h5f = h5py.File(path + '/dqn_data.h5', 'w')
    h5f.create_dataset('states', data=myBuffer.pre_s)
    h5f.create_dataset('actions', data=actions)

    # Save network weights
    h5f.create_dataset('W1', data=W1)
    h5f.create_dataset('W2', data=W2)
    h5f.create_dataset('W3', data=W3)
    h5f.create_dataset('b1', data=b1)
    h5f.create_dataset('b2', data=b2)
    h5f.create_dataset('b3', data=b3)
    h5f.close()