#!/usr/bin/env python

path = "./data/session_xyz" # Path for saving data

#================ DQN Controller ===============#

# Q-Network
init_value = 1e-3			# Maximum weight value for initialization
action_size = 3				# Number of actions
h_size = 200				# Hidden layer neurons

# Training
l_rate = 0.0001				# Learning rate
batch_size = 32             # How many experiences to use for each training step.
update_freq = 4             # How often to perform a training step.
startE = 1                  # Starting chance of random action
endE = 0.1                  # Final chance of random action
pre_train_steps = 1000      # How many steps of random actions before training begins.
anneling_steps = 99000.     # How many steps of training to reduce startE to endE.
max_steps = 170000			# Maximum number of training steps
num_episodes = 10000		# Maximum number of episodes

# MDP
reset_distance = 0.4		# Distance where episodes are terminated
y = .99                     # Discount factor on the target Q-values
max_epLength = 1000         # The max allowed length of an episode.

# DQN parameters
buffersize = 5000			# Number of recently experienced states used for training batch.
load_model = False          # Whether to load a saved model.
tau = 0.001                 # Rate to update target network toward primary network

# State parameters
dvs_resolution = [128,128]	# Resolution of the original DVS data
dvs_queue = 10				# Length of the FIFO queue saving DVS data during one step
resize_factor = 4			# Factor by which resolution is reduced
crop = [10,6]				# Crop at top [0] and bottom [1] after resizing

# Action parameters
motor_speed = 1.0			# Speed going straight
turn_speed = 0.25			# Speed added/subtracted in turns
pub_rate = 2				# Frequency of ROS motor speed publisher

# Other
total_size = max_steps + max_epLength
resolution = [(dvs_resolution[0]//resize_factor),(dvs_resolution[1]//resize_factor)-crop[0]-crop[1]]
sensor_size = resolution[0]*resolution[1]

#================ DQN-SNN Controller ===============#

snn_pub_rate = 20			# Frequency of motor speed publisher for the SNN controller
snn_dvs_queue = 1			# Length of the FIFO queue saving DVS data during one step
snn_spike_factor = 1.			# Factor scaling the state input