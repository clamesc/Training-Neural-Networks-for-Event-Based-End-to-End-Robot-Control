#!/usr/bin/env python

import numpy as np

path = "./data/session_xyz"			# Path for saving data

# Input image
dvs_resolution = [128,128]			# Original DVS frame resolution
crop_top = 40						# Crop at the top
crop_bottom = 24					# Crop at the bottom
resolution = [8,4]					# Resolution of reduced image

# Network parameters
sim_time = 50.0						# Length of network simulation during each step in ms
t_refrac = 2.						# Refractory period
time_resolution = 0.1				# Network simulation time resolution
iaf_params = {}						# IAF neuron parameters
poisson_params = {}					# Poisson neuron parameters
max_poisson_freq = 300.				# Maximum Poisson firing frequency for n_max
max_spikes = 15.					# number of events during each step for maximum poisson frequency

# R-STDP parameters
w_min = 0.							# Minimum weight value
w_max = 3000.						# Maximum weight value
w0_min = 200.						# Minimum initial random value
w0_max = 201.						# Maximum initial random value
tau_n = 200.						# Time constant of reward signal
tau_c = 1000.						# Time constant of eligibility trace
reward_factor = 0.01				# Reward factor modulating reward signal strength
A_plus = 1.							# Constant scaling strength of potentiaion
A_minus = 1.						# Constant scaling strength of depression				

# Steering wheel model
v_max = 1.5							# Maximum speed
v_min = 1.							# Minimum speed
turn_factor= 0.5					# Factor controls turn radius
turn_pre = 0						# Initial turn speed
v_pre = v_max						# Initial speed
n_max = sim_time//t_refrac - 10.	# Maximum input activity

# Other
reset_distance = 0.2				# Reset distance
rate = 20.							# ROS publication rate motor speed
training_length = 100000			# Lenth of training procedure (1 step ~ 50 ms)
