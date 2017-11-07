#!/usr/bin/env python

import numpy as np

path = './data/session_xyz'				# Path for storing data

# Input image
dvs_resolution = [128,128]				# Original DVS frame resolution
crop_top = 40							# Crop at the top
crop_bottom = 24						# Crop at the bottom
resolution = [8,4]						# Resolution of reduced image

# Network parameters
sim_time = 50.0							# Length of network simulation during each step in ms
t_refrac = 2.							# Refractory period
time_resolution = 0.1					# Network simulation time resolution
iaf_params = {}							# IAF neuron parameters
poisson_params = {}						# Poisson neuron parameters
max_poisson_freq = 1/(t_refrac*1e-3)	# Maximum Poisson firing frequency for n_max
max_spikes = 10.						# number of events during each step for maximum poisson frequency

# Steering wheel model
v_max = 1.5								# Maximum speed
v_min = 1.								# Minimum speed
turn_factor= 0.5						# Factor controls turn radius
turn_pre = 0							# Initial turn speed
v_pre = v_max							# Initial speed
n_max = sim_time//t_refrac				# Maximum input activity

# Other
reset_distance = 0.5					# Reset distance
rate = 20.								# ROS publication rate motor speed

# Static netork weights
weights_l = np.array(	[[250., 250., 250., 500.],
						[250., 250., 500., 1000.],
						[250., 500., 1000., 1500.],
						[500., 1000., 1500., 2000.]])

weights_r = np.array(	[[500., 250., 250., 250.],
						[1000., 500., 250., 250.],
						[1500., 1000., 500., 250.],
						[2000., 1500., 1000., 500.]])