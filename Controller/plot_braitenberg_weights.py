#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Static network weights of Braitenberg vehicle controller
# Fig. 5.5

weights_l = np.array(	[[250., 250., 250., 500., 0., 0., 0., 0.],
						[250., 250., 500., 1000., 0., 0., 0., 0.],
						[250., 500., 1000., 1500., 0., 0., 0., 0.],
						[500., 1000., 1500., 2000., 0., 0., 0., 0.]])

weights_r = np.array(	[[0., 0., 0., 0., 500., 250., 250., 250.],
						[0., 0., 0., 0., 1000., 500., 250., 250.],
						[0., 0., 0., 0., 1500., 1000., 500., 250.],
						[0., 0., 0., 0., 2000., 1500., 1000., 500.]])

fig = plt.figure(figsize=(6,6))

ax1 = plt.subplot(211)
plt.title('Left Weights')
plt.imshow(weights_l, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_l):
	ax1.text(i,j,int(label),ha='center',va='center')

ax2 = plt.subplot(212)
plt.title('Right Weights')
plt.imshow(weights_r, alpha=0.5)
plt.axis('off')
for (j,i),label in np.ndenumerate(weights_r):
	ax2.text(i,j,int(label),ha='center',va='center')

fig.tight_layout()
plt.show()