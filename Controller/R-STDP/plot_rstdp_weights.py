#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt

# R-STDP weights learned
# Fig. 5.7, Fig. 5.8, Fig. 5.10

path = "/data/rstdp/scenario x"
h5f = h5py.File(path + '/rstdp_data.h5', 'r')

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
print w_l.shape
weights_l = np.flipud(w_l[-1].T)
weights_r = np.flipud(w_r[-1].T)

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