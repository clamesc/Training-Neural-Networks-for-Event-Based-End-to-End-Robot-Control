#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# Input sample
# Fig.4.14

im = np.flipud(np.array([[0.,0.,3.,3.],
				[0.,0.,11.,5.],
				[0.,0.,8.,7.],
				[0.,0.,0.,9.],
				[0.,0.,0.,13.],
				[0.,12.,24.,3.],
				[25.,8.,0.,0.],
				[2.,0.,0.,0.]]).T)

plt.imshow(im, cmap='Greys')
plt.axis('off')
plt.show()