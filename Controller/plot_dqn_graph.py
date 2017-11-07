#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt

# DQN training progress figure (e.g. Fig. 5.3 + Fig. 5.4)
# CSV files downloaded from tensorboard

path = '/data/dqn/scenario x/'
data1 = np.genfromtxt(path + 'episodes.csv', delimiter=',')
data2 = np.genfromtxt(path + 'rewards.csv', delimiter=',')
x1 = data1[1:,1]
y1 = data1[1:,2]
x2 = data2[1:,1]
y2 = data2[1:,2]

fig = plt.figure(figsize=(7,4))

ax1 = plt.subplot(211)
ax1.set_ylabel('Time Steps')
plt.grid(linestyle=':')
ax1.set_xlim((0,max(x1)))
ax1.set_ylim([0,1200])
plt.plot(x1,y1,linewidth=1.0)

ax2 = plt.subplot(212, sharex=ax1)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward')
ax2.set_xlim((0,max(x1)))
ax2.set_ylim([0,2900])
plt.grid(linestyle=':')
plt.plot(x2,y2,linewidth=1.0)

fig.tight_layout()
plt.show()
