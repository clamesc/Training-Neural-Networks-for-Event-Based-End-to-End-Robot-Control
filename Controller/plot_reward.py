#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt

# Plot DQN reward
# Fig. 4.7

def normpdf(x, mean=0, sd=0.15):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

res = 0.01
b = 0.6
g = int(b / res)

y1 = [(x-g)*res for x in range(g*2)]
y2 = [(g-x)*res for x in range(g*2)]
x = [(x-g)*res for x in range(g*2)]
plt.plot(x,y1, color='b')
plt.plot(x,y2, color='g')
plt.axvline(x=0.25, linestyle='--', color='0.75')
plt.axvline(x=-0.25, linestyle='--', color='0.75')
plt.axvline(x=0.2, linestyle='--', color='r')
plt.axvline(x=-0.2, linestyle='--', color='r')
plt.axhline(y=0., linewidth=0.5, linestyle='-', color='k')
plt.xticks([-0.5,-0.25,0.,0.25,0.5])
plt.show()