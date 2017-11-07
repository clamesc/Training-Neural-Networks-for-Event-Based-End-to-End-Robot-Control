#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt

# Plot eligibility trace example
# Fig. 2.2

spikes = [50,100,150,450,500,600]
trace = 0.
steps = 1000.
factor = 0.99
traces = []

for i in range(int(steps)):
	trace = factor * trace
	if i in spikes:
		trace = trace + 1
	traces.append(trace)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Time Steps')
ax.set_ylabel('Eligibility Trace')

plt.plot(range(int(steps)), traces)
plt.show()



