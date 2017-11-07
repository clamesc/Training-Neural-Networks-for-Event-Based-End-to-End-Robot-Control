#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from environment import VrepEnvironment
from parameters import *

# Performance graph scenario 2
# Fig. 5.12

env = VrepEnvironment(motor_speed, turn_speed, resolution, reset_distance, pub_rate, dvs_queue, resize_factor, crop)

x1 = env.d1_outer
x2 = x1 + env.d2_outer
x3 = x2 + env.d3_outer
x4 = x3 + env.d4_outer
x5 = x4 + env.d5_outer
x6 = x5 + env.d6_outer

lim = 0.23

path1 = '/data/dqn/scenario 2'
path3 = '/data/rstdp/scenario 2'

h5f = h5py.File(path1 + '/dqn_performance_data.h5', 'r')
h5f2 = h5py.File(path1 + '/snn_performance_data.h5', 'r')
h5f4 = h5py.File(path3 + '/rstdp_performance_data.h5', 'r')

distance = np.array(h5f['distance'], dtype=float)
position = np.array(h5f['position'], dtype=float)
distance2 = np.array(h5f2['distance'], dtype=float)
position2 = np.array(h5f2['position'], dtype=float)
distance4 = np.array(h5f4['distance'], dtype=float)
position4 = np.array(h5f4['position'], dtype=float)

fig1 = plt.figure(figsize=(9,3.7))

gs = gridspec.GridSpec(3, 2, width_ratios=[6, 1]) 

ax1 = plt.subplot(gs[0])
plt.axhline(y=0., linewidth=0.5, color='0.')
plt.axvline(x=x1, linestyle='--', color='0.75')
plt.axvline(x=x2, linestyle='--', color='0.75')
plt.axvline(x=x3, linestyle='--', color='0.75')
plt.axvline(x=x4, linestyle='--', color='0.75')
plt.axvline(x=x5, linestyle='--', color='0.75')
plt.plot(position,distance, color='r')
ax1.set_xlim((0,x6))
ax1.set_ylim((-lim,lim))
ax1.set_xticklabels([])
ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
ax9 = ax1.twiny()
ax9.set_xlabel('Section')
ax9.set_xlim((0.,x6))
ax9.set_xticks([0.5*env.d1_outer,x1+0.5*env.d2_outer,x2+0.5*env.d3_outer,x3+0.5*env.d4_outer,x4+0.5*env.d5_outer,x5+0.5*env.d6_outer])
ax9.set_xticklabels(['A','B','C','D','E','F'])
ax9.tick_params(axis='both', which='both', direction='in', bottom=False, top=False, left=False, right=False)

ax3 = plt.subplot(gs[2])
plt.axhline(y=0., linewidth=0.5, color='0.')
plt.axvline(x=x1, linestyle='--', color='0.75')
plt.axvline(x=x2, linestyle='--', color='0.75')
plt.axvline(x=x3, linestyle='--', color='0.75')
plt.axvline(x=x4, linestyle='--', color='0.75')
plt.axvline(x=x5, linestyle='--', color='0.75')
plt.plot(position2,distance2, color='m')
ax3.set_xlim((0,x6))
ax3.set_ylim((-lim,lim))
ax3.set_ylabel('Distance to Lane-Center [m]')#, position=(0.,0.))
ax3.set_xticklabels([])
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

ax7 = plt.subplot(gs[4])
plt.axhline(y=0., linewidth=0.5, color='0.')
plt.axvline(x=x1, linestyle='--', color='0.75')
plt.axvline(x=x2, linestyle='--', color='0.75')
plt.axvline(x=x3, linestyle='--', color='0.75')
plt.axvline(x=x4, linestyle='--', color='0.75')
plt.axvline(x=x5, linestyle='--', color='0.75')
plt.plot(position4,distance4, color='g')
ax7.set_xlim((0,x6))
ax7.set_ylim((-lim,lim))
ax7.set_xlabel('Position [m]')
ax7.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

s1 = abs(distance).mean()
s2 = abs(distance2).mean()
s4 = abs(distance4).mean()

b = [x*0.01 for x in range(-23,24)]
ax2 = plt.subplot(gs[1])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_ylim((-lim,lim))
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
ax2.set_title('DQN \ne = '+str('{:4.3f}'.format(s1)), loc='left', size='medium', position=(1.1,0.2))
plt.axhline(y=0, linewidth=0.5, color='0.')
plt.hist(distance, bins=b, normed=True, color='r', linewidth=2, orientation=u'horizontal')

ax4 = plt.subplot(gs[3], sharex=ax2)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_ylim((-lim,lim))
ax4.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
ax4.set_title('DQN-SNN \ne = '+str('{:4.3f}'.format(s2)), loc='left', size='medium', position=(1.1,0.2))
plt.axhline(y=0, linewidth=0.5, color='0.')
plt.hist(distance2, bins=b, normed=True, color='m', linewidth=2, orientation=u'horizontal')

ax8 = plt.subplot(gs[5], sharex=ax2)
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.set_ylim((-lim,lim))
ax8.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
ax8.set_title('R-STDP \ne = '+str('{:4.3f}'.format(s4)), loc='left', size='medium', position=(1.1,0.2))
plt.axhline(y=0, linewidth=0.5, color='0.')
ax8.set_xlabel('Histogram')
plt.hist(distance4, bins=b, normed=True, color='g', linewidth=2, orientation=u'horizontal')

plt.subplots_adjust(wspace=0., hspace=0.1, right=0.89, left=0.08)
plt.show()