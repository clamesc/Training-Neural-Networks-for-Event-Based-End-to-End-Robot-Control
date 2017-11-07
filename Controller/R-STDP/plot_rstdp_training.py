#!/usr/bin/env python

import numpy as np
import h5py
from environment import *
import matplotlib.pyplot as plt
from matplotlib import gridspec

# R-STDP training progress
# Fig. 5.6, Fig. 5.9

env = VrepEnvironment()
yo1 = env.d1_outer
yo2 = yo1 + env.d2_outer
yo3 = yo2 + env.d3_outer
yo4 = yo3 + env.d4_outer
yo5 = yo4 + env.d5_outer
yo6 = yo5 + env.d6_outer
yi1 = env.d1_inner
yi2 = yi1 + env.d2_inner
yi3 = yi2 + env.d3_inner
yi4 = yi3 + env.d4_inner
yi5 = yi4 + env.d5_inner
yi6 = yi5 + env.d6_inner

path = "/data/rstdp/scenario x"
h5f = h5py.File(path + '/rstdp_data.h5', 'r')

xlim = 100000

w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)
e_o = np.array(h5f['e_o'], dtype=float)
e_i_o = np.array(h5f['e_i_o'], dtype=float)
e_i = np.array(h5f['e_i'], dtype=float)
e_i_i = np.array(h5f['e_i_i'], dtype=float)

fig = plt.figure(figsize=(7,8))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 2, 2]) 

ax1 = plt.subplot(gs[0])
ax1.set_ylabel('Termination Position [m]', position=(0.,0.))
plt.axhline(y=yo1, linestyle='--', color='0.75')
plt.axhline(y=yo2, linestyle='--', color='0.75')
plt.axhline(y=yo3, linestyle='--', color='0.75')
plt.axhline(y=yo4, linestyle='--', color='0.75')
plt.axhline(y=yo5, linestyle='--', color='0.75')
ax1.set_xlim((0,xlim))
ax1.set_ylim((0,yo6))
ax1.set_xticklabels([])
ax1.text(1000, 27.5, 'Outer Lane', color='0.4')
ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.plot(e_i_o,e_o, 'x', markersize=10.)
ax5 = ax1.twinx()
ax5.set_ylabel('Section')
ax5.set_ylim((0,yo6))
ax5.set_yticks([0.5*env.d1_outer, yo1+ 0.5*env.d2_outer,yo2+ 0.5*env.d3_outer,yo3+ 0.5*env.d4_outer,yo4+ 0.5*env.d5_outer,yo5+ 0.5*env.d6_outer])
ax5.set_yticklabels(['A','B','C','D','E','F'])
ax5.tick_params(axis='both', which='both', direction='in', bottom=False, top=False, left=False, right=False)

ax2 = plt.subplot(gs[1])
plt.axhline(y=yi1, linestyle='--', color='0.75')
plt.axhline(y=yi2, linestyle='--', color='0.75')
plt.axhline(y=yi3, linestyle='--', color='0.75')
plt.axhline(y=yi4, linestyle='--', color='0.75')
plt.axhline(y=yi5, linestyle='--', color='0.75')
ax2.set_xlim((0,xlim))
ax2.set_ylim((0,yi6))
ax2.set_xticklabels([])
ax2.text(1000, 25, 'Inner Lane', color='0.4')
ax2.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
plt.plot(e_i_i,e_i, 'x', markersize=10.)
ax6 = ax2.twinx()
ax6.set_ylabel('Section')
ax6.set_ylim((0,yi6))
ax6.set_yticks([0.5*env.d1_inner,yi1+0.5*env.d2_inner,yi2+0.5*env.d3_inner,yi3+0.5*env.d4_inner,yi4+0.5*env.d5_inner,yi5+0.5*env.d6_inner])
ax6.set_yticklabels(['C','B','A','F','E','D'])
ax6.tick_params(axis='both', which='both', direction='in', bottom=False, top=False, left=False, right=False)


ax3 = plt.subplot(gs[2])
ax3.set_ylabel('Weight', position=(0.,0.))
ax3.set_xlim((0,xlim))
ax3.set_ylim((0,2300))
ax3.set_xticklabels([])
ax3.text(1000, 2100, 'Left Motor', color='0.4')
ax3.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_l.shape[1]):
	for j in range(w_l.shape[2]):
		plt.plot(w_i, w_l[:,i,j])

ax4 = plt.subplot(gs[3], sharey=ax3)
ax4.set_xlim((0,xlim))
ax4.set_ylim((0,2300))
ax4.text(1000, 2100, 'Right Motor', color='0.4')
ax4.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
for i in range(w_r.shape[1]):
	for j in range(w_r.shape[2]):
		plt.plot(w_i, w_r[:,i,j])
ax4.set_xlabel('Simulation Time [1 step = 50 ms]')


fig.tight_layout()
plt.subplots_adjust(wspace=0., hspace=0.)
plt.show()