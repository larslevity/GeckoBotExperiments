# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:22:05 2020

@author: AmP
"""


import matplotlib.pyplot as plt
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import save as my_save
from Src import calibration
from Src import inverse_kinematics
from Src import roboter_repr
from Src import load


import obstacle_utils as uti
import obstacle_plot_methods as pf

version = 'vS11'
exp = 'exp_0'


needed_steps = {}


# %% Load Data
dirpath = exp + '/'

sets = load.get_csv_set(dirpath)
db = uti.load_data(dirpath, sets)
POSE_IDX = uti.find_poses_idx(db, neighbors=10)
for i in range(len(db)):
    needed_steps[i] = len(POSE_IDX[i])-1

# %% Track
pf.plot_track(db, POSE_IDX, save_as_tikz=False, tags=[0,2,3,4,5],
              show_cycles=0, linewidth=1, alpha=.5)

pf.plot_track(db, POSE_IDX, save_as_tikz=False, tags=[1],
              show_cycles=0, linewidth=4)


plt.plot([0], [0], marker='o', color='k', mfc='orange', markersize=10)
plt.text(2, 2, str('start'), fontsize=30)

XREF = [(20, 30), (-45, 50), (20, 95)]
for i, (x, y) in enumerate(XREF):
    plt.plot(x, y, marker='o', color='black', markersize=12, mfc='red')
    plt.text(x, y, 'goal '+str(i+1), fontsize=30)


plt.grid()
plt.xticks([-45, 0, 20.001], ['-45', '0', '20'])
plt.yticks([0.001, 30, 50, 95], ['0', '30', '50', '95'])
plt.ylim((-25, 120))
plt.xlim((-25, 110))
plt.axis('scaled')

plt.grid()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
#plt.grid()

kwargs = {'extra_axis_parameters':
          {'x=.1cm', 'y=.1cm', 'anchor=origin', 'xmin=-55',
           'xmax=37','axis line style={draw opacity=0}',
           'ymin=-20, ymax=105', 'tick pos=left',}}

my_save.save_plt_as_tikz('Out/exp-gait.tex',
    additional_tex_code='\\draw[color0, line width=1mm, -latex] (0,0)--(0,1);',
    **kwargs)



# %% Q1Q2

Q1_all = {}
Q2_all = {}

for exp_idx in range(len(db)):
    Q1_all[exp_idx] = []
    Q2_all[exp_idx] = []
    
    for pidx in POSE_IDX[exp_idx]:
        Q1_all[exp_idx].append(abs(db[exp_idx]['aIMG6'][pidx]))
        Q2_all[exp_idx].append(db[exp_idx]['aIMG7'][pidx])


# %%


plt.figure('Q1Q2')
ax1 = plt.gca()
ax2 = ax1.twinx()


maxstep = max([len(Q1_all[exp_idx]) for exp_idx in range(len(db))])


Q1_mean = []
Q2_mean = []
Q1_std = []
Q2_std = []

for step in range(maxstep):
    q1_i = []
    q2_i = []
    for exp_idx in range(len(db)):
        try:
            q1 = Q1_all[exp_idx][step]
            q2 = Q2_all[exp_idx][step]
        except IndexError:
            q1, q2 = np.nan, np.nan
        q1_i.append(q1)
        q2_i.append(q2)
    Q1_mean.append(np.nanmean(np.array(q1_i)))
    Q1_std.append(np.nanstd(np.array(q1_i)))
    Q2_mean.append(np.nanmean(np.array(q2_i)))
    Q2_std.append(np.nanstd(np.array(q2_i)))
Q1_mean = np.array(Q1_mean)
Q2_mean = np.array(Q2_mean)
Q1_std = np.array(Q1_std)
Q2_std = np.array(Q2_std)

# find mean idx of reached goals
reached_goal = [0]
for idx, (q1, std) in enumerate(zip(Q1_mean, Q1_std)):
    if idx == 0 or idx == len(Q1_mean)-1:
        continue
    else:
        last_q1, last_std = Q1_mean[idx-1], Q1_std[idx-1]
        next_q1, next_std = Q1_mean[idx+1], Q1_std[idx+1]
        if (q1-std < last_q1-last_std and q1-std < next_q1-next_std
            and q1-std < 49):
            reached_goal.append(idx)


#for exp_idx in range(len(db)):
#    Q1 = Q1_all[exp_idx]
#    Q2 = Q2_all[exp_idx]
#
#    ax1.plot(Q1, 'b')
#    ax2.plot(Q2, 'r')

ax1.plot(Q1_mean, 'b')
ax1.fill_between(range(maxstep), Q1_mean+Q1_std, Q1_mean-Q1_std,
                 facecolor='blue', alpha=.5)

ax2.plot(Q2_mean, 'r')
ax2.fill_between(range(maxstep), Q2_mean+Q2_std, Q2_mean-Q2_std,
                 facecolor='red', alpha=.5)
    #n = 0
    #for idx, d in enumerate(DIST):
    #    if d < 10:
    #        ax1.plot([idx, idx], [43, 95], 'gray')
    #        ax1.text(idx-1, 43, 'Reached {}'.format(n),  horizontalalignment='right')
    #        n += 1

ax1.set_xticks(reached_goal)
ax1.grid()

ax2.set_ylim(-.7, .7)
ax2.set_yticks([-.5, 0, .5])

ax1.set_yticks([50, 70, 90])
ax1.set_ylim(40, 100)
ax1.set_ylabel('step length q1 (deg)', color='blue')
ax2.set_ylabel('steering q2 (1)', color='red')
ax1.set_xlabel('number of steps (1)')
ax2.tick_params('y', colors='red')
ax1.tick_params('y', colors='blue')

my_save.save_plt_as_tikz('Out/Q1Q2.tex')