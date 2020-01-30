# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:22:05 2020

@author: AmP
"""


import matplotlib.pyplot as plt
import matplotlib.patches as pat
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
pf.plot_track(db, POSE_IDX, save_as_tikz=False, tags=[0,1,2,3,4,5])

kwargs = {'extra_axis_parameters':
    {'x=.1cm', 'y=.1cm', 'anchor=origin'}}
my_save.save_plt_as_tikz('Out/Track.tex', **kwargs)



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

for exp_idx in range(len(db)):
    Q1 = Q1_all[exp_idx]
    Q2 = Q2_all[exp_idx]

    ax1.plot(Q1, 'b')
    ax2.plot(Q2, 'r')

    #n = 0
    #for idx, d in enumerate(DIST):
    #    if d < 10:
    #        ax1.plot([idx, idx], [43, 95], 'gray')
    #        ax1.text(idx-1, 43, 'Reached {}'.format(n),  horizontalalignment='right')
    #        n += 1

ax1.grid()
ax2.set_ylim(-.7, .7)
ax2.set_yticks([-.5, 0, .5])
ax1.set_ylim(40, 100)
ax1.set_ylabel('step length q1 (deg)', color='blue')
ax2.set_ylabel('steering q2 (1)', color='red')
ax1.set_xlabel('steps (1)')
ax2.tick_params('y', colors='red')
ax1.tick_params('y', colors='blue')

my_save.save_plt_as_tikz('Out/Q1Q2.tex')