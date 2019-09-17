# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval_pathPlanner as ev
from Src import load
from Src import plot_fun_pathPlanner as pf


modes = [
#        'without_eps_correction_x1_50',
        'eps_corrected_x1_50',
#        'without_eps_correction_x1_70',
#        'eps_corrected_x1_70',
#        'without_eps_correction_x1_90',
#        'eps_corrected_x1_90',
        ]
#runs = ['L', 'SL', 'R', 'SR']
runs = ['R']


needed_steps = {}
for mode in modes:
    needed_steps[mode] = {}
    for run in runs:
        # %% ### Load Data

        dirpath = mode + '/' + run + '/'

        sets = load.get_csv_set(dirpath)
        db = ev.load_data_pathPlanner(dirpath, sets)
        POSE_IDX = ev.find_poses_idx(db, neighbors=10)
        needed_steps[mode][run] = [len(idx)-1 for idx in POSE_IDX]

        # %% ### Track of feet:
        pf.plot_track(db, POSE_IDX, run, mode, save_as_tikz=False)

        # %% Plot DEPS
        mat = pf.plot_deps(db, POSE_IDX, run, mode, save_as_tikz=False)


        # %%
        alp_err = ev.calc_errors(db, POSE_IDX)

        # %% play

#        for channel in range(6):
#            alpha = [db[0]['aIMG{}'.format(channel)][idx] for idx in POSE_IDX[0]]
#            
#            dalpha = [db[0]['aIMG{}'.format(channel)][idx] - db[0]['aIMG{}'.format(channel)][idx-1] for idx in POSE_IDX[0]]
#        plt.figure('PLAY')
#        plt.plot(db[0]['r2'])
#        plt.plot(db[0]['r3'])
#        vals = [db[0]['r3'][idx] for idx in POSE_IDX[0]]
#        plt.plot(POSE_IDX[0], vals, 'o')
# %% 

pf.plot_needed_steps(needed_steps, runs, modes, save_as_tikz=False)
plt.show()