# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:29:53 2019

@author: AmP
"""

# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from matplotlib2tikz import save as tikz_save

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval_pathPlanner as ev
from Src import load
from Src import plot_fun_pathPlanner as pf
from Src import save as my_save



import utils as uti

# %%

modes = [
#        'straight_1',
#        'straight_2',
        'straight_3',
#        'curve_1',
#        'curve_2',
#        'curve_3',
        ]


settings = {
    'curve_1': {'nexps': [0, 1, 2, 3, 4],
                'startidx': 4},
    'curve_2': {'nexps': [0, 1, 2, 3, 4],
                'startidx': 2},
    'curve_3': {'nexps': [0, 1, 2, 3, 4],
                'startidx': 4},
    'straight_1': {'nexps': [0, 1, 2, 3],
                   'startidx': 4},
    'straight_2': {'nexps': [0, 1],
                   'startidx': 0},
    'straight_3': {'nexps': [0, 1, 2],
                   'startidx': 8},
        }


version = 'v40'

n_steps = {}
for mode in modes:
    n_steps[mode] = {}
    # %% ### Load Data

    dirpath = mode + '/'

    sets = load.get_csv_set(dirpath)
    db = ev.load_data_pathPlanner(dirpath, sets, version=version)
    
    print('find idxes....')
    POSE_IDX = uti.ieee_find_poses_idx(db, neighbors=10)
    n_steps[mode] = [len(idx)-1 for idx in POSE_IDX]

    # %% ### Track of feet:
#    print('plot track....')
#    pf.plot_track(db, POSE_IDX, 'L', mode, save_as_tikz=False)


    # %%
    predict_poses = 4
    start_idx = settings[mode]['startidx']
    nexps = settings[mode]['nexps']

    print('calc predictions errors....')
    
    ALPERR, PERR, EPSERR, alpsig, psig, epsig, gait_predicted = \
        uti.calc_errors(db, POSE_IDX, version, mode=mode,
                        nexps=nexps, predict_poses=predict_poses,
                        start_idx=start_idx)
    # %%
    col = pf.get_marker_color()
    plt.figure('PredictionErrorsP')
    markers = [1]
    for idx in markers:
        plt.plot(PERR[idx], label='marker {}'.format(idx), color=col[idx])
        mu, sig = PERR[idx], psig[idx]
        plt.fill_between(range(len(mu)), mu+sig, mu-sig,
                         facecolor=col[idx], alpha=0.5)
    plt.ylabel('Prediction Error of Position |p_m  p_p|', color=col[idx])
    plt.gca().tick_params('y', colors=col[idx])


    ax = plt.gca().twinx()
    ax.plot(EPSERR, label='eps', color='purple')
    ax.fill_between(range(len(EPSERR)), EPSERR+epsig, EPSERR-epsig,
                         facecolor='purple', alpha=0.5)
    ax.set_ylabel('Prediction Error of EPS |e_m  e_p|', color='purple')
    ax.tick_params('y', colors='purple')

    plt.xlabel('Step count')
    ax.set_xticks([int(x) for x in range(predict_poses+1)])
    plt.grid()

    plt.savefig('Out/PredictionERROR_'+str(mode)+'_startIDX_'+str(start_idx)+'.png',
                dpi=300)
    
#    fdir = path.dirname(path.abspath(__file__))
    my_save.save_plt_as_tikz('/Out/' + mode + '_error.tex')
#    tikz_save(mode + '_error.tex')


# %%

    gait_predicted.save_as_tikz(mode+'_gait')


# %%
    plt.figure('PredictionErrors-ALP')
    for idx in range(len(ALPERR)):
        plt.plot(ALPERR[idx], label='marker {}'.format(idx))
    plt.xlabel('Step count')
    plt.ylabel('Prediction Error of Angle |a_m - a_p|')

  


plt.show()