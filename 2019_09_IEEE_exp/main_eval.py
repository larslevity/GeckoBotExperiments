# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:29:53 2019

@author: AmP
"""

# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval_pathPlanner as ev
from Src import load
from Src import plot_fun_pathPlanner as pf
import utils as uti

# %%

modes = [
#        'straight_3',
        'curve_1',
        ]

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
    start_idx = 4
    print('calc predictions errors....')
    
    ALPERR, PERR, EPSERR, alpsig, psig, epsig = \
        uti.calc_errors(db, POSE_IDX, version, mode=mode,
                        nexps=None, predict_poses=predict_poses,
                        start_idx=start_idx)
    # %%
    col = pf.get_marker_color()
    plt.figure('PredictionErrors-P')
    markers = [1]
    for idx in markers:
        plt.plot(PERR[idx], label='marker {}'.format(idx), color=col[idx])
        mu, sig = PERR[idx], psig[idx]
        plt.fill_between(range(len(mu)), mu+sig, mu-sig,
                         facecolor=col[idx], alpha=0.5)
    plt.ylabel('Prediction Error of Position |p_m - p_p|', color=col[idx])
    plt.gca().tick_params('y', colors=col[idx])


    ax = plt.gca().twinx()
    ax.plot(EPSERR, label='eps', color='purple')
    ax.fill_between(range(len(EPSERR)), EPSERR+epsig, EPSERR-epsig,
                         facecolor='purple', alpha=0.5)
    ax.set_ylabel('Prediction Error of EPS |e_m - e_p|', color='purple')
    ax.tick_params('y', colors='purple')

    plt.xlabel('Step count')
    ax.set_xticks([int(x) for x in range(predict_poses+1)])
    plt.grid()

    plt.savefig('Out/PredictionERROR_'+str(mode)+'_startIDX_'+str(start_idx)+'.png',
                dpi=300)


# %%
    plt.figure('PredictionErrors-ALP')
    for idx in range(len(ALPERR)):
        plt.plot(ALPERR[idx], label='marker {}'.format(idx))
    plt.xlabel('Step count')
    plt.ylabel('Prediction Error of Angle |a_m - a_p|')

  


plt.show()