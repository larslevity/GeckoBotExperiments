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

modes = [
        'straight_1',
#        'curve_1',
        ]

version = 'v40'

n_steps = {}
for mode in modes:
    n_steps[mode] = {}
    # %% ### Load Data

    dirpath = mode + '/'

    sets = load.get_csv_set(dirpath)
    db = ev.load_data_pathPlanner(dirpath, sets, version=version)
    POSE_IDX = uti.ieee_find_poses_idx(db, neighbors=10)
    n_steps[mode] = [len(idx)-1 for idx in POSE_IDX]

    # %% ### Track of feet:
    pf.plot_track(db, POSE_IDX, 'L', mode, save_as_tikz=False)

    # %% Plot DEPS
    mat = pf.plot_deps(db, POSE_IDX, 'L', mode, save_as_tikz=False)

    # %%
    ALPERR, PERR, EPSERR, alpsig, psig, epsig = \
        uti.calc_errors(db, POSE_IDX, version , runs=1, predict_poses=2)
    # %%
    col = pf.get_marker_color()
    plt.figure('PredictionErrors-P')
    markers = [4]
    for idx in markers:
        plt.plot(PERR[idx], label='marker {}'.format(idx), color=col[idx])
        mu, sig = PERR[idx], psig[idx]
        plt.fill_between(range(len(mu)), mu+sig, mu-sig,
                         facecolor=col[idx], alpha=0.5)
    plt.xlabel('Step count')
    plt.ylabel('Prediction Error of Position |p_m - p_p|')

# %%
    plt.figure('PredictionErrors-ALP')
    for idx in range(len(ALPERR)):
        plt.plot(ALPERR[idx], label='marker {}'.format(idx))
    plt.xlabel('Step count')
    plt.ylabel('Prediction Error of Angle |a_m - a_p|')

    plt.figure('PredictionErrors-EPS')
    plt.plot(EPSERR, label='eps')
    plt.xlabel('Step count')
    plt.ylabel('Prediction Error of EPS |e_m - e_p|')


plt.show()