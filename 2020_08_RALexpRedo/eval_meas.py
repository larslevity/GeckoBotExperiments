#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:43:59 2020

@author: ls
"""

import matplotlib.pyplot as plt
import numpy as np
from tikzplotlib import save as tikz_save

import utils as uti


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


import plot_fun as pf

from Src import calibration



# %%

modes = [
        'straight_1',
        'straight_2',
        'straight_3',
#        'curve_1',
#        'curve_2',
#        'curve_3',
        ]


version = 'vS12'


# %%

n_steps = {}

MU_DX = {}
SIG_DX = {}

for idx in range(6):
    MU_DX[idx] = {}
    SIG_DX[idx] = {}

MU_DEPS = {}
SIG_DEPS = {}
MEAS = {}
GAITS_cor = []
GAITS_raw = []

DX = np.zeros((len(modes)))
DY = np.zeros((len(modes)))
DXSIG = np.zeros((len(modes)))
DYSIG = np.zeros((len(modes)))
DEPS = np.zeros((len(modes)))
DEPSSIG = np.zeros((len(modes)))


n_cyc = 1
exp_idx = 0

len_leg, len_tor = calibration.get_len(version)
ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]


for mode_idx, mode in enumerate(modes):
    n_steps[mode] = {}
    # %% ### Load Data

    dirpath = mode + '/'

    sets = uti.get_csv_set(dirpath)
    db = uti.load_data(dirpath, sets)

    print('find idxes....')
    POSE_IDX = uti.find_poses_idx(db, neighbors=10)
    n_steps[mode] = [len(idx)-1 for idx in POSE_IDX]

    # %% ### Track of feet:
    print('plot track....')
    pf.plot_track(db, POSE_IDX, 'L', mode)



    # %% DX/DY
    dx_mean = []
    dy_mean = []
    for idx in range(2, n_steps[mode][exp_idx]-2, 2):  # drop first and last 2 poses
        _, eps_init, fpos, _, _, _ = \
            uti.extract_measurement(db[exp_idx], POSE_IDX[exp_idx][idx])
        x1_init = np.r_[fpos[0][1], fpos[1][1]]

        _, eps, fpos, _, _, _ = uti.extract_measurement(
                db[exp_idx], POSE_IDX[exp_idx][idx+2])
        x1 = np.r_[fpos[0][1], fpos[1][1]]
        travel = uti.rotate(x1 - x1_init, np.deg2rad(-eps_init))
        dx_mean.append(travel[0])
        dy_mean.append(travel[1])

    DX[mode_idx] = np.nanmean(dx_mean)
    DY[mode_idx] = np.nanmean(dy_mean)

    DXSIG[mode_idx] = np.nanstd(dx_mean)
    DYSIG[mode_idx] = np.nanstd(dy_mean)



    # %% EPS
    print('plot eps....')
    plt.figure('eps'+'mode')
    # drop first and last cycle for evaluation of eps
    eps, t = pf.plot_eps(db, [POSE_IDX[0][2:-2]],
                         mode, version, save_as_tikz=False)
    deps = np.nanmean(np.diff(eps))*2  # mean deps/cycle
    deps_sig = np.nanstd(np.diff(eps))*2  # sig deps/cycle
    DEPS[mode_idx] = deps
    DEPSSIG[mode_idx] = deps_sig




# %% ABSOLUTE PLOTS
# %% DEPS

kwargs = {'extra_axis_parameters': {'anchor=origin',
                                    'axis line style={draw=none}',
                                    'xtick style={draw=none}',
                                    'height=6cm',
                                    'width=10cm'}}

colors = {
        '$w_{\\varphi}=10$': 'blue',
        '$w_{\\varphi}=1$': 'red',
        '$w_{\\varphi}=0.1$': 'orange',
        }



def mapped(modes):
    mapped = []
    for mode in modes:
        if mode[-1] == '1':
            val = '10'
        elif mode[-1] == '2':
            val = '1'
        elif mode[-1] == '3':
            val = '0.1'
        mapped.append('$w_{\\varphi}=%s$' % val)
    return mapped


mu = {mode: [abs(DEPS[mode_idx]), abs(DX[mode_idx])/ell_n[2]*100, abs(DY[mode_idx])/ell_n[2]*100] for mode_idx, mode in enumerate(mapped(modes))}
sig = {mode: [DEPSSIG[mode_idx], DXSIG[mode_idx]/ell_n[2]*100, DYSIG[mode_idx]/ell_n[2]*100] for mode_idx, mode in enumerate(mapped(modes))}

labels = ['$\Delta \epsilon ~ (^\circ)$', '$\Delta x / \ell_n ~ (\%)$', '$\Delta y / \ell_n ~ (\%)$']

ax = uti.barplot(mu, mapped(modes), labels, colors,
                 sig, num='error-deps')

ax.set_ylabel('')
ax.set_xlabel('')
ax.grid(True, axis='y')
ax.set_ylim((0, 150))



tikz_save('Out/eval_' + mode[:-2] + '.tex', **kwargs)
#