# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:38:25 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval as ev
from Src import save
from Src import load
from Src import kin_model
from Src import predict_pose as pp
from Src import plot_fun as pf


# %% LOAD
versions = ['big', 'small']
styles = ['-', '-']
rots = [5.5, 9]
len_legs = [13, 9]
len_tors = [14, 10]

DRAW_SHOTS = False

ls = {version: style for style, version in zip(styles, versions)}
rotation = {version: rot for rot, version in zip(rots, versions)}
len_leg = {version: val for val, version in zip(len_legs, versions)}
len_tor = {version: val for val, version in zip(len_tors, versions)}
color_eps = {version: val for val, version in zip(
        ['mediumpurple', 'darkmagenta'], versions)}
color_v = {version: val for val, version in zip(
        ['purple', 'orange'], versions)}


fig, ax = plt.subplots(nrows=2, ncols=1, num='Alpha during cycle', sharex=True)

for version in versions:
    dirpath = version + '/'
    sets = load.get_csv_set(dirpath)
    db, cyc = ev.load_data(dirpath, sets)
    prop = pf.calc_prop(db, cyc)

    # %% # correction of epsilon
    rotate = rotation[version]
    for exp in range(len(db)):
        eps0 = db[exp]['eps'][cyc[exp][1]]
#        eps0 = 0
        for marker in range(6):
            X = db[exp]['x{}'.format(marker)]
            Y = db[exp]['y{}'.format(marker)]
            X, Y = ev.rotate_xy(X, Y, -eps0+rotate)
            db[exp]['x{}'.format(marker)] = X
            db[exp]['y{}'.format(marker)] = Y
        db[exp]['eps'] = ev.add_offset(db[exp]['eps'], -eps0+rotate)

    # %% Plot alpha
    for axis in [5]:
        cycs = [1,2,3,4]
        alp, siga = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), cycs)
        p, sigp = ev.calc_mean_of_axis(db, cyc, 'p{}'.format(axis), cycs)
        t, sigt = ev.calc_mean_of_axis(db, cyc, 'time', cycs)

        # downsample for tikz
        prop = .8
        alp = ev.downsample(alp, proportion=prop)
        p = ev.downsample(p, proportion=prop)
        t_s = ev.rm_offset(ev.downsample(t, proportion=prop))
        siga = ev.downsample(siga, proportion=prop)
        sigp = ev.downsample(sigp, proportion=prop)

        ax[1].plot(t_s, alp, '-', color=color_v[version])
        ax[1].fill_between(t_s, alp+siga, alp-siga, facecolor=color_v[version], alpha=0.5)
        ax[0].plot(t_s, p, '-', color=color_v[version])
        ax[0].fill_between(t_s, alp+sigp, alp-sigp, facecolor=color_v[version], alpha=0.5)






# %%

ax[0].grid()
ax[1].grid()
ax[0].set_ylim((0, 1.2))

ax[0].set_ylabel(r'applied pressure $p$ (bar)')
ax[1].set_ylabel(r'bending angle $\alpha$ ($^\circ$)')
ax[1].set_xlabel(r'time $t$ (s)')

save.save_as_tikz('pics/pressure-alpha.tex')


plt.show()
