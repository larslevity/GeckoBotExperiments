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
versions = ['vS11']
styles = ['-']
rots = [9]
len_legs = [9]
len_tors = [10]


ls = {version: style for style, version in zip(styles, versions)}
rotation = {version: rot for rot, version in zip(rots, versions)}
len_leg = {version: val for val, version in zip(len_legs, versions)}
len_tor = {version: val for val, version in zip(len_tors, versions)}
color_eps = {version: val for val, version in zip(
        ['mediumpurple', 'darkmagenta'], versions)}


clb = {}

for version in versions:
    clb[version] = {}
    dirpath = version + '/'
    sets = load.get_csv_set(dirpath)
    db = ev.load_data_pathPlanner(dirpath, sets)

    col = ev.get_marker_color()

    alpha = {}
    pressure = {}
    reference = {}

    for exp in range(len(db)):
        for axis in range(6):
            alpha[axis] = np.array(db[exp]['aIMG{}'.format(axis)])
            pressure[axis] = np.array(db[exp]['p{}'.format(axis)])
            reference[axis] = np.array(db[exp]['r{}'.format(axis)])
        # remove all nans:
        pressure[axis] = pressure[axis][~np.isnan(alpha[axis])]
        reference[axis] = reference[axis][~np.isnan(alpha[axis])]
        alpha[axis] = alpha[axis][~np.isnan(alpha[axis])]

#    save.save_as_tikz('pics/'+version+'_track.tex', **kwargs)

    # %% Pressure-alpha relation
    for axis in range(6):
        deg = 5

        # only vals with ref!=0
        idx = reference[axis] != 0
        # only vals with alp != nan
        for iidx in enumerate(idx):
            if np.isnan(alpha[axis][iidx]):
                idx[iidx] = False
        # only vals with ref[i] > ref[i-1]
        for iidx in range(len(idx))[1:]:
            if reference[axis][iidx] <= reference[axis][iidx-1]:
                idx[iidx] = False
        # only vals with alpha[i] > 0
        for iidx in enumerate(idx):
            if alpha[axis][iidx] < 0:
                idx[iidx] = False

        alp_f = alpha[axis][idx]
        p_f = pressure[axis][idx]

        # shift
        shift = 0  # -min(alp_f)
        alp_f = alp_f + shift

        coef = np.polyfit(alp_f, p_f, deg)
        clb[version][axis] = list(coef)

        coef_s = ['%1.3e' % c for c in coef]
        print('Actuator %s:\t%s' % (axis, coef_s))
        poly = np.poly1d(coef)
        alp = np.linspace(min(alp_f)-2, max(alp_f)+2, 100)

        plt.figure('CLB'+str(axis))
        plt.plot(alpha[axis]+shift, pressure[axis], ':',
                 label='measurements')
        plt.plot(alp_f, p_f, 'o', label='used measurements')
        plt.plot(alp, poly(alp), '-x', label='fitted')

        plt.grid()
        plt.xlim((-20, 150))
        plt.ylim((-.1, 1.3))
        plt.xlabel(r'bending angle $\alpha$ ($^\circ$)')
        plt.ylabel(r'pressure $p$ (bar)')
        plt.legend(loc='lower right')
        save.save_as_tikz('pics/'+version+'_clb_'+str(axis)+'.tex')

#        plt.figure(1)
#        plt.plot(pressure[axis], alpha[axis])


plt.show()

print(clb)
