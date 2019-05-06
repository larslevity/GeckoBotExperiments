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

DRAW_SHOTS = False

ls = {version: style for style, version in zip(styles, versions)}
rotation = {version: rot for rot, version in zip(rots, versions)}
len_leg = {version: val for val, version in zip(len_legs, versions)}
len_tor = {version: val for val, version in zip(len_tors, versions)}
color_eps = {version: val for val, version in zip(
        ['mediumpurple', 'darkmagenta'], versions)}


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

    # ### eps during cycle
    eps, sige = ev.calc_mean_of_axis(db, cyc, 'eps', [1])
    t_, sigt = ev.calc_mean_of_axis(db, cyc, 'time', [1])
    t_ = np.array(ev.rm_offset(t_))
    t = t_/t_[-1]  # scale to length of 1
    # hack to remove high freq noise
    for idx in range(1, len(t)):
        if abs(eps[idx] - eps[idx-1]) > 1:
            eps[idx] = eps[idx-1]
        if abs(sige[idx] - sige[idx-1]) > 1:
            sige[idx] = sige[idx-1]

    eps_ds = ev.downsample(eps, prop)
    t_ds = ev.downsample(t, prop)
    sige_ds = ev.downsample(sige, prop)
    plt.figure('Epsilon corrected')
    plt.plot(t_ds, eps_ds, ls[version], color=color_eps[version])
    plt.fill_between(t_ds, eps_ds+sige_ds, eps_ds-sige_ds,
                     facecolor=color_eps[version], alpha=0.5)

    # %% Plot Track
    figtrack, axtrack = plt.subplots(subplot_kw=dict(aspect='equal'),
                                     num='Track of feet'+version)
    col = ev.get_marker_color()
    positions = [{}, {}]
    alpha = {}
    pressure = {}
    SIGXY = [{}, {}]

    for axis in range(6):
        x, sigx = ev.calc_mean_of_axis(db, cyc, 'x{}'.format(axis), [0, 1])
        y, sigy = ev.calc_mean_of_axis(db, cyc, 'y{}'.format(axis), [0, 1])

        xds = ev.downsample(x, prop)
        yds = ev.downsample(y, prop)
        sigxds = ev.downsample(sigx, prop)
        sigyds = ev.downsample(sigy, prop)

        axtrack.plot(xds, yds, ls[version], color=col[axis])
        for xx, yy, sigxx, sigyy in zip(xds, yds, sigxds, sigyds):
            if not np.isnan(xx):
                el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
                                 facecolor=col[axis], alpha=.3)
                axtrack.add_artist(el)

        a, siga = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), [0, 1])
        p, sigp = ev.calc_mean_of_axis(db, cyc, 'p{}'.format(axis), [0, 1])
        alpha[axis] = a
        pressure[axis] = p
        positions[0][axis] = x
        positions[1][axis] = y
        SIGXY[0][axis] = sigx
        SIGXY[1][axis] = sigy
    axtrack.set_xlabel(r'x position (cm)')
    axtrack.set_ylabel(r'y position (cm)')
    axtrack.set_xlim((-20, 35))
    axtrack.set_ylim((-20, 15))
    axtrack.grid()
    kwargs = {'extra_axis_parameters': {'x=.15cm', 'y=.15cm'}}
#    save.save_as_tikz('pics/'+version+'_track.tex', **kwargs)

    # %% Pressure-alpha relation
    for axis in range(6):
        deg = 5
        if axis in [2]:
            coef = np.polyfit(alpha[axis][460:725], pressure[axis][460:725], deg)
        elif axis in [3]:
            coef = np.polyfit(alpha[axis][700:-10], pressure[axis][700:-10], deg)
        else:
            coef = np.polyfit(alpha[axis][50:340], pressure[axis][50:340], deg)
        coef_s = ['%1.3e' % c for c in coef]
        print('Actuator %s:\t%s' % (axis, coef_s))
        poly = np.poly1d(coef)
        if axis in [2, 3]:
            alp = np.linspace(-10, 100, 100)
        else:
            alp = np.linspace(-10, 130, 100)

        plt.figure(axis)
        plt.plot(alpha[axis], pressure[axis])
        plt.plot(alp, poly(alp))
#        
#        plt.figure(1)
#        plt.plot(pressure[axis], alpha[axis])


plt.show()
