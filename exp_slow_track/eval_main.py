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
        x, sigx = ev.calc_mean_of_axis(db, cyc, 'x{}'.format(axis), [1])
        y, sigy = ev.calc_mean_of_axis(db, cyc, 'y{}'.format(axis), [1])

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

        a, siga = ev.calc_mean_of_axis(db, cyc, 'aIMG{}'.format(axis), [1])
        p, sigp = ev.calc_mean_of_axis(db, cyc, 'p{}'.format(axis), [1])
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
        deg = 3
        coef = np.polyfit(alpha[axis], pressure[axis], deg)
        coef_s = ['%1.3e' % c for c in coef]
        print('Actuator %s:\t%s' % (axis, coef_s))
        poly = np.poly1d(coef)
        if axis in [2, 3]:
            alp = np.linspace(-100, 100, 100)
        else:
            alp = np.linspace(-10, 100, 100)

        plt.figure(axis)
        plt.plot(alpha[axis], pressure[axis])
        plt.plot(alp, poly(alp))
#        
#        plt.figure(1)
#        plt.plot(pressure[axis], alpha[axis])


    # %% ####### plot eps inclination
#    for x1, y1, x4, y4, i in zip(positions[0][1], positions[1][1],
#                                 positions[0][4], positions[1][4],
#                                 range(len(positions[0][4]))):
#        if np.mod(i, 30) == 0:
#            plt.plot([x1, x4], [y1, y4], '--k')

    # %% ####### plot gecko in first, mid and end position:
#    for idx in [0, len(eps)/2, len(eps)-1]:
#        pos = ([positions[0][axis][idx] for axis in range(6)],
#               [positions[1][axis][idx] for axis in range(6)])
#        alp = [alpha[axis][idx] for axis in range(6)]
#        alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
#        eps_ = eps[idx]
#
#        pose, marks, ell, alp__ = kin_model.extract_pose(
#                alp_, eps_, pos, len_leg=9, len_tor=10, max_alp_dif=15)
#        plt.plot(pose[0], pose[1], '.', color='gray', markersize=1)
#        for jdx in range(6):
#            plt.plot(marks[0][jdx], marks[1][jdx], 'o', markersize=5, color=col[jdx])
#        print 'idx: ', ell, [a_ - a__ for a_, a__ in zip(alp_, alp__)]

    # %%  # ## compare to model -- withot stretching
#    init_pose = [(90, 1, -90, 90, 1), 0, (-5, 0)]
#    step = 45
#    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 0, 0]]
#           for gam in range(-90, 91, step)]
#    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 0]]
#            for gam in range(-90, 90, step)[::-1]]  # revers
#    ref = ref + ref2
#
#    x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
#                                             len_leg=13, len_tor=14,
#                                             dev_ang=.1)
##    pp.plot_gait(*pp.start_mid_end(*data))
#
#    markers = pp.marker_history(marks)
#    for idx, marker in enumerate(markers):
#        x, y = marker
#        plt.plot(x, y, '-', color=col[idx])
#    # %% SINGLE SHOTS
#    if DRAW_SHOTS:
#        shots_at_time = [0, 29, 50, 75, 100]
#        indexes = [ev.closest_index(t, val/100.) for val in shots_at_time]
#        for shot, idx in enumerate(indexes):
#            fig, ax = plt.subplots(num='Shot' + version + '{}'.format(idx),
#                                   subplot_kw=dict(aspect='equal'))
#            for axis in range(6):
#                x, y = positions[0][axis], positions[1][axis]
#                sigx, sigy = SIGXY[0][axis], SIGXY[1][axis]
#                # downsample for tikz
#                prop = .1
#                x, y = ev.downsample(x, prop), ev.downsample(y, prop)
#                sigx, sigy = ev.downsample(sigx, prop), ev.downsample(sigy, prop)
#                # plot xy
#                ax.plot(x, y, color=col[axis], linewidth=20)
##                # plot sigma xy
##                for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
##                    el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
##                                     facecolor=col[axis], alpha=.3)
##                    ax.add_artist(el)
#            # plot eps inclination
#            x1, y1, x4, y4 = (positions[0][1][idx], positions[1][1][idx],
#                              positions[0][4][idx], positions[1][4][idx])
#            dx = x4 - x1
#            dy = y4 - y1
#            plt.plot([x1-dx, x4+dx], [y1-dy, y4+dy],
#                     '--', color='mediumpurple', linewidth=20)
#            # draw best fit gecko
#            pos = ([positions[0][axis][idx] for axis in range(6)],
#                   [positions[1][axis][idx] for axis in range(6)])
#            alp = [alpha[axis][idx] for axis in range(6)]
#            alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
#            eps_ = eps[idx]
#    
#            pose, marks, ell, alp__ = kin_model.extract_pose(
#                    alp_, eps_, pos, len_leg=len_leg[version],
#                    len_tor=len_tor[version])
#            for jdx in range(6):
#                plt.plot(marks[0][jdx], marks[1][jdx], 'o',
#                         markersize=60, color=col[jdx])
#    
#            gecko_tikz_str = save.tikz_draw_gecko(alp__, ell, eps_,
#                                                  (marks[0][0], marks[1][0]),
#                                                  linewidth='2mm')
#    
#    #        plt.plot(pose[0], pose[1], '.', color='gray', markersize=1)
#            plt.axis('off')
#            save.save_as_tikz('pics/shots/'+version+'shot_{}.tex'.format(shots_at_time[shot]),
#                              gecko_tikz_str, scale=.2)
#
## %%
#
#plt.figure('Epsilon corrected')
#plt.grid()
#plt.xlabel(r'time in cycle $t/t_{cyc}$ (1)')
#plt.ylabel(r'orientation angle epsilon $\varepsilon$ (deg)')
#save.save_as_tikz('pics/eps.tex')



plt.show()
