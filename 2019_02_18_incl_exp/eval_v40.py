# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:25:49 2019

@author: ls
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

col = ev.get_marker_color()

version = 'v40'
ptrn = 'adj_ptrn'
incls = ['28', '48', '63', '76']
Ts = 0.03
VOLUME = {'v40': 1,
          'vS11': .7}

INCL, VELX, VELY, ENERGY = [], [], [], []
SIGVELX, SIGVELY, SIGENERGY = [], [], []


for incl in incls:
    INCL.append(int(incl))
    # %% ### Load Data

    dirpath = version+'/'+ptrn+'/incl_'+incl+'/'

    sets = load.get_csv_set(dirpath)
    #sets = [sets[0]]
    #print sets
    db, cyc = ev.load_data(dirpath, sets)

    # correction of epsilon
    rotate = 5
    for exp in range(len(db)):
        #eps0 = db[exp]['eps'][cyc[exp][1]]
        eps0 = 0
        for marker in range(6):
            X = db[exp]['x{}'.format(marker)]
            Y = db[exp]['y{}'.format(marker)]
            X, Y = ev.rotate_xy(X, Y, -eps0+rotate)
            db[exp]['x{}'.format(marker)] = X
            db[exp]['y{}'.format(marker)] = Y
        db[exp]['eps'] = ev.add_offset(db[exp]['eps'], -eps0+rotate)


    # %% ### eps during cycle
    plt.figure('Epsilon corrected')
    #for exp in range(len(db)):
    #    for idx in range(6):
    #        eps = db[exp]['eps'][cyc[exp][1]:cyc[exp][-1]]
    #        # hack to remove high freq noise
    #        for idx in range(1, len(eps)):
    #            if abs(eps[idx] - eps[idx-1]) > 30:
    #                eps[idx] = eps[idx-1]
    #        t = db[exp]['time'][cyc[exp][1]:cyc[exp][-1]]
    #        plt.plot(t, eps, ':', color='mediumpurple')

    eps, sige = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'eps')
    t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')
    eps = ev.downsample(eps)
    t = ev.downsample(t)
    sige = ev.downsample(sige)

    t = ev.rm_offset(t)
    plt.plot(t, eps, '-', color='mediumpurple', linewidth=2)
    plt.fill_between(t, eps+sige, eps-sige,
                     facecolor='mediumpurple', alpha=0.5)
    plt.grid()
    plt.xlabel(r'time $t$ (s)')
    plt.ylabel(r'orientation angle $\varepsilon$ ($^\circ$)')
    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.1cm'}}
    save.save_as_tikz(dirpath+'eps.tex', **kwargs)


    # %% ### Track of feet:
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'), num='Track of feet')
    
    #for exp in range(len(db)):
    #    for idx in range(6):
    #        x = db[exp]['x{}'.format(idx)][cyc[exp][1]:cyc[exp][-1]]
    #        y = db[exp]['y{}'.format(idx)][cyc[exp][1]:cyc[exp][-1]]
    #        plt.plot(x, y, color=col[idx])

    for idx in range(6):
        x, sigx = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'x{}'.format(idx))
        y, sigy = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'y{}'.format(idx))
        if idx == 1:
            DIST = x[-1] - x[0]
        prop = .05
        x = ev.downsample(x, prop)
        y = ev.downsample(y, prop)
        sigx = ev.downsample(sigx, prop)
        sigy = ev.downsample(sigy, prop)
        plt.plot(x, y, color=col[idx])
    #    plt.plot(x[0], y[0], 'o', markersize=20, color=col[idx])
        for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
            if not np.isnan(xx):
                el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
                                 facecolor=col[idx], alpha=.3)
                ax.add_artist(el)
    ax.grid()
    ax.set_xlabel('x position (cm)')
    ax.set_ylabel('y position (cm)')
    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.1cm'}}
    save.save_as_tikz(dirpath+'track.tex', **kwargs)

    # %% ### Alpha during cycle:
    col = ev.get_actuator_color()
    fig, ax = plt.subplots(nrows=2, ncols=1, num='Alpha during cycle', sharex=True)
    for axis in range(6):
        alp, siga = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'aIMG{}'.format(axis))
        t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')

        # downsample for tikz
        prop = .05
        alp = ev.downsample(alp, prop)
        t_s = ev.rm_offset(ev.downsample(t, prop))
        siga = ev.downsample(siga, prop)

        if axis in [0, 3, 4]:
            axidx = 0
        elif axis in [1, 2, 5]:
            axidx = 1
        ax[axidx].plot(t_s, alp, '-', color=col[axis])
        ax[axidx].fill_between(t_s, alp+siga, alp-siga, facecolor=col[axis],
                               alpha=0.5)
    ax[0].grid()
    ax[1].grid()

    ax[0].set_ylabel(r'bending angle $\alpha$ ($^\circ$)')
    ax[1].set_ylabel(r'bending angle $\alpha$ ($^\circ$)')
    ax[1].set_xlabel(r'time $t$ (s)')

    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.02cm',
                                        'ytick={-90,-45,0,45, 90}'}}
    save.save_as_tikz(dirpath+'alpha.tex', **kwargs)


    # %% Velocity

    db = ev.calc_velocity(db, Ts)
    col = ev.get_actuator_color()
    fig, ax = plt.subplots(nrows=2, ncols=1,
                           num='velocity during cycle', sharex=True)
    VX, VY = [], []
    SIGVX, SIGVY = [], []
    for axis in [2, 3]:
        vx, sigvx = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'x{}dot'.format(axis))
        vy, sigvy = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'y{}dot'.format(axis))
        t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')
        VX.append(np.nanmean(vx))
        VY.append(np.nanmean(vy))
        vxm = [np.nanmean(db[exp]['x{}dot'.format(axis)][cyc[exp][1]:cyc[exp][-2]])
               for exp in range(len(db))]
        vym = [np.nanmean(db[exp]['y{}dot'.format(axis)][cyc[exp][1]:cyc[exp][-2]])
               for exp in range(len(db))]
        SIGVX.append(np.nanstd(vxm))
        SIGVY.append(np.nanstd(vym))

        # downsample for tikz
        prop = .1
        vx = ev.downsample(vx, prop)
        vy = ev.downsample(vy, prop)
        t_s = ev.rm_offset(ev.downsample(t, prop))
        sigvx = ev.downsample(sigvx, prop)
        sigvy = ev.downsample(sigvy, prop)

        ax[0].plot(t_s, vx, '-', color=col[axis])
        ax[0].fill_between(t_s, vx+sigvx, vx-sigvx, facecolor=col[axis], alpha=0.5)

        ax[1].plot(t_s, vy, '-', color=col[axis])
        ax[1].fill_between(t_s, vy+sigvy, vy-sigvy, facecolor=col[axis], alpha=0.5)

    vxmean = np.mean(VX)
    vymean = np.mean(VY)
    sigvxmean = np.mean(SIGVX)
    sigvymean = np.mean(SIGVY)
    VELX.append(vxmean)
    VELY.append(vymean)
    SIGVELX.append(sigvxmean)
    SIGVELY.append(sigvymean)

    ax[0].plot([t_s[0], t_s[-1]], [vxmean]*2, ':', linewidth=2, color='gray')
    ax[1].plot([t_s[0], t_s[-1]], [vymean]*2, ':', linewidth=2, color='gray')

    ax[0].grid()
    ax[1].grid()

    ax[0].set_ylabel(r'velocity $\dot{x}$ (cm/s)')
    ax[1].set_ylabel(r'velocity $\dot{y}$ (cm/s)')
    ax[1].set_xlabel(r'time $t$ (s)')
    ax[1].set_ylim((-5, 5))
    ax[0].set_ylim((-5, 10))

    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.2cm',
                                        'ytick={-10, -5, 0, 5, 10}'}}
    save.save_as_tikz(dirpath+'velocity.tex', **kwargs)

    # %% ### pressure during cycle:
    col = ev.get_actuator_color()
    fig, ax = plt.subplots(nrows=2, ncols=1,
                           num='pressure during cycle', sharex=True)
    MAXPressure = {}
    for axis in range(6):
        p, sigp = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'p{}'.format(axis))
        r, _ = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'r{}'.format(axis))
        t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')
        MAXPressure[axis] = max(r)

        # downsample for tikz
        prop = .05
        p = ev.downsample(p, prop)
        t_s = ev.rm_offset(ev.downsample(t, prop))
        sigp = ev.downsample(sigp, prop)

        if axis in [0, 3, 4]:
            axidx = 0
        elif axis in [1, 2, 5]:
            axidx = 1
        ax[axidx].plot(t_s, p, '-', color=col[axis])
        ax[axidx].fill_between(t_s, p+sigp, p-sigp, facecolor=col[axis],
                               alpha=0.5)
    ax[0].grid()
    ax[1].grid()

    ax[0].set_ylabel(r'pressure $p$ (bar)')
    ax[1].set_ylabel(r'pressure $p$ (bar)')
    ax[1].set_xlabel(r'time $t$ (s)')

    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=2cm',
                                        'ytick={0, .5, 1}'}}
    save.save_as_tikz(dirpath+'pressure.tex', **kwargs)

    min_len = min([len(cycle) for cycle in cyc])
    n_cyc = min_len - 1  # first is skipped

    energy = sum([val[1] for val in MAXPressure.items()])*VOLUME[version]*n_cyc/DIST

    ENERGY.append(energy)


# %% VEL via incl

VELX = np.array(VELX)
VELY = np.array(VELY)
SIGVELX = np.array(SIGVELX)
SIGVELY = np.array(SIGVELY)

plt.figure('incl- vel')
plt.plot(INCL, VELX, color='red')
plt.fill_between(INCL, VELX+SIGVELX, VELX-SIGVELX, facecolor='red', alpha=0.5)






plt.show()
