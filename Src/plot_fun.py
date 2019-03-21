# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:33:12 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import eval as ev
import save


def plot_track(db, cyc, incl, prop, dirpath):
    col = ev.get_marker_color()
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'),
                           num='Track of feet '+incl)
    prop_track = prop*.5

    for idx in range(6):
        x, sigx = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'x{}'.format(idx))
        y, sigy = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'y{}'.format(idx))
        if idx == 1:
            DIST = x[-1] - x[0]
        x = ev.downsample(x, proportion=prop_track)
        y = ev.downsample(y, proportion=prop_track)
        sigx = ev.downsample(sigx, proportion=prop_track)
        sigy = ev.downsample(sigy, proportion=prop_track)
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
    ax.set_xlim((-20, 65))
    ax.set_ylim((-20, 20))
    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.1cm'}}
    save.save_as_tikz('tikz/'+dirpath+'track.tex', **kwargs)
    return DIST


def plot_eps(db, cyc, incl, prop, dirpath):
    fig, ax = plt.subplots(num='epsilon during cycle '+incl)

    eps, sige = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'eps')
    t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')
    eps = ev.downsample(eps, proportion=prop)
    t = ev.downsample(t, proportion=prop)
    sige = ev.downsample(sige, proportion=prop)

    t = ev.rm_offset(t)
    TIME = t[-1]
    ax.plot(t, eps, '-', color='mediumpurple', linewidth=2)
    ax.fill_between(t, eps+sige, eps-sige,
                    facecolor='mediumpurple', alpha=0.5)
    ax.set_ylim((-20, 50))
    ax.grid()
    ax.set_xlabel(r'time $t$ (s)')
    ax.set_ylabel(r'orientation angle $\varepsilon$ ($^\circ$)')
    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.05cm'}}
    save.save_as_tikz('tikz/'+dirpath+'eps.tex', **kwargs)
    return TIME


def plot_alpha(db, cyc, incl, prop, dirpath):
    col = ev.get_actuator_color()
    fig, ax = plt.subplots(nrows=2, ncols=1,
                           num='Alpha during cycle'+incl, sharex=True)
    ALPHA, SIGALPHA = [], []
    t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')
    t_s = ev.rm_offset(ev.downsample(t, proportion=prop))

    alp_dfx_0 = {}
    for exp in range(len(db)):
        alp_dfx_0[exp] = {}
        dfx_0 = ev.find_dfx_idx(db[exp], 'f0')
        for axis in range(6):
            alp_dfx_0[exp][axis] = [db[exp]['aIMG{}'.format(axis)][idx] for idx in dfx_0]
    alp_dfx_1 = {}
    for exp in range(len(db)):
        alp_dfx_1[exp] = {}
        dfx_1 = ev.find_dfx_idx(db[exp], 'f1')
        for axis in range(6):
            alp_dfx_1[exp][axis] = [db[exp]['aIMG{}'.format(axis)][idx] for idx in dfx_1]

    
    TIMEA = t
    for axis in range(6):
        alp_, siga_ = ev.calc_mean_of_axis_multi_cyc(
                db, cyc, 'aIMG{}'.format(axis))

        # downsample for tikz
        alp = ev.downsample(alp_, proportion=prop)
        siga = ev.downsample(siga_, proportion=prop)
        ALPHA.append(alp_)
        SIGALPHA.append(siga_)

        if axis in [0, 3, 4]:
            axidx = 0
        elif axis in [1, 2, 5]:
            axidx = 1
        ax[axidx].plot(t_s, alp, '-', color=col[axis])
        ax[axidx].fill_between(t_s, alp+siga, alp-siga, facecolor=col[axis],
                               alpha=0.5)
    ax[0].grid()
    ax[1].grid()
    ax[0].set_ylim((-95, 95))
    ax[1].set_ylim((-95, 95))

    ax[0].set_ylabel(r'bending angle $\alpha$ ($^\circ$)')
    ax[1].set_ylabel(r'bending angle $\alpha$ ($^\circ$)')
    ax[1].set_xlabel(r'time $t$ (s)')

    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.02cm',
                                        'ytick={-90,-45,0,45, 90}'}}
    save.save_as_tikz('tikz/'+dirpath+'alpha.tex', **kwargs)
    return ALPHA, SIGALPHA, TIMEA, alp_dfx_0, alp_dfx_1


def plot_velocity(db, cyc, incl, prop, dirpath, Ts, DIST, TIME, VELX, VELY,
                  SIGVELX, SIGVELY):
    db = ev.calc_velocity(db, Ts)
    col = ev.get_actuator_color()
    fig, ax = plt.subplots(nrows=2, ncols=1,
                           num='velocity during cycle'+incl, sharex=True)
    VX, VY = [], []
    SIGVX, SIGVY = [], []
    for axis in [2, 3]:
        vx, sigvx = ev.calc_mean_of_axis_multi_cyc(
                db, cyc, 'x{}dot'.format(axis))
        vy, sigvy = ev.calc_mean_of_axis_multi_cyc(
                db, cyc, 'y{}dot'.format(axis))
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
        vx = ev.downsample(vx, proportion=prop)
        vy = ev.downsample(vy, proportion=prop)
        t_s = ev.rm_offset(ev.downsample(t, proportion=prop))
        sigvx = ev.downsample(sigvx, prop)
        sigvy = ev.downsample(sigvy, prop)

        ax[0].plot(t_s, vx, '-', color=col[axis])
        ax[0].fill_between(t_s, vx+sigvx, vx-sigvx,
          facecolor=col[axis], alpha=0.5)

        ax[1].plot(t_s, vy, '-', color=col[axis])
        ax[1].fill_between(t_s, vy+sigvy, vy-sigvy,
          facecolor=col[axis], alpha=0.5)

    vxmean_ = DIST/TIME
    vxmean = np.mean(VX)
    vymean = np.mean(VY)
    sigvxmean = np.mean(SIGVX)
    sigvymean = np.mean(SIGVY)
    VELX.append(vxmean_)
    VELY.append(vymean)
    SIGVELX.append(sigvxmean)
    SIGVELY.append(sigvymean)

    ax[0].plot([t_s[0], t_s[-1]], [vxmean]*2, ':', linewidth=2, color='gray')
    ax[0].plot([t_s[0], t_s[-1]], [vxmean_]*2, ':', linewidth=2, color='k')
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
    save.save_as_tikz('tikz/'+dirpath+'velocity.tex', **kwargs)
    return VELX, SIGVELX, VELY, SIGVELY


def plot_pressure(db, cyc, incl, prop, dirpath, ptrn, VOLUME, version, DIST,
                  ENERGY):
    col = ev.get_actuator_color()
    fig, ax = plt.subplots(nrows=2, ncols=1,
                           num='pressure during cycle'+incl, sharex=True)
    MAXPressure = {}
    for axis in range(6):
        p, sigp = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'p{}'.format(axis))
        r, _ = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'r{}'.format(axis))
        t, sigt = ev.calc_mean_of_axis_multi_cyc(db, cyc, 'time')
        MAXPressure[axis] = max(r)

        # downsample for tikz
        p = ev.downsample(p, proportion=prop)
        t_s = ev.rm_offset(ev.downsample(t, proportion=prop))
        sigp = ev.downsample(sigp, proportion=prop)

        if axis in [0, 3, 4]:
            axidx = 0
        elif axis in [1, 2, 5]:
            axidx = 1
        ax[axidx].plot(t_s, p, '-', color=col[axis])
        ax[axidx].fill_between(t_s, p+sigp, p-sigp, facecolor=col[axis],
                               alpha=0.5)
    ax[0].grid()
    ax[1].grid()
    ax[0].set_ylim((0, 1.2))
    ax[1].set_ylim((0, 1.2))

    ax[0].set_ylabel(r'pressure $p$ (bar)')
    ax[1].set_ylabel(r'pressure $p$ (bar)')
    ax[1].set_xlabel(r'time $t$ (s)')

    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=2cm',
                                        'ytick={0, .5, 1}'}}
    save.save_as_tikz('tikz/'+dirpath+'pressure.tex', **kwargs)

    min_len = min([len(cycle) for cycle in cyc])
    n_cyc = min_len - 2  # first is skipped and last too
    if int(incl) >= 76 and ptrn == 'adj_ptrn':
        n_cyc = n_cyc*10./12.  # ptrn for 76 has twice actuation of each leg

    energy = (sum([val[1]*1e2 for val in MAXPressure.items()])
              * VOLUME[version] * n_cyc/DIST)

    ENERGY.append(abs(energy))
    return ENERGY


def plot_vel_incl(VELX, VELY, SIGVELX, SIGVELY, INCL, ENERGY, version, ptrn):
    VELX = np.array(VELX)
    VELY = np.array(VELY)
    SIGVELX = np.array(SIGVELX)
    SIGVELY = np.array(SIGVELY)

    fig, ax = plt.subplots(num='incl - vel')
    ax.plot(INCL, VELX, color='red')
    ax.fill_between(INCL, VELX+SIGVELX, VELX-SIGVELX,
                    facecolor='red', alpha=0.5)

    ax.set_xlabel(r'inclination angle $\delta$ ($^\circ$)')
    ax.set_ylabel(r'mean of velocity $\Delta \bar{x} / \Delta t$ (cm/s)')
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(INCL, ENERGY)
    ax2.set_ylabel(r'Energy Consumption per shift (kJ/cm)')

    kwargs = {'extra_axis_parameters': {'xtick={0, 28, 48, 63, 76}'}}
    save.save_as_tikz('tikz/'+version+'/incl-vel-energy-{}.tex'.format(ptrn),
                      **kwargs)



def plot_alpha_incl(VELX, VELY, SIGVELX, SIGVELY, INCL, ENERGY, version, ptrn):
    VELX = np.array(VELX)
    VELY = np.array(VELY)
    SIGVELX = np.array(SIGVELX)
    SIGVELY = np.array(SIGVELY)

    fig, ax = plt.subplots(num='incl - vel')
    ax.plot(INCL, VELX, color='red')
    ax.fill_between(INCL, VELX+SIGVELX, VELX-SIGVELX,
                    facecolor='red', alpha=0.5)

    ax.set_xlabel(r'inclination angle $\delta$ ($^\circ$)')
    ax.set_ylabel(r'mean of velocity $\Delta \bar{x} / \Delta t$ (cm/s)')
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(INCL, ENERGY)
    ax2.set_ylabel(r'Energy Consumption per shift (kJ/cm)')

    kwargs = {'extra_axis_parameters': {'xtick={0, 28, 48, 63, 76}'}}
    save.save_as_tikz('tikz/'+version+'/incl-vel-energy-{}.tex'.format(ptrn),
                      **kwargs)



def calc_prop(db, cyc):
    min_len = min([len(cycle) for cycle in cyc])
    n_cyc = min_len - 1
    len_exp = cyc[0][n_cyc] - cyc[0][1]
    prop = 400./len_exp
    prop = .99 if prop >= 1. else prop
    return prop


def epsilon_correction(db, cyc):
    # correction of epsilon
    rotate = 5
    for exp in range(len(db)):
        # eps0 = db[exp]['eps'][cyc[exp][1]]
        eps0 = 0
        for marker in range(6):
            X = db[exp]['x{}'.format(marker)]
            Y = db[exp]['y{}'.format(marker)]
            X, Y = ev.rotate_xy(X, Y, -eps0+rotate)
            db[exp]['x{}'.format(marker)] = X
            db[exp]['y{}'.format(marker)] = Y
        db[exp]['eps'] = ev.add_offset(db[exp]['eps'], -eps0+rotate)
    return db


def plot_incl_alp_dfx(TIMEA, incls, ALP, ALP_dfx_0, ALP_dfx_1, version, ptrn):
    col = ev.get_actuator_color()
    alp_dfx = {}
    act = {0: 0,
           1: 1,
           2: 1,
           3: 0,
           4: 0,
           5: 1}
    for incl in incls:
        alp_dfx[incl] = {}
        for axis in range(6):
            mean_alp_dfx_0 = np.nan
            mean_alp_dfx_1 = np.nan
            for exp in range(len(ALP_dfx_0[incl])):
                mean_ = np.nanmean(ALP_dfx_0[incl][exp][axis])
                mean_alp_dfx_0 = np.nanmean([mean_alp_dfx_0, mean_])
            for exp in range(len(ALP_dfx_1[incl])):
                mean_ = np.nanmean(ALP_dfx_1[incl][exp][axis])
                mean_alp_dfx_1 = np.nanmean([mean_alp_dfx_1, mean_])
            alp_dfx[incl][axis] = {}
            alp_dfx[incl][axis][0] = mean_alp_dfx_0
            alp_dfx[incl][axis][1] = mean_alp_dfx_1
#            idx = len(TIMEA['00'])
#            if axis in [2]:
#                t = ev.rm_offset(TIMEA[incl][:idx])
#                plt.plot(t, ALP[incl][axis][:idx], color=col[axis])
#                t = ev.rm_offset([TIMEA[incl][0], TIMEA[incl][idx-1]])
#                plt.plot(t, [alp_dfx[incl][axis][act[axis]]]*2, color=col[axis])

    alp_dfx_act = {}
    alp_dfx_uact = {}

    for axis in range(6):
        alp_dfx_act[axis] = [alp_dfx[incl][axis][act[axis]] for incl in incls]
    for axis in range(6):
        alp_dfx_uact[axis] = [alp_dfx[incl][axis][int(not act[axis])] for incl in incls]

    INCL = [float(incl) for incl in incls]
    plt.figure('raw data')
    for axis in [0, 1, 2, 3, 4, 5]:
        plt.plot(INCL, alp_dfx_act[axis], color=col[axis])
    for axis in [0, 1, 4, 5]:
        plt.plot(INCL, alp_dfx_uact[axis], ':', color=col[axis])

    plt.grid()
    plt.xlabel(r'inclination angle $\delta$ ($^\circ$)')
    plt.ylabel(r'bending angle just before defixation')


    plt.figure('')
    alp_dfx_act_ = {}
    alp_dfx_uact_ = {}
    sigalp_dfx_act_ = {}
    sigalp_dfx_uact_ = {}
    for axis in [0, 2, 4]:
        alp_dfx_act_[axis] = np.array([np.mean([alp_dfx_act[axis][idx], alp_dfx_act[axis+1][idx]]) for idx in range(len(alp_dfx_act[axis]))])
        sigalp_dfx_act_[axis] = np.array([np.std([alp_dfx_act[axis][idx], alp_dfx_act[axis+1][idx]]) for idx in range(len(alp_dfx_act[axis]))])
        alp_dfx_uact_[axis] = np.array([np.mean([alp_dfx_uact[axis][idx], alp_dfx_uact[axis+1][idx]]) for idx in range(len(alp_dfx_act[axis]))])
        sigalp_dfx_uact_[axis] = np.array([np.std([alp_dfx_uact[axis][idx], alp_dfx_uact[axis+1][idx]]) for idx in range(len(alp_dfx_act[axis]))])
    
        plt.plot(INCL, alp_dfx_act_[axis], color=col[axis])
        plt.fill_between(INCL, alp_dfx_act_[axis]+sigalp_dfx_act_[axis], alp_dfx_act_[axis]-sigalp_dfx_act_[axis], facecolor=col[axis],
                                   alpha=0.5)
    for axis in [0, 4]:
        plt.plot(INCL, alp_dfx_uact_[axis], ':', color=col[axis])
        plt.fill_between(INCL, alp_dfx_uact_[axis]+sigalp_dfx_uact_[axis], alp_dfx_uact_[axis]-sigalp_dfx_uact_[axis], facecolor=col[axis],
                                   alpha=0.5)
    
    plt.grid()
    plt.xlabel(r'inclination angle $\delta$ ($^\circ$)')
    plt.ylabel(r'bending angle $\alpha$ just before defixation ($^\circ$)')

    kwargs = {'extra_axis_parameters': {'xtick={0, 28, 48, 63, 76}'}}
    save.save_as_tikz('tikz/'+version+'/incl-alp_dfx-{}.tex'.format(ptrn),
                      **kwargs)