#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:32:22 2019

@author: ls
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
from itertools import islice


from Src import save


def get_marker_color():
    return ['red', 'orange', 'darkred', 'blue', 'darkorange', 'darkblue']


def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


def calc_xbar(xref, xbot, epsbot):
    """ maps the reference point in global COS to robot COS """
    xref = np.array([[xref[0]], [xref[1]]])
    xbot = np.array([[xbot[0]], [xbot[1]]])
    return rotate(xref - xbot, np.deg2rad(-epsbot))


def calc_deps(xbar):
    return np.rad2deg(np.arctan2(xbar[1], xbar[0]))


def plot_track(db, POSE_IDX, run, mode, save_as_tikz=False, show_cycles=1):
    prop = calc_prop(db)
    col = 'red'

    plt.figure('Track'+mode)
    for exp_idx, dset in enumerate(db):
#        plt.figure('Track'+mode+str(exp_idx))
        for idx in [1]:  # pos torso front & ref
            x = dset['x{}'.format(idx)]
            y = dset['y{}'.format(idx)]
            x = downsample(x, proportion=prop)
            y = downsample(y, proportion=prop)

            # remove double entries
            x_, y_ = [], []
            for xi, xj, yi in zip(x, x[1:], y):
                if round(xi, 2) != round(xj, 2) and not np.isnan(xi):
                    x_.append(xi)
                    y_.append(yi)

            if idx == 8:
                try:
                    plt.plot(x_, y_, 'o', color=col)
                except KeyError:
                    plt.plot(x_, y_, 'o')
            else:
                try:
                    plt.plot(x_, y_, '-', color=col)
                except KeyError:
                    plt.plot(x_, y_, '-')
        if show_cycles:
            x = [dset['x1'][pose_idx] for pose_idx in POSE_IDX[exp_idx]]
            y = [dset['y1'][pose_idx] for pose_idx in POSE_IDX[exp_idx]]
            try:
                plt.plot(x, y, 'o', color=col)
            except KeyError:
                plt.plot(x, y, 'o')

    plt.grid(1)
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    plt.axis('equal')
#    plt.xlim(-10, 165)
#    plt.ylim(-60, 60)
    if save_as_tikz:
        if isinstance(save_as_tikz, str):
            name = save_as_tikz
        else:
            name = 'Out/track_'+mode+'_'+run+'.tex'
        kwargs = {'extra_axis_parameters':
            {'x=.1cm', 'y=.1cm', 'anchor=origin'}}
        save.save_plt_as_tikz(name, **kwargs)
        print(run)
#    plt.show()


def calc_prop(db):
    len_exp = min([len(dset['pr1']) for dset in db])
    prop = 400./len_exp
    prop = .99 if prop >= 1. else prop
    return prop


def downsample(rows, proportion=.1):
    return np.array(
            list(islice(list(rows), 0, len(list(rows)), int(1/proportion))))



def plot_needed_steps(needed_steps, runs, modes, save_as_tikz=False):
    
    width_mode = .9
    
    fig, ax = plt.subplots()
    
    rectdic = {}
    lentries = []
    for jdx, mode in enumerate(modes):
        rectdic[mode] = {}
        N = len(modes)
        col = 'red'
        for idx, key in enumerate(needed_steps[mode]):
            n = len(needed_steps[mode][key])
            width = width_mode/(N)  # the width of the bars
            x = np.array([idx - width/2 + ii*width/n for ii in range(n)])
            if N > 1:
                aux = width_mode - width
                shift = -aux/2 + (jdx)*aux/(N-1)
                x = x + shift
            rectdic[mode][idx] = ax.bar(x, needed_steps[mode][key], width/n*1,
                   color=col)
        patch = pat.Patch(color=col, label=mode[-5:].replace('_', ''))  # last 5 chars
        lentries.append(patch)

    plt.legend(handles=lentries)
    ax.set_ylabel('Number of steps')
    ax.set_xlabel('Set Point')
    ax.set_xticks([i for i in range(len(runs))])
    ax.set_xticklabels(runs)
    
    def autolabel(rectdic):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for mode in rectdic:
            for key in rectdic[mode]:
                for rect in rectdic[mode][key]:
                    height = rect.get_height()
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
#    autolabel(rectdic)
    if save_as_tikz:
        save.save_plt_as_tikz('tikz/needed_steps.tex')


def calc_mean_stddev(mat):
    mu1 = np.nanmean(mat, axis=1)
    sigma1 = np.nanstd(mat, axis=1)
    return mu1, sigma1


def plot_eps(db, POSE_IDX, run, mode, save_as_tikz=False):
#    plt.figure('eps')
#    plt.title('RUN: ' + run + ' - Eps')
    col = 'green'

    max_poses = max([len(pidx) for pidx in POSE_IDX])
    EPSMAT = np.empty((max_poses, len(db)))
    TMAT = np.empty((max_poses, len(db)))
    EPSMAT[:] = np.nan
    TMAT[:] = np.nan

    try:
        color = col
    except KeyError:
        color = None

    for exp_idx, dset in enumerate(db):
        plt.plot(dset['time'], dset['eps'], color=color)
        EPS = np.take(dset['eps'], POSE_IDX[exp_idx])
        T = np.take(dset['time'], POSE_IDX[exp_idx])
      
        for pidx, (eps, t) in enumerate(zip(EPS, T)):
            EPSMAT[pidx][exp_idx] = eps
            TMAT[pidx][exp_idx] = t

    mu_eps, sig = calc_mean_stddev(EPSMAT)
    t, sigt = calc_mean_stddev(TMAT)
    plt.plot(t, mu_eps, 'o', color=color)
    plt.fill_between(t, mu_eps+sig, mu_eps-sig,
                     facecolor=color, alpha=0.5)
    plt.xlabel('time')
    plt.ylabel('robot orientation epsilon')
    plt.grid()
    if save_as_tikz:
        if isinstance(save_as_tikz, str):
            name = save_as_tikz
        else:
            name = 'tikz/track_'+mode+'_'+run+'.tex'
        save.save_plt_as_tikz(name)

    return mu_eps, t


def plot_deps_over_steps(db, POSE_IDX, run, mode, save_as_tikz=False):
    fig, ax1 = plt.subplots(num='Deps'+run)
    ax1.set_title('RUN: ' + run + ' - Delta Eps')

    max_poses = max([len(pidx) for pidx in POSE_IDX])
    DEPSMAT = np.empty((max_poses, len(db)))
    DEPSMAT[:] = np.nan
    XBAR = np.empty((max_poses, len(db)))
    YBAR = np.empty((max_poses, len(db)))
    XBAR[:] = np.nan
    YBAR[:] = np.nan

    for exp_idx, dset in enumerate(db):
        EPS = np.take(dset['eps'], POSE_IDX[exp_idx])
        X1 = np.take(dset['x1'], POSE_IDX[exp_idx])
        Y1 = np.take(dset['y1'], POSE_IDX[exp_idx])
        XREF = np.take(dset['x8'], POSE_IDX[exp_idx])
        YREF = np.take(dset['y8'], POSE_IDX[exp_idx])

        for pidx, (eps, x1, y1, xref, yref) in enumerate(
                zip(EPS, X1, Y1, XREF, YREF)):
            xbar = calc_xbar((xref, yref), (x1, y1), eps)
            deps = calc_deps(xbar)
            XBAR[pidx][exp_idx] = xbar[0]
            YBAR[pidx][exp_idx] = xbar[1]
            DEPSMAT[pidx][exp_idx] = deps

    mx, sigx = calc_mean_stddev(XBAR)
    my, sigy = calc_mean_stddev(YBAR)
    d = np.linalg.norm([mx, my], axis=0)
    dsig = np.linalg.norm([sigx, sigy], axis=0)

    color='blue'
    mu, sig = calc_mean_stddev(DEPSMAT)
    ax1.plot(mu, color=color)
    ax1.fill_between(range(len(mu)), mu+sig, mu-sig,
                     facecolor=color, alpha=0.5)
    ax1.set_xlabel('step index $i$')
    ax1.set_ylabel('relative angle between robot orientation and goal $\Delta eps_i$',
                   color=color)
    ax1.tick_params('y', colors=color)

    color='red'
    ax2 = ax1.twinx()
    ax2.plot(d, ':', color=color)
    ax2.fill_between(range(len(mu)), d+dsig, d-dsig,
                     facecolor=color, alpha=0.5)
    ax2.set_ylabel('distance to goal $|x_i|$', color=color)
    ax2.tick_params('y', colors=color)
    
    if save_as_tikz:
        save.save_plt_as_tikz('tikz/deps_'+mode+'_'+run+'.tex')

    return DEPSMAT  




