# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:38:11 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
from itertools import islice


from Src import save


def get_run_color():
    cols = {'180': 'orange',
            'L': 'blue',            # Left
            'RL': 'darkblue',       # reverse Left
            'SL': 'darkblue',
            'RFL': 'lightblue',     # reverse far left
            'R': 'red',
            'SR': 'darkred',
            'RR': 'darkred',
            'RFR': 'lightred'
            }
    return cols


def get_mode_color():
    cols = { 
        'without_eps_correction_x1_90': 'red',
        'without_eps_correction_x1_70': 'salmon',
        'without_eps_correction_x1_50': 'lightsalmon',
        'eps_corrected_x1_50': 'deepskyblue',
        'eps_corrected_x1_70': 'cornflowerblue',
        'eps_corrected_x1_90': 'royalblue',
            }
    return cols


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


def plot_track(db, POSE_IDX, run, mode, save_as_tikz=False):
    prop = calc_prop(db)
    col = get_run_color()

    plt.figure('Track'+mode)
    for exp_idx, dset in enumerate(db):
#        plt.figure('Track'+mode+str(exp_idx))
        for idx in [1, 8]:  # pos torso front & ref
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
                plt.plot(x_, y_, 'o', color=col[run])
            else:
                plt.plot(x_, y_, '-', color=col[run])
            
        x = [dset['x1'][pose_idx] for pose_idx in POSE_IDX[exp_idx]]
        y = [dset['y1'][pose_idx] for pose_idx in POSE_IDX[exp_idx]]
        plt.plot(x, y, 'o', color=col[run])

    plt.grid(1)
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    plt.axis('equal')
#    plt.xlim(-10, 165)
#    plt.ylim(-60, 60)
    if save_as_tikz:
        kwargs = {'extra_axis_parameters':
            {'x=.1cm', 'y=.1cm', 'anchor=origin'}}
        save.save_as_tikz('tikz/track_'+mode+'.tex', **kwargs)
        print(run)
#    plt.show()


def calc_prop(db):
    len_exp = min([len(dset['p1']) for dset in db])
    prop = 400./len_exp
    prop = .99 if prop >= 1. else prop
    return prop


def downsample(rows, proportion=.1):
    return np.array(
            list(islice(list(rows), 0, len(list(rows)), int(1/proportion))))



def plot_needed_steps(needed_steps, runs, modes, save_as_tikz=False):
    colors = get_mode_color()
    
    width_mode = .9
    
    fig, ax = plt.subplots()
    
    rectdic = {}
    lentries = []
    for jdx, mode in enumerate(modes):
        rectdic[mode] = {}
        N = len(modes)
        col = colors[mode]
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
        patch = pat.Patch(color=col, label=mode[-5:])  # last 5 chars
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
        save.save_as_tikz('tikz/needed_steps.tex')


def calc_mean_stddev(mat):
    mu1 = np.nanmean(mat, axis=1)
    sigma1 = np.nanstd(mat, axis=1)
    return mu1, sigma1


def plot_deps(db, POSE_IDX, run, mode, save_as_tikz=False):
    plt.figure('Deps'+run)
    plt.title(run)
    col = get_mode_color()

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

    mx, _ = calc_mean_stddev(XBAR)
    my, _ = calc_mean_stddev(YBAR)
    d = -np.linalg.norm([mx, my], axis=0)

    mu, sig = calc_mean_stddev(DEPSMAT)
    plt.plot(d, mu, color=col[mode])
    plt.fill_between(d, mu+sig, mu-sig,
                     facecolor=col[mode], alpha=0.5)

    return DEPSMAT    
