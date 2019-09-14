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
            'RFL': 'lightblue',     # reverse far left
            'R': 'red',
            'RR': 'darkred',
            'RFR': 'lightred'
            }
    return cols


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



def plot_needed_steps(needed_steps, runs):
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects = {}
    for idx, key in enumerate(needed_steps):
        n = len(needed_steps[key])
        x = np.linspace(idx - width/2, idx + width/2, n)
        rects[idx] = ax.bar(x, needed_steps[key], width/n)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of steps')
    ax.set_xticks([i for i in range(len(runs))])
    ax.set_xticklabels(runs)
    
    def autolabel(rectdic):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for key in rectdic:
            for rect in rectdic[key]:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    autolabel(rects)




