# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:38:11 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

from Src import eval as ev
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


def plot_track(db, run, prop, mode):
    col = get_run_color()

    plt.figure('Track'+mode)
    for exp_idx, dset in enumerate(db):
        for idx in [1, 8]:  # pos torso front & ref
            x = dset['x{}'.format(idx)]
            y = dset['y{}'.format(idx)]
            x = ev.downsample(x, proportion=prop)
            y = ev.downsample(y, proportion=prop)

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

    plt.grid(1)
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    plt.axis('equal')
    plt.xlim(-10, 165)
    plt.ylim(-60, 60)
    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.1cm', 'anchor=origin'}}
    save.save_as_tikz('tikz/track_'+mode+'.tex', **kwargs)
    print(run)
#    plt.show()


def calc_prop(db):
    len_exp = min([len(dset['p1']) for dset in db])
    prop = 400./len_exp
    prop = .99 if prop >= 1. else prop
    return prop