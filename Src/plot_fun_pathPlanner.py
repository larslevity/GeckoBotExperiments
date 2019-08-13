# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:38:11 2019

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


def plot_track(db, run, prop, dirpath):
    col = ev.get_marker_color()
    col += ['black']*3
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'),
                           num='Track of feet '+run)
    for exp_idx, dset in enumerate(db):
        for idx in [1, 8]:  # pos torso front & ref
            x = dset['x{}'.format(idx)]
            y = dset['y{}'.format(idx)]
            x = ev.downsample(x, proportion=prop)
            y = ev.downsample(y, proportion=prop)
            plt.plot(x, y, 'o', color=col[exp_idx])
        #    plt.plot(x[0], y[0], 'o', markersize=20, color=col[idx])
#            for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
#                if not np.isnan(xx):
#                    el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
#                                     facecolor=col[idx], alpha=.3)
#                    ax.add_artist(el)
    ax.grid()
    ax.set_xlabel('x position (cm)')
    ax.set_ylabel('y position (cm)')
#    ax.set_xlim((-20, 65))
#    ax.set_ylim((-20, 20))
#    kwargs = {'extra_axis_parameters': {'x=.1cm', 'y=.1cm'}}
#    save.save_as_tikz('tikz/'+dirpath+'track.tex', **kwargs)
    print(run)
    plt.show()


def calc_prop(db):
    len_exp = min([len(dset['p1']) for dset in db])
    prop = 400./len_exp
    prop = .99 if prop >= 1. else prop
    return prop