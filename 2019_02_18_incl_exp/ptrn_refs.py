#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:56:23 2019

@author: ls
"""

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from Src import save
from Src import kin_model
from Src import predict_pose as pp



def generate_ptrn(a0, a1, a23, a4, a5, f0, f1, f2, f3):
    ref = [
        [[a0, a1, -a23, a4, a5], [f0, f1, f2, f3]]
          ]
    init_pose = [(a0, a1, -a23, a4, a5), 0, (0, 0)]
    return init_pose, ref


refs_adj_ = [
    [1, 90, -120, 1, 90, 0, 1, 1, 0],
    [1, 90, -120, 1, 90, 1, 1, 1, 1],
    [1, 90, -120, 1, 90, 1, 0, 0, 1],

    [90, 1, 120, 90, 1, 1, 0, 0, 1],
    [90, 1, 120, 90, 1, 1, 1, 1, 1],
    [90, 1, 120, 90, 1, 0, 1, 1, 0]
        ]

refs = [
    [1, 90, -90, 1, 90, 0, 1, 1, 0],
    [1, 90, -90, 1, 90, 1, 1, 1, 1],
    [1, 90, -90, 1, 90, 1, 0, 0, 1],

    [90, 1, 90, 90, 1, 1, 0, 0, 1],
    [90, 1, 90, 90, 1, 1, 1, 1, 1],
    [90, 1, 90, 90, 1, 0, 1, 1, 0]
        ]

refs_adj = [
    [1., 90, -90, 1, 90, 0, 1, 1, 0],
    [1., 90, -90, 1, 90, 0, 1, 1, 1],
    [1., 90, -90, 1, 1., 0, 1, 1, 1],
    [1., 90, -90, 1, 90, 1, 1, 1, 1],
    [90, 90, -90, 1, 90, 1, 1, 1, 1],
    [1., 90, -90, 1, 90, 1, 1, 1, 1],
    [1., 90, -90, 1, 90, 1, 0, 0, 1],

    [90, 1., 90, 90, 1, 1, 0, 0, 1],
    [90, 1., 90, 90, 1, 1, 0, 1, 1],
    [90, 1., 90, 1., 1, 1, 0, 1, 1],
    [90, 1., 90, 90, 1, 1, 1, 1, 1],
    [90, 90, 90, 90, 1, 1, 1, 1, 1],
    [90, 1., 90, 90, 1, 1, 1, 1, 1],
    [90, 1., 90, 90, 1, 0, 1, 1, 0]
        ]


for idx, pose in enumerate(refs):
    init_pose, ref = generate_ptrn(*pose)
    x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
                                             len_leg=13, len_tor=14,
                                             dev_ang=.1)
    pp.plot_gait(*data)

    gecko_tikz_str = save.tikz_draw_gecko(ref[0][0], [1, 1, 1.4, 1, 1], 0, (0, 0),
                                          linewidth='1mm', fix=ref[0][1])
    plt.figure()
    save.save_as_tikz('tikz/refs/'+'std'+'{}.tex'.format(idx),
                      gecko_tikz_str, scale=1)



