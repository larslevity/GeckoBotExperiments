#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:47:21 2019

@author: ls
"""


import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import eval as ev
import save
import kin_model
import predict_pose as pp


if 1:
    print('TRACK FEET')
    fig, ax = plt.subplots(subplot_kw=dict(aspect='equal'),
                           num='Track of feet')
    col = ['red', 'orange', 'magenta', 'blue', 'green', 'darkred']

    # ## compare to model -- withot stretching
    init_pose = [(90, 1, -90, 90, 1), 0, (-5, 0)]
    step = 10
    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 0, 0]]
           for gam in range(-90, 91, step)]
    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 0]]
            for gam in range(-90, 90, step)[::-1]]  # revers
    ref = ref + ref2

    x, r, data, cst, marks = pp.predict_pose(ref, init_pose, True, False,
                                             len_leg=13, len_tor=14,
                                             dev_ang=.1)
#    pp.plot_gait(*pp.start_mid_end(*data))
    positions = [{}, {}]
    markers = pp.marker_history(marks)
    for idx, marker in enumerate(markers):
        x, y = marker
        positions[0][idx] = x
        positions[1][idx] = y
        plt.plot(x, y, '-', color=col[idx])

    # ############################################# SINGLE SHOTS ##############
    # ################# plain track
    for idx in [0, 9, 18, 27, 36]:
        print(idx)
        fig, ax = plt.subplots(num='Track of feet {}'.format(idx),
                               subplot_kw=dict(aspect='equal'))
        for axis in range(6):
            x, y = positions[0][axis], positions[1][axis]
            # plot xy
            plt.plot(x, y, color=col[axis], linewidth=20)
            # plot sigma xy
#            for xx, yy, sigxx, sigyy in zip(x, y, sigx, sigy):
#                el = pat.Ellipse((xx, yy), sigxx*2, sigyy*2,
#                                 facecolor=col[axis], alpha=.3)
#                ax.add_artist(el)
        # plot eps inclination
        x1, y1, x4, y4 = (positions[0][1][idx], positions[1][1][idx],
                          positions[0][4][idx], positions[1][4][idx])
        dx = x4 - x1
        dy = y4 - y1
        plt.plot([x1-dx, x4+dx], [y1-dy, y4+dy],
                 '--', color='mediumpurple', linewidth=20)
        # draw best fit gecko
        pos = ([positions[0][axis][idx] for axis in range(6)],
               [positions[1][axis][idx] for axis in range(6)])
        x = data[-1][idx]
        n_limbs = 5
        alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]

        for jdx in range(6):
            plt.plot(positions[0][jdx][idx], positions[1][jdx][idx], 'o',
                     markersize=60, color=col[jdx])

        gecko_tikz_str = save.tikz_draw_gecko(alp, ell, eps,
                                              (positions[0][0][idx], positions[1][0][idx]),
                                              linewidth='2mm')


        plt.axis('off')
        save.save_as_tikz('pics/simulation/track_{}.tex'.format(idx), gecko_tikz_str,
                          scale=.2)

plt.show()
