# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:11:35 2019

@author: AmP
"""

import cv2
import IMGprocessing


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import eval_pathPlanner as ev
from Src import load
from Src import plot_fun_pathPlanner as pf
import utils as uti


modes = [
#        'straight_3',
        'curve_1',
        ]

version = 'v40'


ell0 = [len_leg, len_leg, len_tor, len_leg, len_leg]

for mode in modes:
    frame = cv2.imread('pics/'+mode+'/'+mode+'_(0).jpg', 1)

    # measure
    alpha, eps, positions, xref = IMGprocessing.detect_all(frame)
    X1 = (positions[0][1], positions[1][1])
    col = (0, 0, 0)
    IMGprocessing.draw_positions(frame, positions, xref, thick=2, col=col)
    IMGprocessing.draw_eps(frame, X1, eps, color=col, dist=70, thick=2)
#        IMGprocessing.draw_pose(frame, alpha, eps, positions, ell0, col=col)

    # correction
    alpha_opt, eps_opt, positions_opt = \
        correct_measurement(alpha, eps, positions)
    col = (10, 100, 200)
    X1_opt = (positions_opt[0][1], positions_opt[1][1])
    IMGprocessing.draw_pose(frame, alpha_opt, eps_opt, positions_opt, ell0,
                            col=col)
    IMGprocessing.draw_eps(frame, X1_opt, eps_opt, color=col,
                           dist=50, thick=2)
    img = IMGprocessing.draw_positions(frame, positions_opt, xref,
                                       thick=2, col=col)

    print('coords in main:')
    print('alp:\t', [round(opt-ori, 2) for ori, opt in zip(alpha, alpha_opt)])
    print('deps:\t', round(eps_opt - eps, 2))

    cv2.imwrite('test.jpg', img)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
