#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:30:46 2020

@author: ls
"""

import cv2

import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import calibration
from Src import IMGprocessing
from Src import inverse_kinematics


import merge_img


modes = [
        'straight_1',
        'straight_2',
        'straight_3',
        'curve_1',
        'curve_2',
        'curve_3',
        ]

version = 'vS12'

len_leg, len_tor = [158, 190]  # in px

ell0 = [len_leg, len_leg, len_tor, len_leg, len_leg]





n_poses = 3
start_idx = 2


def line_normal(img_exp, point, eps, h, col=(.5,.5,.5,1)):
    point0 = (int(point[0]+np.cos(np.deg2rad(90-eps))*500),
              int(h-(point[1]-np.sin(np.deg2rad(90-eps))*500)))
    point1 = (int(point[0]-np.cos(np.deg2rad(90-eps))*500),
              int(h-(point[1]+np.sin(np.deg2rad(90-eps))*500)))
    cv2.line(img_exp, point0, point1, col, 2)



for mode in modes:
    for idx in range(n_poses):
        frame = cv2.imread('pics/'+mode+'/'+mode+'_({}).jpg'.format(idx+start_idx), 1)

        # measure
        alpha, eps, positions, xref = IMGprocessing.detect_all(frame)
        X1 = (positions[0][1], positions[1][1])
        col = (0, 0, 0)
#        IMGprocessing.draw_positions(frame, positions, xref, thick=2, col=col)

#        if idx == 0:
#            IMGprocessing.draw_eps(frame, X1, eps, color=col, dist=70, thick=2)
#        IMGprocessing.draw_pose(frame, alpha, eps, positions, ell0, col=col)
        
        if idx%2 == 0:
            alpha = alpha[:2] + [-alpha[3]] + alpha[-2:]
        else:
            alpha = alpha[:3] + alpha[-2:]
    
        # correction
        alpha_opt, eps_opt, positions_opt = \
            inverse_kinematics.correct_measurement(alpha, eps, positions, len_leg, len_tor)
        alpha_opt = alpha_opt[:3] + [-alpha_opt[2]] + alpha_opt[-2:]
        
    
        col = (10, 100, 200)
        if np.isnan(positions_opt[0][1]):
            X1_opt = X1
            eps_opt = eps
        else:
            X1_opt = (int(positions_opt[0][1]), int(positions_opt[1][1]))
        if idx == 0:
            x1_0 = X1_opt
            eps0 = eps_opt
        
#        IMGprocessing.draw_pose(frame, alpha_opt, eps_opt, positions_opt, ell0,
#                                col=col)
        if idx == 0 or idx+1 == n_poses:
            IMGprocessing.draw_eps(frame, X1_opt, eps_opt, color=col,
                                   dist=200, thick=2)
#        img = IMGprocessing.draw_positions(frame, positions_opt, xref, thick=2, col=col)
        alpha_opt = alpha_opt[:3] + alpha_opt[-2:]
    
        
        print('coords in main:')
        print('dalp:\t', [round(opt-ori, 2) for ori, opt in zip(alpha, alpha_opt)])
        print('deps:\t', round(eps_opt - eps, 2))
    # %%    
        if idx == 0:
            img_exp = merge_img.merge_pics(frame, fg_opa=.1)
        else:
            opa = .01 + (.7)*idx/(n_poses-1)
            img_exp = merge_img.merge_pics(frame, img_exp, fg_opa=opa)
        if idx+1 == n_poses:
            (h, w) = img_exp.shape[:2]
            cv2.line(img_exp, (int(x1_0[0]+np.cos(np.deg2rad(90-eps0))*450),
                               int(h-(x1_0[1]-np.sin(np.deg2rad(90-eps0))*450))),
                              (int(X1_opt[0]+np.cos(np.deg2rad(90-eps0))*450),
                               int(h-(X1_opt[1]-np.sin(np.deg2rad(90-eps0))*450))),
                                   (1,0,0, 1), 2)

            if mode[:8] == 'straight':
                line_normal(img_exp, x1_0, eps0, h)
                line_normal(img_exp, X1_opt, eps0, h)

    # %% rotate
    h, w = img_exp.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, -eps0, 1)
#    M = cv2.getRotationMatrix2D(center, 0, 1)

    img_exp = cv2.warpAffine(img_exp, M, (h,w))
    cv2.imwrite('Out/'+mode+'.png', img_exp*255)

#    cv2.imshow('frame', img_exp)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
