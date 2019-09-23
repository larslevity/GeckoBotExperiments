# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:11:35 2019

@author: AmP
"""

import cv2



import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import calibration
from Src import IMGprocessing
from Src import inverse_kinematics
from Src import plot_fun_pathPlanner as pf
import utils as uti

import merge_img


modes = [
        'straight_1',
        'straight_2',
        'straight_3',
        'curve_1',
        'curve_2',
        'curve_3',
        ]

version = 'v40'

len_leg, len_tor = calibration.get_len_px(version)

ell0 = [len_leg, len_leg, len_tor, len_leg, len_leg]





n_poses = 5


for mode in modes:
    for idx in range(n_poses):
        frame = cv2.imread('pics/'+mode+'/'+mode+'_({}).jpg'.format(idx+1), 1)

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
        X1_opt = (int(positions_opt[0][1]), int(positions_opt[1][1]))
        if idx == 0:
            x1_0 = X1_opt
        
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
        
        if idx == 0:
            img_exp = merge_img.merge_pics(frame, fg_opa=.1)
        else:
            opa = .01 + (.7)*idx/(n_poses-1)
            img_exp = merge_img.merge_pics(frame, img_exp, fg_opa=opa)
        if idx+1 == n_poses:
            (h, w) = img_exp.shape[:2]
            cv2.line(img_exp, (int(x1_0[0]), h-int(x1_0[1])+200),
                              (int(X1_opt[0]), h-int(X1_opt[1])+200),
                                   (1,0,0, 1), 2)

    cv2.imwrite(mode+'.png', img_exp*255)

#    cv2.imshow('frame', img_exp)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
