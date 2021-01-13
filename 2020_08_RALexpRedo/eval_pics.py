#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:30:46 2020

@author: ls
"""

import cv2
import time

import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import calibration
from Src import IMGprocessing
from Src import inverse_kinematics


import merge_img


modes = [
#        'straight_1',
#        'straight_2',
#        'straight_3',
        'curve_1',
        'curve_2',
        'curve_3',
        ]

version = 'vS12'

len_leg, len_tor = [162, 195]  # in px

ell0 = [len_leg, len_leg, len_tor, len_leg, len_leg]





n_poses = 3
start_idx = 5  # 2 or straight | 5 for curve


def line_normal(img_exp, point, eps, h, col=(.5,.5,.5,1)):
    point0 = (int(point[0]+np.cos(np.deg2rad(90-eps))*500),
              int(h-(point[1]-np.sin(np.deg2rad(90-eps))*500)))
    point1 = (int(point[0]-np.cos(np.deg2rad(90-eps))*500),
              int(h-(point[1]+np.sin(np.deg2rad(90-eps))*500)))
    cv2.line(img_exp, point0, point1, col, 2)


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def change_saturation(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:    
        lim_upper = 255 - value
        s[s > lim_upper] = 255
        s[s <= lim_upper] += value
    else:    
        lim_lower = 0 - value
        s[s < lim_lower] = 0
        s[s >= lim_lower] -= abs(value)
        
    

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


for mode in modes:
    poses = []
    for idx in range(n_poses):
        frame = cv2.imread('pics/'+mode+'/'+mode+'_({}).jpg'.format(idx+start_idx), 1)
        
        

        # measure
        alpha, eps, positions, xref = IMGprocessing.detect_all(frame)
        X1 = (positions[0][1], positions[1][1])
        col = (0, 0, 100)
#        IMGprocessing.draw_positions(frame, positions, xref, thick=2, col=col)

#        if idx == 0:
#            IMGprocessing.draw_eps(frame, X1, eps, color=col, dist=70, thick=2)
#        IMGprocessing.draw_pose(frame, alpha, eps, positions, ell0, col=col, thickness=2)
        
        if idx%2 == 0:
            alpha = alpha[:2] + [-alpha[3]] + alpha[-2:]
        else:
            alpha = alpha[:3] + alpha[-2:]
    
        # correction
        alpha_opt, eps_opt, positions_opt = \
            inverse_kinematics.correct_measurement(alpha, eps, positions, len_leg, len_tor)
        alpha_opt = alpha_opt[:3] + [-alpha_opt[2]] + alpha_opt[-2:]
        
    
        col = (0, 0, 100)
        if np.isnan(positions_opt[0][1]):
            X1_opt = X1
            eps_opt = eps
        else:
            X1_opt = (int(positions_opt[0][1]), int(positions_opt[1][1]))
        if idx == 0:
            x1_0 = X1_opt
            eps0 = eps_opt
        
#        IMGprocessing.draw_pose(frame, alpha_opt, eps_opt, positions_opt, ell0, col=col, thickness=3)
        poses.append([alpha_opt, eps_opt, positions_opt, ell0])
        

        if idx == 0 or idx+1 == n_poses:
            IMGprocessing.draw_eps(frame, X1_opt, eps_opt, color=col,
                                   dist=200, thick=2)
#        img = IMGprocessing.draw_positions(frame, positions_opt, xref, thick=2, col=col)
        alpha_opt = alpha_opt[:3] + alpha_opt[-2:]
    
        
        print('coords in main:')
        print('dalp:\t', [round(opt-ori, 2) for ori, opt in zip(alpha, alpha_opt)])
        print('deps:\t', round(eps_opt - eps, 2))

        

    # %%    
        
        
        frame = change_saturation(frame, value=-100)
#        frame = increase_brightness(frame, value=0)
    
        if idx == 0:
            img_exp = merge_img.merge_pics(frame, fg_opa=0)
        else:
            opa = (.7)*idx/(n_poses-1)
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


    # %% draw extracted poses

    for jdx, pose in enumerate(poses):
        col = [.7-jdx*.4/(len(poses)-1)]*3 + [1]
        alpha_opt, eps_opt, positions_opt, ell0 = pose
        IMGprocessing.draw_pose(img_exp, alpha_opt, eps_opt, positions_opt, ell0, col=col, thickness=3)

        
    

   # %% rotate
    h, w = img_exp.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(x1_0, -eps0, 1)
#    M = cv2.getRotationMatrix2D(center, -eps0, 1)

    img_exp = cv2.warpAffine(img_exp, M, (h,w))

    cv2.imwrite('Out/'+mode+'.png', img_exp*255)

# %% crop
    x0, y0 = x1_0
    print('y0:', y0)
    if mode[:8] == 'straight':
        if mode[-1] == '1':
            crop_img = img_exp[y0-250:y0+250, x0-410:x0+530,:]
        if mode[-1] == '2':
            crop_img = img_exp[y0-330:y0+170, x0-430:x0+510,:]
        if mode[-1] == '3':
            crop_img = img_exp[y0-550:y0-50, x0-360:x0+580,:]

    if mode[:5] == 'curve':
        if mode[-1] == '1':
            crop_img = img_exp[y0-290:y0+250, x0-430:x0+290,:]
        if mode[-1] == '2':
            crop_img = img_exp[y0-290:y0+250, x0-410:x0+310,:]
        if mode[-1] == '3':
            crop_img = img_exp[y0-330:y0+210, x0-430:x0+290,:]


## %% save   
#    img_exp = cv2.cvtColor(img_exp, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Out/'+mode+'_cropped_.png', crop_img*255)

#    cv2.imshow('frame', img_exp)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


