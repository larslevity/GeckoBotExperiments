#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:52:54 2020

@author: ls
"""


import matplotlib.pyplot as plt
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from Src import save as my_save
import plotfun_GaitLawExp as pf
import gait_law_utils as uti
from Src import calibration
from Src import inverse_kinematics
from Src import roboter_repr


# %%


Q1 = np.array([60])
Q2 = np.array([-.5])
#Q1 = np.array([80, 90])
#Q2 = np.array([-.5, .5])

DEPS = np.zeros((len(Q2), len(Q1)))
AMPLITUDE = np.zeros((len(Q2), len(Q1)))
DX = np.zeros((len(Q2), len(Q1)))
DY = np.zeros((len(Q2), len(Q1)))
DXSIG = np.zeros((len(Q2), len(Q1)))
DYSIG = np.zeros((len(Q2), len(Q1)))
X_idx = np.zeros((len(Q2), len(Q1)))
Y_idx = np.zeros((len(Q2), len(Q1)))
version = 'vS11'
c1val = 'c110_redo'

n_cyc = 1
sc = 10  # scale factor
dx, dy = 2.8*sc, (3+1.5*(n_cyc-1 if n_cyc > 1 else 1))*sc

len_leg, len_tor = calibration.get_len(version)
ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]

n_steps = {}
MEAS = {}
GAITS_cor = []
GAITS_raw = []


# %%
for q1_idx, q1 in enumerate(Q1):
    q1str = str(q1)
    MEAS[q1str] = {}
    for q2_idx, q2 in enumerate(Q2):
        # %% ### Load Data

        q2str = str(q2).replace('.', '').replace('00', '0')
        MEAS[q1str][q2str] = {}
        X_idx[q2_idx][q1_idx] = q2_idx*dx
        Y_idx[q2_idx][q1_idx] = q1_idx*dy

        dirpath = c1val + '/'
        name = 'q1_' + q1str + 'q2_' + q2str

        # %% 01 Raw data
        print('load data:', name)
        try:
            db = func_timeout(1, uti.load_data, args=(dirpath, [name], 1))
        except FunctionTimedOut:
            db = []
            print('can not load data...')

        print('find poses ...')
        POSE_IDX = uti.find_poses_idx(db, neighbors=10)
        n_steps[name] = len(POSE_IDX[0])-1

        pf.plot_track(db, POSE_IDX, name, version,
                      save_as_tikz='Out/visualization_data_processing/01.tex',
                      show_cycles=0)

        # %% 02 Scaled/shifted/rotated data
        print('load data:', name)
        try:
            db = func_timeout(1, uti.load_data, args=(dirpath, [name]))
        except FunctionTimedOut:
            db = []
            print('can not load data...')

        print('find poses ...')
        POSE_IDX = uti.find_poses_idx(db, neighbors=10)
        n_steps[name] = len(POSE_IDX[0])-1

        pf.plot_track(db, POSE_IDX, name, version,
                      save_as_tikz='Out/visualization_data_processing/02.tex',
                      show_cycles=1)

# %% 03 Identify Poses

        # Extract Pose
        exp_idx = 0
        gait_cor = roboter_repr.GeckoBotGait()
        gait_raw = roboter_repr.GeckoBotGait()
        gait_mean = roboter_repr.GeckoBotGait()
        gait_raw_help = \
            {i: roboter_repr.GeckoBotGait() for i in range(2*n_cyc + 1)}
        X1_init = (q2_idx*dx, q1_idx*dy)
        eps_0 = 90
        print('startpunkt fuer Plot:  ', X1_init)

        X = {i: [] for i in range(2*n_cyc + 1)}
        FPOS = {i: [] for i in range(2*n_cyc + 1)}
        FIX = {i: [] for i in range(2*n_cyc + 1)}

        for pidx in range(0, n_steps[name]):
            alp, eps_init, fpos, _, fix, _ = uti.extract_measurement(
                        db[exp_idx], POSE_IDX[exp_idx][pidx])
            x = alp + ell_n + [eps_init]
            pose_raw = roboter_repr.GeckoBotPose(x, fpos, fix)
            pose_raw.fpos_real = fpos
            gait_raw.append_pose(pose_raw)

        
        gait_raw.plot_markers(figname='03')
        plt.axis('off')
        gait_tex = gait_raw.get_tikz_repr(linewidth='1.5mm')
        my_save.save_plt_as_tikz('Out/visualization_data_processing/03.tex',
                                 additional_tex_code=gait_tex,
                                 scale=2,
                                 scope='scale=.1, opacity=.8')

#%% Do math

        for pidx in range(2, n_steps[name]-2, 2):  # drop first and last 2poses
            alp, eps_init, fpos, _, fix, _ = uti.extract_measurement(
                        db[exp_idx], POSE_IDX[exp_idx][pidx])
            eps_init -= eps_0   # for plot purpose
            x1_init = np.r_[fpos[0][1], fpos[1][1]]

            for idx in range(2*n_cyc + 1):
                alp, eps, fpos, p, fix, _ = uti.extract_measurement(
                        db[exp_idx], POSE_IDX[exp_idx][pidx+idx])
                # shift to zero:
                xpos, ypos = fpos
                fpos = ([x - x1_init[0] for x in xpos],
                        [y - x1_init[1] for y in ypos])
                # rotate
                fpos = uti.rotate_feet(fpos, -eps_init)
                # shift to plot position
                xpos, ypos = fpos
                fpos = ([x + X1_init[0] for x in xpos],
                        [y + X1_init[1] for y in ypos])
                # rotate also eps
                x = alp + ell_n + [eps-eps_init]
                X[idx].append(x)
                FPOS[idx].append(fpos)
                FIX[idx].append(fix)
                # append to gait for better comprhension
                pose_raw = roboter_repr.GeckoBotPose(x, fpos, fix)
                pose_raw.fpos_real = (fpos)
                gait_raw_help[idx].append_pose(pose_raw)

        # calc mean of all measurements
        gait_raw_mean = \
            {i: roboter_repr.GeckoBotGait() for i in range(2*n_cyc + 1)}
        for idx in range(2*n_cyc + 1):
            xpos_mean, ypos_mean = [], []
            for i in range(6):
                xmean = np.nanmean(
                    [FPOS[idx][cyc][0][i] for cyc in range(len(FPOS[idx]))])
                xpos_mean.append(xmean)
                ymean = np.nanmean(
                    [FPOS[idx][cyc][1][i] for cyc in range(len(FPOS[idx]))])
                ypos_mean.append(ymean)
            state_x_mean = []
            for i in range(11):
                mean = np.nanmean(
                    [X[idx][cyc][i] for cyc in range(len(X[idx]))])
                state_x_mean.append(mean)
            fix_mean = []
            for i in range(4):
                fix = np.nanmean(
                    [FIX[idx][cyc][i] for cyc in range(len(FIX[idx]))])
                fix = 0 if fix < 1 else 1
                fix_mean.append(fix)
            pose_mean = roboter_repr.GeckoBotPose(
                    state_x_mean, (xpos_mean, ypos_mean), fix_mean)
            pose_mean.fpos_real = (xpos_mean, ypos_mean)

            gait_raw_mean[idx].append_pose(pose_mean)
            gait_mean.append_pose(pose_mean)
            # double check:
            plt.figure('check' + name)
            gait_raw_help[idx].plot_gait()
            gait_raw_mean[idx].plot_gait()

        print(idx, 'pose complete:', pose_mean.complete())

# %% 04 Shift Poses

        gait_tex = ''
        for idx in range(3):
            gait_raw_help[idx].plot_markers(figname='04')
            gait_tex = (gait_tex + '\n%%%%%%%\n' 
                        + gait_raw_help[idx].get_tikz_repr(
                                linewidth='1.5mm'))
        plt.axis('off')
        my_save.save_plt_as_tikz('Out/visualization_data_processing/04.tex',
                                 additional_tex_code=gait_tex,
                                 scale=3,
                                 scope='scale=.1, opacity=.8')



# %% 05 Mean of Poses
        gait_mean.plot_gait()
        gait_mean.plot_markers(1, figname='05')
        gait_tex = gait_mean.get_tikz_repr(linewidth='1.5mm')
        plt.axis('off')
        my_save.save_plt_as_tikz('Out/visualization_data_processing/05.tex',
                                 additional_tex_code=gait_tex,
                                 scale=3,
                                 scope='scale=.1, opacity=.8')



# %% 06 Correct Pose

        # correct gait raw
        for pose_raw in gait_mean.poses:
            def correct(alp, eps, fpos, fix):
                alp_c, eps_c, fpos_c = inverse_kinematics.correct_measurement(
                        alp, eps, fpos, len_leg, len_tor)
                x_c = alp_c + ell_n + [eps_c]
                pose_cor = roboter_repr.GeckoBotPose(x_c, fpos_c, fix)
                pose_cor.fpos_real = fpos
                return pose_cor
            try:
                if pose_raw.complete():
                    print('correct measurement ...')
                    alp, eps = pose_raw.alp, pose_raw.eps
                    fix, fpos = pose_raw.f, pose_raw.markers
                    pose_cor = func_timeout(1, correct, args=(
                            alp, eps, fpos, fix))
                    gait_cor.append_pose(pose_cor)
            except FunctionTimedOut:
                print('time out. cant correct measurement ...')


        gait_cor.plot_gait()
        gait_cor.plot_markers(1, figname='06')
        gait_tex = gait_cor.get_tikz_repr(linewidth='1.5mm')
        plt.axis('off')
        my_save.save_plt_as_tikz('Out/visualization_data_processing/06.tex',
                                 additional_tex_code=gait_tex,
                                 scale=3,
                                 scope='scale=.1, opacity=.8')


        GAITS_cor.append(gait_cor)

# %%

gait_tex = ''
for gait in GAITS_cor:
#    gait.plot_gait(g='c')
#    gait.plot_travel_distance()
    gait.plot_orientation(length=.5*sc)
    gait_tex = gait_tex + '\n%%%%%%%\n' + gait.get_tikz_repr(linewidth='.7mm')


my_save.save_plt_as_tikz('Out/visualization_data_processing/gait.tex',
                         additional_tex_code=gait_tex,
                         scale=.7,
                         scope='scale=.1, opacity=.8')


plt.show()