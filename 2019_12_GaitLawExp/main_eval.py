#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:57:26 2019

@author: ls
"""


import matplotlib.pyplot as plt
import numpy as np



import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from Src import save as my_save
import plotfun_GaitLawExp as pf
import gait_law_utils as uti
from Src import calibration
from Src import inverse_kinematics

# %%


Q1 = np.array([50, 60, 70, 80, 90])
Q2 = np.array([-.5, -.25, 0, .25, .5])

Q1 = np.array([80, 90])
#Q2 = [.5]

DEPS = np.zeros((len(Q2), len(Q1)))
X_idx = np.zeros((len(Q2), len(Q1)))
Y_idx = np.zeros((len(Q2), len(Q1)))
version = 'vS11'

n_cyc = 3
dx, dy = 3.5, 3+2.5*(n_cyc-1 if n_cyc > 1 else 1)

len_leg, len_tor = calibration.get_len(version)
ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]



# %%

n_steps = {}
MEAS = {}

for q1_idx, q1 in enumerate(Q1):
    q1str = str(q1)
    MEAS[q1str] = {}
    for q2_idx, q2 in enumerate(Q2):
        # %% ### Load Data
        q2str = str(q2).replace('.', '').replace('00', '0')
        MEAS[q1str][q2str] = {}
        X_idx[q2_idx][q1_idx] = q2_idx*dx
        Y_idx[q2_idx][q1_idx] = -q1_idx*dy

        dirpath = version + '/'        
        name = 'q1_' + q1str + 'q2_' + q2str

        db = uti.load_data(dirpath, [name])
        POSE_IDX = uti.find_poses_idx(db, neighbors=10)
        n_steps[name] = len(POSE_IDX[0])-1

    # %% ### Track of feet:
        print('plot track....')
        pf.plot_track(db, POSE_IDX, name, version, save_as_tikz=False)
        
        # Extract Pose
        
        exp_idx = 0
        idx = 0
        for idx in range(2*n_cyc):
            c = (1-float(idx)/2/n_cyc)*.8
            col = (c, c, 0)

            alp, eps, fpos, p, fix, _ = uti.extract_measurement(db[exp_idx], POSE_IDX[exp_idx][idx])
#            x = alp + ell_n + [eps]
#            uti.plot_pose(x, fpos, fix, col=(c, c, 1))

            # corrected current_pose
            alp_c, eps_c, fpos_c = inverse_kinematics.correct_measurement(
                    alp, eps, fpos, len_leg=len_leg, len_tor=len_tor)
            x_c = alp_c + ell_n + [eps_c]
            uti.plot_pose(x_c, fpos_c, fix, col=(c, c, 0))
        
        
        
    # %% EPS
        
        eps = pf.plot_eps(db, POSE_IDX, name, version, save_as_tikz=False)
        
        MEAS[q1str][q2str]['eps'] = eps
        DEPS[q2_idx][q1_idx] = np.nanmean(np.diff(eps))*2  # mean deps/cycle

# %% EPS

fig = plt.figure('DEPSperCycle')
levels = np.arange(-21, 22, 3)
contour = plt.contourf(X_idx, Y_idx, DEPS, alpha=1,
                       cmap='RdBu', levels=levels)
surf = plt.contour(X_idx, Y_idx, DEPS, levels=levels, colors='k')
plt.clabel(surf, levels, inline=True, fmt='%2.0f')

plt.xticks(X_idx.T[0], [round(x, 2) for x in Q2])
plt.yticks(Y_idx[0], [round(x, 1) for x in Q1])
plt.ylabel('step length $q_1$')
plt.xlabel('steering $q_2$')







# %%

#    # %%
#    predict_poses = 4
#    start_idx = settings[mode]['startidx']
#    nexps = settings[mode]['nexps']
#
#    print('calc predictions errors....')
#
#    (ALPERR, PERR, EPSERR, alpsig, psig, epsig, gait_predicted, DXm, DXsig,
#     depsm, depssig, deps_sim_m, deps_sim_sig, DXsim_m, DXsim_sig) = \
#        uti.calc_errors(db, POSE_IDX, version, mode=mode,
#                        nexps=nexps, predict_poses=predict_poses,
#                        start_idx=start_idx)
#
## %%
#    for idx in PERR:
#        MU_P[idx][mode] = PERR[idx][1:]
#        SIG_P[idx][mode] = psig[idx][1:]
#        MU_DX[idx][mode] = DXm[idx][1:]
#        SIG_DX[idx][mode] = DXsig[idx][1:]
#        MU_DX_SIM[idx][mode] = DXsim_m[idx][1:]
#        SIG_DX_SIM[idx][mode] = DXsim_sig[idx][1:]
#
#    MU_EPS[mode] = EPSERR[1:]
#    SIG_EPS[mode] = epsig[1:]
#
#    MU_DEPS[mode] = depsm[1:]
#    SIG_DEPS[mode] = depssig[1:]
#    MU_DEPS_SIM[mode] = deps_sim_m[1:]
#    SIG_DEPS_SIM[mode] = deps_sim_sig[1:]
#
## %%
#
#    gait_predicted.save_as_tikz(mode+'_gait')
#
#
## %%
##    plt.figure('PredictionErrors-ALP')
##    for idx in range(len(ALPERR)):
##        plt.plot(ALPERR[idx], label='marker {}'.format(idx))
##    plt.xlabel('Step count')
##    plt.ylabel('Prediction Error of Angle |a_m - a_p|')
#
## %% barplot
#
#if 0:  # old plots / relative plots
#    # POS
#    for idx in [1]:
#        mu = MU_P[idx]
#        sig = SIG_P[idx]
#        N = max([len(v) for v in mu.values()])
#        ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
#                         sig, num='error-p')
#    #    ax.set_ylim((0, 330))
#        ax.set_ylabel('$|{p}_m - {p}_p|/\\ell_{{n}}$ (\%)')
#        ax.set_xlabel('Pose Count')
#        ax.grid(True, axis='y')
#    
#        kwargs = {'extra_axis_parameters': {'anchor=origin',
#                                            'axis line style={draw=none}',
#                                            'xtick style={draw=none}'}}
#        my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_position_error_bar.tex',
#                                 **kwargs)
#    
#    # %% EPS
#    
#    mu = MU_EPS
#    sig = SIG_EPS
#    N = max([len(v) for v in mu.values()])
#    ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
#                     sig, num='error-eps')
#    ax.set_ylabel('$|{eps}_m - {eps}_p|$ (deg)')
#    ax.set_xlabel('Pose Count')
#    ax.grid(True, axis='y')
#    
#    kwargs = {'extra_axis_parameters': {'anchor=origin',
#                                        'axis line style={draw=none}',
#                                        'xtick style={draw=none}'}}
#    my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_epsilon_error_bar.tex',
#                             **kwargs)
#
#
## %% ABSOLUTE PLOTS
## %% DEPS
#
#kwargs = {'extra_axis_parameters': {'anchor=origin',
#                                    'axis line style={draw=none}',
#                                    'xtick style={draw=none}',
#                                    'height=6cm',
#                                    'width=10cm'}}
#
#
#mu = MU_DEPS
#sig = SIG_DEPS
#N = max([len(v) for v in mu.values()])
#ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
#                 sig, num='error-deps')
#
#ax.set_ylabel('${eps}_m - {eps}_0$ (deg)')
#ax.set_xlabel('pose count')
#ax.grid(True, axis='y')
#ax.set_ylim((0, 170))
#
#my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_deps.tex',
#                         **kwargs)
#
#
## %% DEPS_SIM
#
#mu = MU_DEPS_SIM
#sig = None
#N = max([len(v) for v in mu.values()])
#ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
#                 sig, num='error-deps_sim')
#
#ax.set_ylabel('${eps}_p - {eps}_0$ (deg)')
#ax.set_xlabel('pose count')
#ax.grid(True, axis='y')
#ax.set_ylim((0, 170))
#
#my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_deps_sim.tex',
#                         **kwargs)
#
#
## %% DX
#
#for idx in [1]:
#    mu = MU_DX[idx]
#    sig = SIG_DX[idx]
#    N = max([len(v) for v in mu.values()])
#    ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
#                     sig, num='dx')
#    ax.set_ylim((0, 500))
#    ax.set_ylabel('$Delta x_m/l_n$ (%)')
#    ax.set_xlabel('pose count')
#    ax.grid(True, axis='y')
#
#    my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_dx.tex',
#                             **kwargs)
#
#
## %% DX_SIM
#
#for idx in [1]:
#    mu = MU_DX_SIM[idx]
#    sig = None
#    N = max([len(v) for v in mu.values()])
#    ax = uti.barplot(mu, modes, [str(i+1) for i in range(N)], colors,
#                     sig, num='dx_sim')
#    ax.set_ylim((0, 500))
#    ax.set_ylabel('$Delta x_p/l_n$ (%)')
#    ax.set_xlabel('pose count')
#    ax.grid(True, axis='y')
#
#    my_save.save_plt_as_tikz('/Out/' + mode[:-2] + '_dx_sim.tex',
#                             **kwargs)
#
#
## %%
#plt.show()