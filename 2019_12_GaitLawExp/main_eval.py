#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:57:26 2019

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
#from Src import timeout
#from Src import exception


# %%


Q1 = np.array([50, 60, 70, 80, 90])
Q2 = np.array([-.5, -.25, 0, .25, .5])

#Q1 = np.array([60])
#Q2 = [0.25]

DEPS = np.zeros((len(Q2), len(Q1)))
X_idx = np.zeros((len(Q2), len(Q1)))
Y_idx = np.zeros((len(Q2), len(Q1)))
version = 'vS11'

n_cyc = 2
sc = 10  # scale factor
dx, dy = 3.5*sc, (3+2.5*(n_cyc-1 if n_cyc > 1 else 1))*sc

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

        dirpath = version + '/'
        name = 'q1_' + q1str + 'q2_' + q2str

        print('load data:', name)
        try:
            db = func_timeout(1, uti.load_data, args=(dirpath, [name]))
        except FunctionTimedOut:
            db = []
            print('can not load data...')
        print('find poses ...')
        POSE_IDX = uti.find_poses_idx(db, neighbors=10)
        POSE_IDX = [pidx[:13] for pidx in POSE_IDX]
        n_steps[name] = len(POSE_IDX[0])-1

    # %% ### Track of feet:
        print('plot track....')
        pf.plot_track(db, POSE_IDX, name, version, save_as_tikz=False)

        # Extract Pose
        exp_idx = 0
        idx = 0
        gait_cor = roboter_repr.GeckoBotGait()
        gait_raw = roboter_repr.GeckoBotGait()
        for idx in range(2*n_cyc):

            X1_init = (q2_idx*dx, q1_idx*dy)
            alp, eps, fpos, p, fix, _ = uti.extract_measurement(
                    db[exp_idx], POSE_IDX[exp_idx][idx])
            # shift:
            xpos, ypos = fpos
            fpos = ([x + X1_init[0] for x in xpos],
                    [y + X1_init[1] for y in ypos])
            x = alp + ell_n + [eps]
            pose_raw = roboter_repr.GeckoBotPose(x, fpos, fix)
            gait_raw.append_pose(pose_raw)
            
#            print(idx, 'pose complete:', pose_raw.complete())

#            # corrected current_pose
            def correct(alp, eps, fpos):
                alp_c, eps_c, fpos_c = inverse_kinematics.correct_measurement(
                        alp, eps, fpos, len_leg, len_tor)
                x_c = alp_c + ell_n + [eps_c]
                pose_cor = roboter_repr.GeckoBotPose(x_c, fpos_c, fix)
                return pose_cor
            try:
                if pose_raw.complete():
                    print('correct measurement ...')
                    pose_cor = func_timeout(1, correct, args=(alp, eps, fpos))
                    gait_cor.append_pose(pose_cor)
            except FunctionTimedOut:
                print('time out. cant correct measurement ...')

        GAITS_cor.append(gait_cor)
        GAITS_raw.append(gait_raw)

    # %% EPS
        print('plot eps....')
        plt.figure('eps')
        eps, t = pf.plot_eps(db, POSE_IDX, name, version, save_as_tikz=False)

        MEAS[q1str][q2str]['eps'] = eps
        MEAS[q1str][q2str]['time'] = t
        MEAS[q1str][q2str]['time_full'] = db[0]['time']
        MEAS[q1str][q2str]['eps_full'] = db[0]['eps']
        deps = np.nanmean(np.diff(eps))*2  # mean deps/cycle
        DEPS[q2_idx][q1_idx] = deps

# %% DEPS Visual
    plt.figure('eps'+q1str)
    col = pf.get_run_color()[name]
    Q2_ = [-.5, -.25, 0]
    for q2_idx, q2 in enumerate(Q2):
        q2str = str(q2).replace('.', '').replace('00', '0')
        plt.plot(MEAS[q1str][q2str]['time_full'],
                 MEAS[q1str][q2str]['eps_full'], col)
        t, eps = MEAS[q1str][q2str]['time'], MEAS[q1str][q2str]['eps']
        deps = DEPS[q2_idx][q1_idx]
        plt.plot(t, eps, 'o', color=col, alpha=.5, markersize=4)
        

        # Polyfit
        idx = np.isfinite(t) & np.isfinite(eps)  # filter NaN
        epsdot, c = np.polyfit(t[idx], eps[idx], 1)
        dt = t[-1] - t[0]
        # counter check
        deps_ = epsdot*dt/(len(eps)-1)*2
        print(deps, deps_)
        plt.plot([t[0], t[-1]], [c, c+dt*epsdot], 'k')
        
        poly = np.poly1d([epsdot, c])
        # deviation of eps from mean of deps trend for each pose
        deps_mean_eps = [poly(ti) - epsi for ti, epsi in zip(t[idx], eps[idx])]
        # mean deviation of eps of deps trend
        mean_deps_mean_eps = np.mean(np.abs(deps_mean_eps))
        MEAS[q1str][q2str]['abs_deps'] = mean_deps_mean_eps
        
        # plot
        for deps_mean, ti in zip(deps_mean_eps, t[idx]):
            plt.plot([ti, ti], [poly(ti), poly(ti)-deps_mean], 'k')
        plt.text(t[-1]+3, eps[-1], 'Deps/cyc =' + str(round(deps_, 1)) + '  deps=' + str(round(mean_deps_mean_eps, 1)),
                 ha="left", va="bottom",
                 bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),
                           ))

#    mean_abs_deps = np.mean([MEAS[q1str][key]['abs_deps'] for key in MEAS[q1str]])
#    MEAS[q1str]['abs_deps'] = mean_abs_deps
    
    # plot
    plt.ylabel('robot orientation epsilon [deg]')
    plt.xlabel('time [s]')
    plt.title('q1=' + q1str)
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(10.5, 8)
    fig.savefig('eps'+q1str+'.png', transparent=True,
                dpi=300, bbox_inches='tight')
    kwargs = {'extra_axis_parameters': {'width=10cm', 'height=6cm',
                                        'tick pos=left'}}
    my_save.save_plt_as_tikz('tex/eps'+q1str+'.tex', **kwargs)

    

# %% ABS DEPS # Schwankung des Roboters um seine Trendlinie
plt.figure('mean abs deps')
abs_deps = [MEAS[key]['abs_deps'] for key in MEAS]
plt.plot(Q1, abs_deps)
plt.xticks(Q1)
plt.yticks([round(a, 1) for a in abs_deps])
plt.xlabel('step length $q_1$ [deg]')
plt.ylabel('mean of oscillation amplitude [deg]')
plt.grid()

my_save.save_plt_as_tikz('tex/oscillation_amplitude.tex')


# %% EPS
plt.figure('eps')
plt.xlabel('time')
plt.ylabel('robot orientation epsilon')
fig = plt.gcf()
fig.set_size_inches(10.5, 8)
fig.savefig('eps.png', transparent=True,
            dpi=300, bbox_inches='tight')


# %% EPS / GAIT
print('create figure: EPS/GAIT')

fig = plt.figure('GeckoBotGait')
levels = np.arange(-65, 66, 5)
if len(Q1) > 1:
    contour = plt.contourf(X_idx, Y_idx, DEPS*n_cyc, alpha=1,
                           cmap='RdBu_r', levels=levels)
# surf = plt.contour(X_idx, Y_idx, DEPS, levels=levels, colors='k')
# plt.clabel(surf, levels, inline=True, fmt='%2.0f')


for gait in GAITS_raw:
    gait.plot_gait(g='c')
    gait.plot_orientation(length=.5*sc)

#for gait in GAITS_cor:
#    gait.plot_gait(g='c')
#    gait.plot_orientation(length=.5*sc)



for xidx, x in enumerate(list(DEPS)):
    for yidx, deps in enumerate(list(x)):
        plt.text(X_idx[xidx][yidx], Y_idx[xidx][yidx]-2.2*sc, round(deps*n_cyc, 1),
                 ha="center", va="bottom",
                 bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           ))


plt.xticks(X_idx.T[0], [round(x, 2) for x in Q2])
plt.yticks(Y_idx[0], [round(x, 1) for x in Q1])
plt.ylabel('step length $q_1$')
plt.xlabel('steering $q_2$')
plt.axis('scaled')

plt.grid()
ax = fig.gca()
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)


fig.set_size_inches(10.5, 8)
fig.savefig('gait.png', transparent=True,
            dpi=300, bbox_inches='tight')


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