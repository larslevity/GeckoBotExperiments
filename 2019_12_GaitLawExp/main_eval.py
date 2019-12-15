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
AMPLITUDE = np.zeros((len(Q2), len(Q1)))
DX = np.zeros((len(Q2), len(Q1)))
DY = np.zeros((len(Q2), len(Q1)))
DXSIG = np.zeros((len(Q2), len(Q1)))
DYSIG = np.zeros((len(Q2), len(Q1)))
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


    # %% DX/DY
        dx_mean = []
        dy_mean = []
        for idx in [0, 2, 4, 6, 8]:
            _, eps, fpos, _, _, _ = uti.extract_measurement(db[exp_idx], POSE_IDX[exp_idx][idx])
            x1_init = np.r_[fpos[0][1], fpos[1][1]]
            eps_init = eps
            
            _, eps, fpos, _, _, _ = uti.extract_measurement(db[exp_idx], POSE_IDX[exp_idx][idx+2])
            x1 = np.r_[fpos[0][1], fpos[1][1]]
            travel = uti.rotate(x1 - x1_init, np.deg2rad(-eps_init))
            dx_mean.append(travel[0])
            dy_mean.append(travel[1])
            
            
        DX[q2_idx][q1_idx] = np.nanmean(dx_mean)
        DY[q2_idx][q1_idx] = np.nanmean(dy_mean)

        DXSIG[q2_idx][q1_idx] = np.nanstd(dx_mean)
        DYSIG[q2_idx][q1_idx] = np.nanstd(dy_mean)
            

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
        AMPLITUDE[q2_idx][q1_idx] = mean_deps_mean_eps

        # plot
        for deps_mean, ti in zip(deps_mean_eps, t[idx]):
            plt.plot([ti, ti], [poly(ti), poly(ti)-deps_mean], 'k')
        plt.text(t[-1]+3, eps[-1], 'Deps/cyc =' + str(round(deps_, 1)) + '  deps=' + str(round(mean_deps_mean_eps, 1)),
                 ha="left", va="bottom",
                 bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),
                           ))

    mean_abs_deps = np.mean([MEAS[q1str][key]['abs_deps'] for key in MEAS[q1str]])
    mean_abs_deps_sig = np.std([MEAS[q1str][key]['abs_deps'] for key in MEAS[q1str]])
    
    MEAS[q1str]['abs_deps'] = mean_abs_deps
    MEAS[q1str]['abs_deps_sig'] = mean_abs_deps_sig
    
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
abs_deps_sig = [MEAS[key]['abs_deps_sig'] for key in MEAS]
plt.errorbar(Q1, abs_deps, yerr=abs_deps_sig)
plt.xticks(Q1)
plt.yticks([round(a, 1) for a in abs_deps])
plt.xlabel('step length $q_1$ [deg]')
plt.ylabel('mean of oscillation amplitude [deg]')
plt.grid()

my_save.save_plt_as_tikz('tex/oscillation_amplitude.tex')

#%%
levels = np.arange(0, 5,.5)

contour = plt.contourf(X_idx, Y_idx, AMPLITUDE, alpha=1, cmap='coolwarm',
                       levels=levels)
surf = plt.contour(X_idx, Y_idx, AMPLITUDE, levels=levels, colors='k')
plt.clabel(surf, levels, inline=True, fmt='%2.0f')
plt.xticks(X_idx.T[0], [round(x, 2) for x in Q2])
plt.yticks(Y_idx[0], [round(x, 1) for x in Q1])
plt.ylabel('step length $q_1$')
plt.xlabel('steering $q_2$')

fig = plt.gcf()
fig.set_size_inches(10.5, 8)
fig.savefig('tex/oscillation_amplitude_heatmap.png', transparent=True,
            dpi=300, bbox_inches='tight')


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

# PLOT VECTOR FIELD
fig, ax = plt.subplots(num='DXDY')
X1__, X2__ = np.meshgrid(Q1, Q2)
X1_ = X1__.flatten()
X2_ = X2__.flatten()

roundon = 3
order = 2

Adic = {}
Adic[0] = [X1_*0+1]
Adic[1] = [X1_, X2_]
Adic[2] = [X1_**2, X2_**2, X1_*X2_]
Adic[3] = [X1_**3, X2_**3, X1_**2*X2_, X2_**2*X1_]
Adic[4] = [X1_**4, X2_**4, X1_**3*X2_**1, X1_**2*X2_**2, X1_**1*X2_**3]
Adic[5] = [X1_**5, X2_**5, X1_**4*X2_**1, X1_**3*X2_**2, X1_**2*X2_**3, X1_**1*X2_**4]

Tdic = {}
Tdic[0] = '{}'
Tdic[1] = ' + {}x_1^1 + {}x_2^1'
Tdic[2] = ' + {}x_1^2 + {}x_2^2 + {}x_1^1x_2^1'
Tdic[3] = ' + {}x_1^3 + {}x_2^3 + {}x_1^2x_2^1 + {}x_1^1x_2^2'
Tdic[4] = ' + {}x_1^4 + {}x_2^4 + {}x_1^3x_2^1 + {}x_1^2x_2^2 + {}x_1^1x_2^3'
Tdic[5] = ' + {}x_1^5 + {}x_2^5 + {}x_1^4x_2^1 + {}x_1^3x_2^2 + {}x_1^2x_2^3 + {}x_1^1x_2^4'

pys = (
        '{}'
        + '+ {}*x1 + {}*x2'
        + '+ {}*x1**2 + {}*x2**2 + {}*x1**1*x2**1'
        + '+ {}*x1**3 + {}*x2**3 + {}*x1**2*x2**1 + {}*x1**1*x2**2'
        + '+ {}*x1**4 + {}*x2**4 + {}*x1**3*x2**1 + {}*x1**2*x2**2 + {}*x1**1*x2**3'
#                + '+ {}*x1**5 + {}*x2**5 + {}*x1**4*x2**1 + {}*x1**3*x2**2 + {}*x1**2*x2**3 + {}*x1**1*x2**4'
        )
tex = ''
for i in range(order+1):
    tex += Tdic[i]


def flat_list(l):
    return [item for sublist in l for item in sublist]
    
A = [Adic[i] for i in range(order+1)]
A = flat_list(A)
A = np.array(A).T
    


BDX = DX.flatten()
coeff, r, rank, s = np.linalg.lstsq(A, BDX)
coeff_ = [round(c, roundon) for c in coeff]
dx = tex.format(*coeff_)

FITDX = X1_*0.0
for c, a in zip(coeff_, A.T):
    FITDX += c*a

BDY = DY.flatten()
coeff, r, rank, s = np.linalg.lstsq(A, BDY)
coeff_ = [round(c, roundon) for c in coeff]
dy = tex.format(*coeff_)

FITDY = X1_*0.0
for c, a in zip(coeff_, A.T):
    FITDY += c*a

error_x = np.reshape(((BDX - FITDX)), np.shape(X1__), order='C')
error_y = np.reshape(((BDY - FITDY)), np.shape(X1__), order='C')
error_len_abs = np.sqrt((error_x**2 + error_y**2))
error_len_rel = error_len_abs / np.reshape(np.sqrt(BDX**2 + BDY**2), np.shape(X1__)) * 100
mean_error = round(np.mean(np.nanmean(error_len_rel, 0)[1:]), 2)  # x1=0 excluded
#       


scale = 60
q = ax.quiver(X2__, X1__, DX, DY, units='x', scale=scale)
ax.scatter(X2__, X1__, color='0.5', s=10)
ax.quiver(X2__, X1__, FITDX, FITDY, units='x', scale=scale, color='blue')
ax.quiver(X2__, X1__, error_x, error_y, units='x', scale=scale, color='red')

ax.grid()
plt.xlim([-.6, .7])

plt.xticks(Q2, [round(x, 2) for x in Q2])
plt.yticks(Q1, [round(x, 1) for x in Q1])
plt.ylabel('step length $q_1$')
plt.xlabel('steering $q_2$')



fig = plt.gcf()
#fig.set_size_inches(10.5, 8.5)
fig.savefig('tex/FitDXDY_order_{}_round_{}.png'.format(order, roundon),
            dpi=300, trasperent=True, bbox_inches='tight')




plt.show()