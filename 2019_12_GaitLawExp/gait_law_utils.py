#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:17:40 2019

@author: ls
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as pat

from Src import calibration
from Src import kin_model
from Src import roboter_repr
from Src import inverse_kinematics
from Src import save as my_save
from Src import load



def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


# rot: x = cos(a)*x - sin(a)*y
#      y = sin(a)*x + cos(a)*y


def load_data(path, sets):
    dataBase = []
#    xscale = 145./1000  # 1000px -> 145cm
#    xshift = -22  # cm
#    yshift = -63  # cm
    xscale = 112./1000  # after changing resolution of RPi
    xshift = -12 - 50  # cm
    yshift = -45 - 20  # cm

    for exp in sets:
        data = load.read_csv(path+"{}.csv".format(exp))

        try:
            start_idx = data['f0'].index(1)  # upper left foot attached 1sttime
        except ValueError:  # no left foot is fixed
            start_idx = 0
        # correction
        start_time = data['time'][start_idx]
        start_eps = data['eps'][start_idx]
        if np.isnan(start_eps):
            i = 0
            while np.isnan(start_eps):
                i += +1
                start_eps = data['eps'][start_idx+i]
            print('took start_eps from idx', start_idx + i,
                  '(start_idx: ', start_idx, ')')
            if i > 10:
                print('MORE THAN 10 MEASUREMENTS!!')
        
        data['time'] = \
            [round(data_time - start_time, 3) for data_time in data['time']]
        for key in data:
            if key[0] in ['x', 'y']:
                shift = xshift if key[0] == 'x' else yshift
                data[key] = [i*xscale + shift for i in data[key]]
            if key == 'eps':
                data[key] = [np.mod(e+180-start_eps, 360)-180+90 for e in data[key]]
        # rotate:
        for idx in range(6):
            x = data['x{}'.format(idx)]
            y = data['y{}'.format(idx)]
            X, Y = [], []
            for vec in zip(x, y):
                xrot, yrot = rotate(vec, np.deg2rad(-start_eps+90))
                X.append(xrot)
                Y.append(yrot)
            data['x{}'.format(idx)] = X
            data['y{}'.format(idx)] = Y

        dataBase.append(data)

    return dataBase



def find_poses_idx(db, neighbors=5):
    IDX = []
    failed = 0
    for exp_idx in range(len(db)):
        pose_idx = []
        start_idx = db[exp_idx]['f1'].index(1)
        for idx in range(start_idx, len(db[exp_idx]['pr3'])-1, 1):
            if db[exp_idx]['pr3'][idx] != db[exp_idx]['pr3'][idx+1]:
                if not pose_idx:  # empty list
                    pose_idx.append(idx)
                else:
                    for jdx in range(idx, idx-neighbors, -1):  # look the last neigbors
                        if not np.isnan(db[exp_idx]['aIMG2'][jdx]):
                            # check
                            dr = db[exp_idx]['pr2'][idx] - db[exp_idx]['pr2'][jdx]
                            if abs(dr) > .1:
                                failed += 1
                                pose_idx.append(idx)  # append ori
                                break
                            else:
                                pose_idx.append(jdx)
                                break
                        elif jdx == idx-neighbors+1:
                            failed += 1
                            pose_idx.append(idx)  # append ori
        # last#
        idx = len(db[exp_idx]['pr3'])-1
        for jdx in range(idx, idx-100, -1):  # look the last neigbors
            if not np.isnan(db[exp_idx]['aIMG2'][jdx]):
                # check
                dr = db[exp_idx]['pr2'][idx] - db[exp_idx]['pr2'][jdx]
                if abs(dr) > .1:
                    failed += 1
                    pose_idx.append(idx)  # append ori
                    break
                else:
                    pose_idx.append(jdx)
                    break
        IDX.append(pose_idx)
        print('failed:', failed)
    return IDX


def extract_measurement(measurement, idx):
    alp = [measurement['aIMG{}'.format(j)][idx] for j in range(6)]
    fposx = [measurement['x{}'.format(j)][idx] for j in range(6)]
    fposy = [measurement['y{}'.format(j)][idx] for j in range(6)]
    p = [measurement['pr{}'.format(j)][idx] for j in range(6)]
    fix = [measurement['f{}'.format(j)][idx] for j in range(4)]
    eps = measurement['eps'][idx]
    xref = measurement['x7'][idx]
    yref = measurement['y7'][idx]
    if p[2] == 0:  # right elly actuated
        alp = alp[0:2] + [-alp[3]] + alp[-2:]
    else:  # left belly
        alp = alp[0:3] + alp[-2:]

    return (alp, eps, (fposx, fposy), p, fix, (xref, yref))


def error_of_prediction(db, IDX, start_idx, n_predictions, version='vS11',
                        mode='1', exp_idx=0):

    # init
    ALPERR = {i: np.empty((n_predictions)) for i in range(5)}
    PERR = {i: np.empty((n_predictions)) for i in range(6)}
    XERR = {i: np.empty((n_predictions)) for i in range(6)}
    YERR = {i: np.empty((n_predictions)) for i in range(6)}
    EPSERR = np.empty((n_predictions))
    DEPS = np.empty((n_predictions))
    DEPS_SIM = np.empty((n_predictions))
    DX = {i: np.empty((n_predictions)) for i in range(6)}
    DY = {i: np.empty((n_predictions)) for i in range(6)}
    DX_SIM = {i: np.empty((n_predictions)) for i in range(6)}
    DY_SIM = {i: np.empty((n_predictions)) for i in range(6)}
    for i in range(6):
        if i < 5:
            ALPERR[i][:] = np.nan
        PERR[i][:] = np.nan
        DX[i][:] = np.nan
        DY[i][:] = np.nan
        DX_SIM[i][:] = np.nan
        DY_SIM[i][:] = np.nan
        XERR[i][:] = np.nan
        YERR[i][:] = np.nan
    EPSERR[:] = np.nan
    DEPS[:] = np.nan
    DEPS_SIM[:] = np.nan

    len_leg, len_tor = calibration.get_len(version)
    ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]

    # current pose
    alp, eps, fpos, p, fix, _ = extract_measurement(db, IDX[start_idx])
#    plot_pose(x, fpos, fix, col='gray')
    eps0 = eps
    x0 = np.array(fpos[0])
    y0 = np.array(fpos[1])

    # ### if straight 3 and left rear foot is not visible
    if np.isnan(fpos[0][3]) and mode == 'straight_3':
        alp3 = 4
        alp[3] = alp3
        X1 = (fpos[0][1], fpos[1][1])
        fpos_ = inverse_kinematics._calc_coords2(ell_n, alp, eps, X1)
        fpos[0][3] = fpos_[0][3]
        fpos[1][3] = fpos_[1][3]
    ##############
    x = alp + ell_n + [eps]
    ##############

    print('\n\nstart pos alp:', [round(a, 2) for a in x[:5]])
    print('start p:', p)

    # corrected current_pose
    alp_c, eps_c, fpos_c = inverse_kinematics.correct_measurement(
            alp, eps, fpos, len_leg=len_leg, len_tor=len_tor)
    x_c = alp_c + ell_n + [eps_c]
#    plot_pose(x_c, fpos_c, fix, col='silver')

    # init gaits
    gait_predicted = roboter_repr.GeckoBotGait(
            roboter_repr.GeckoBotPose(x_c, fpos_c, fix))
    gait_measured = roboter_repr.GeckoBotGait(
            roboter_repr.GeckoBotPose(x_c, fpos_c, fix, fpos_real=fpos))

    # end pose
    alp_n, eps_n, fpos_n, p_n, fix_n, _ = \
        extract_measurement(db, IDX[start_idx+n_predictions])
    x_n = alp_n + ell_n + [eps_n]
#    plot_pose(x_n, fpos_n, fix_n, col='black')
    print('end pos alp:', [round(a, 2) for a in x_n[:5]], '\n')

    # reference & in-between poses
    REF, F = [], [fix]
    for d_idx in range(n_predictions):
        alp_i, eps_i, fpos_i, p_i, fix_i, _ = extract_measurement(db, IDX[start_idx+d_idx+1])
        # collect references
        a_i = [round(a, 2) for a in calibration.get_alpha(p_i, version)]
        for idx_a, a in enumerate(a_i):
            if np.isnan(a):
                a_i[idx_a] = alp_i[idx_a]  # if inv clb not possible, take measured angle
        REF.append([a_i, F[d_idx]])
        F.append(fix_i)
        # collect in between poses
        alp_ic, eps_ic, fpos_ic = inverse_kinematics.correct_measurement(
            alp_i, eps_i, fpos_i, len_leg=len_leg, len_tor=len_tor)
        x_ic = alp_ic + ell_n + [eps_ic]
        gait_measured.append_pose(roboter_repr.GeckoBotPose(
                x_ic, fpos_ic, F[-2], fpos_real=fpos_i))
    

    f_l, f_o, f_a = calibration.get_kin_model_params(mode[-1])


    # Prediction loop
    if not np.isnan(fpos[0]).any():
        # init loop
        x_p, fpos_p = x_c, fpos_c  # predict from the corrected pose
        for d_idx in range(n_predictions): # predicted next pose
            reference = REF[d_idx]
            x_p, fpos_p, fix_p, constraint, cost = \
                kin_model.predict_next_pose(reference, x_p, fpos_p,
                                            f=[f_l, f_o, f_a],
                                            len_leg=len_leg, len_tor=len_tor)
            gait_predicted.append_pose(roboter_repr.GeckoBotPose(x_p, fpos_p, fix_p))
#            plot_pose(x_p, fpos_p, fix_p, col='coral')

            # calc error
            alp_p, eps_p = x_p[0:5], x_p[-1]
            alp_i = gait_measured.poses[d_idx+1].get_alpha()
            alp_err = np.array(alp_i) - np.array(alp_p)
            fpos_i = gait_measured.poses[d_idx+1].fpos_real
            x_err = np.array(fpos_p[0]) - np.array(fpos_i[0])
            y_err = np.array(fpos_p[1]) - np.array(fpos_i[1])
            p_err = np.linalg.norm([x_err, y_err], axis=0)/len_tor*100
            eps_err = eps_p - gait_measured.poses[d_idx+1].get_eps()

            # absolute vals
            deps_m = gait_measured.poses[d_idx+1].get_eps() - eps0
            deps_sim = eps_p - eps0
            dx = np.array(fpos_i[0]) - x0
            dy = np.array(fpos_i[1]) - y0
            dx_sim = np.array(fpos_p[0]) - x0
            dy_sim = np.array(fpos_p[1]) - y0

            dx = np.linalg.norm([dx, dy], axis=0)/len_tor*100
            dx_sim = np.linalg.norm([dx_sim, dy_sim], axis=0)/len_tor*100

            # flip eps
            eps_err = np.mod(eps_err + 180, 360) - 180
            print('eps_err:', eps_err)

            # save error
            for idx in range(6):
                if idx < 5:
                    ALPERR[idx][d_idx] = alp_err[idx]
                PERR[idx][d_idx] = p_err[idx]
                XERR[idx][d_idx] = x_err[idx]
                YERR[idx][d_idx] = y_err[idx]
                DX[idx][d_idx] = dx[idx]
                DY[idx][d_idx] = dy[idx]
                DX_SIM[idx][d_idx] = dx_sim[idx]
                DY_SIM[idx][d_idx] = dy_sim[idx]
            EPSERR[d_idx] = eps_err
            DEPS[d_idx] = deps_m
            DEPS_SIM[d_idx] = deps_sim

#        plot_pose(x_p, fpos_p, fix_p, col='red')

        # plot gaits
        f, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
        plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')
        gait_predicted.plot_gait(fignum=None, ax=axes[1], g=0)
        gait_measured.plot_gait(fignum=None, ax=axes[0], g=1)

        plt.savefig('Out/EXP_'+mode+'_EXPIDX_'+str(exp_idx)
                    + '_startIDX_' + str(start_idx)+'.png', dpi=300)

    return (ALPERR, PERR, EPSERR, (XERR, YERR), gait_predicted, DEPS, DX, DY,
            DEPS_SIM, DX_SIM, DY_SIM)


def calc_errors(db, POSE_IDX, version='vS11', nexps=None, predict_poses=1,
                start_idx=0, mode='1'):
    ALPERR = {i: np.empty((predict_poses+1, len(db))) for i in range(5)}
    PERR = {i: np.empty((predict_poses+1, len(db))) for i in range(6)}
    EPSERR = np.empty((predict_poses+1, len(db)))
    DEPS = np.empty((predict_poses+1, len(db)))
    DX = {i: np.empty((predict_poses+1, len(db))) for i in range(6)}
    DY = {i: np.empty((predict_poses+1, len(db))) for i in range(6)}
    DEPS_SIM = np.empty((predict_poses+1, len(db)))
    DX_SIM = {i: np.empty((predict_poses+1, len(db))) for i in range(6)}
    DY_SIM = {i: np.empty((predict_poses+1, len(db))) for i in range(6)}
    for i in range(6):
        if i < 5:
            ALPERR[i][:] = np.nan
        PERR[i][:] = np.nan
        DX[i][:] = np.nan
        DY[i][:] = np.nan
        DX_SIM[i][:] = np.nan
        DY_SIM[i][:] = np.nan
    EPSERR[:] = np.nan
    DEPS[:] = np.nan
    DEPS_SIM[:] = np.nan
    for exp_idx in range(len(db)):
        EPSERR[0][exp_idx] = 0
        for i in range(6):
            if i < 5:
                ALPERR[i][0][exp_idx] = 0
            PERR[i][0][exp_idx] = 0

    if nexps:
        data = [db[i] for i in nexps]
        POSE_IDX = [POSE_IDX[i] for i in nexps]
    else:
        data = db

    for exp_idx, dset in enumerate(data):
        dset = data[exp_idx]

        plt.figure('Prediction'+str(exp_idx)+'_'+str(start_idx))
        (alp_err, p_err, eps_err, (x_err, y_err), gait_predicted, deps, dx,
         dy, deps_sim, dx_sim, dy_sim) = \
            error_of_prediction(dset, POSE_IDX[exp_idx], start_idx, predict_poses,
                                version, mode, exp_idx)
        for idx in range(6):
            for p_i in range(predict_poses):
                if idx < 5:
                    ALPERR[idx][p_i+1][exp_idx] = alp_err[idx][p_i]
                PERR[idx][p_i+1][exp_idx] = p_err[idx][p_i]
                DX[idx][p_i+1][exp_idx] = dx[idx][p_i]
                DY[idx][p_i+1][exp_idx] = dy[idx][p_i]
                DX_SIM[idx][p_i+1][exp_idx] = dx_sim[idx][p_i]
                DY_SIM[idx][p_i+1][exp_idx] = dy_sim[idx][p_i]
        for p_i in range(predict_poses):
            EPSERR[p_i+1][exp_idx] = eps_err[p_i]
            DEPS[p_i+1][exp_idx] = deps[p_i]
            DEPS_SIM[p_i+1][exp_idx] = deps_sim[p_i]

    ERR_alp_m = {}
    ERR_p_m = {}
    ERR_alp_sig = {}
    ERR_p_sig = {}
    DX_m = {}
    DX_sig = {}
    DXsim_m = {}
    DXsim_sig = {}
    ERR_eps_m, ERR_eps_sig = calc_mean_stddev(EPSERR)
    deps_m, deps_sig = calc_mean_stddev(DEPS)
    deps_sim_m, deps_sim_sig = calc_mean_stddev(DEPS_SIM)
    for i in range(6):
        if i < 5:
            ERR_alp_m[i], ERR_alp_sig[i] = calc_mean_stddev(ALPERR[i])
        ERR_p_m[i], ERR_p_sig[i] = calc_mean_stddev(PERR[i])
        DX_m[i], DX_sig[i] = calc_mean_stddev(DX[i])
        DXsim_m[i], DXsim_sig[i] = calc_mean_stddev(DX_SIM[i])

    return (ERR_alp_m, ERR_p_m, ERR_eps_m, ERR_alp_sig,
            ERR_p_sig, ERR_eps_sig, gait_predicted, DX_m, DX_sig,
            deps_m, deps_sig, deps_sim_m, deps_sim_sig,
            DXsim_m, DXsim_sig)


def plot_pose(x, marks, fix, col='k'):
    pose = roboter_repr.GeckoBotPose(x, marks, fix)
    pose.plot_markers(col=col)
    pose.plot(col)
    plt.axis('equal')


def calc_mean_stddev(mat):
    mu1 = np.nanmean(mat, axis=1)
    sigma1 = np.nanstd(mat, axis=1)
    return mu1, sigma1


def barplot(mu, modes, labels, colors, sig=None,
            save_as_tikz=False, num='errros'):

    width_step = .9
    N = len(modes)

    fig, ax = plt.subplots(num=num)

    rectdic = {}
    lentries = []
    X = np.arange(len(labels))

    for jdx, mode in enumerate(modes):
        w = width_step/N
        x = X + (jdx - (N-1)/2)*w
        col = colors[mode]
        rectdic[mode] = ax.bar(x, mu[mode],
                               yerr=sig[mode] if sig else None,
                               align='center',
                               width=w,
                               ecolor='black', color=col,
                               capsize=10)

        patch = pat.Patch(color=col, label=mode[-5:])  # last 5 chars
        lentries.append(patch)

    plt.legend(handles=lentries)
#    ax.set_ylabel('Number of steps')
#    ax.set_xlabel('Set Point')
    ax.set_xticks([i for i in range(len(labels))])
    ax.set_xticklabels(labels)

    def autolabel(rectdic):
        """Attach a text label above each bar in *rects*,
        displaying its height."""
        for mode in rectdic:
            for rect in rectdic[mode]:
                height = round(rect.get_height(), 1)
                ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
#                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    autolabel(rectdic)
    if save_as_tikz:
        my_save.save_plt_as_tikz('Out/needed_steps.tex')
    return ax


