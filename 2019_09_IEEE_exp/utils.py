# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:45:41 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as pat

from Src import calibration
from Src import kin_model
from Src import roboter_repr
from Src import inverse_kinematics
from Src import save as my_save


def ieee_find_poses_idx(db, neighbors=5):
    IDX = []
    failed = 0
    for exp_idx in range(len(db)):
        pose_idx = []
        start_idx = db[exp_idx]['f1'].index(1)
        for idx in range(start_idx, len(db[exp_idx]['r3'])-1, 1):
            if db[exp_idx]['r3'][idx] != db[exp_idx]['r3'][idx+1]:
                if not pose_idx:  # empty list
                    pose_idx.append(idx)
                else:
                    for jdx in range(idx, idx-neighbors, -1):  # look the last neigbors
                        if not np.isnan(db[exp_idx]['aIMG2'][jdx]):
                            # check
                            dr = db[exp_idx]['r2'][idx] - db[exp_idx]['r2'][jdx]
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
        idx = len(db[exp_idx]['r3'])-1
        for jdx in range(idx, idx-100, -1):  # look the last neigbors
            if not np.isnan(db[exp_idx]['aIMG2'][jdx]):
                # check
                dr = db[exp_idx]['r2'][idx] - db[exp_idx]['r2'][jdx]
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
    p = [measurement['r{}'.format(j)][idx] for j in range(6)]
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
    for i in range(6):
        if i < 5:
            ALPERR[i][:] = np.nan
        PERR[i][:] = np.nan
        XERR[i][:] = np.nan
        YERR[i][:] = np.nan
    EPSERR[:] = np.nan

    len_leg, len_tor = calibration.get_len(version)
    ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]

    # current pose
    alp, eps, fpos, p, fix, _ = extract_measurement(db, IDX[start_idx])
    x = alp + ell_n + [eps]
#    plot_pose(x, fpos, fix, col='gray')

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
            EPSERR[d_idx] = eps_err
            
            
#        plot_pose(x_p, fpos_p, fix_p, col='red')
        
        # plot gaits
        f, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
        plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')
        gait_predicted.plot_gait(fignum=None, ax=axes[1], g=0)
        gait_measured.plot_gait(fignum=None, ax=axes[0], g=1)

        plt.savefig('Out/EXP_'+mode+'_EXPIDX_'+str(exp_idx)+'_startIDX_'+str(start_idx)+'.png', dpi=300)


    return ALPERR, PERR, EPSERR, (XERR, YERR), gait_predicted
    

def calc_errors(db, POSE_IDX, version='vS11', nexps=None, predict_poses=1,
                start_idx=0, mode='1'):
    ALPERR = {i: np.empty((predict_poses+1, len(db))) for i in range(5)}
    PERR = {i: np.empty((predict_poses+1, len(db))) for i in range(6)}
    EPSERR = np.empty((predict_poses+1, len(db)))
    for i in range(6):
        if i < 5:
            ALPERR[i][:] = np.nan
        PERR[i][:] = np.nan
    EPSERR[:] = np.nan
    for exp_idx in range(len(db)):
        EPSERR[0][exp_idx] = 0
        for i in range(6):
            if i < 5:
                ALPERR[i][0][exp_idx] = 0
            PERR[i][0][exp_idx] = 0
    

    if nexps:
        data = [db[i] for i in nexps]
    else:
        data = db

    for exp_idx, dset in enumerate(data):
        dset = db[exp_idx]

       
        plt.figure('Prediction'+str(exp_idx)+'_'+str(start_idx))
        alp_err, p_err, eps_err, (x_err, y_err), gait_predicted = \
            error_of_prediction(dset, POSE_IDX[exp_idx], start_idx, predict_poses,
                                version, mode, exp_idx)
        for idx in range(6):
            for p_i in range(predict_poses):
                if idx < 5:
                    ALPERR[idx][p_i+1][exp_idx] = alp_err[idx][p_i]
                PERR[idx][p_i+1][exp_idx] = p_err[idx][p_i]
        for p_i in range(predict_poses):
            EPSERR[p_i+1][exp_idx] = eps_err[p_i]

    ERR_alp_m = {}
    ERR_p_m = {}
    ERR_alp_sig = {}
    ERR_p_sig = {}
    ERR_eps_m, ERR_eps_sig = calc_mean_stddev(EPSERR)
    for i in range(6):
        if i < 5:
            ERR_alp_m[i], ERR_alp_sig[i] = calc_mean_stddev(ALPERR[i])
        ERR_p_m[i], ERR_p_sig[i] = calc_mean_stddev(PERR[i])

    return (ERR_alp_m, ERR_p_m, ERR_eps_m, ERR_alp_sig,
            ERR_p_sig, ERR_eps_sig, gait_predicted)


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
                               yerr=sig[mode],
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


