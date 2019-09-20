# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:45:41 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import numpy as np
from Src import calibration
from Src import kin_model
from Src import roboter_repr
from Src import inverse_kinematics





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
    if p[2] > 0:  # left elly actuated
        alp = alp[0:2] + alp[-3:]
    else:  # right belly
        alp = alp[0:2] + [-alp[3]] + alp[-2:]

    return (alp, eps, (fposx, fposy), p, fix, (xref, yref))


def error_of_prediction(db, IDX, start_idx, n_predictions, version='vS11',
                        mode='1'):
    len_leg, len_tor = calibration.get_len(version)
    ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]

    # current pose
    alp, eps, fpos, _, fix, _ = extract_measurement(db, IDX[start_idx])
    x = alp + ell_n + [eps]
    print('start pos alp:', [round(a, 2) for a in x[:5]])

    # corrected current_pose
    alp_c, eps_c, fpos_c = inverse_kinematics.correct_measurement(
            alp, eps, fpos, len_leg=len_leg, len_tor=len_tor)
    x_c = alp_c + ell_n + [eps_c]
    plot_pose(x_c, fpos_c, fix, col='silver')

    # plot currentpose
    plot_pose(x, fpos, fix, col='gray')

    # end pose
    alp_n, eps_n, fpos_n, p_n, fix_n, _ = \
        extract_measurement(db, IDX[start_idx+n_predictions])
    x_n = alp_n + ell_n + [eps_n]
    plot_pose(x_n, fpos_n, fix_n, col='black')
    print('end pos alp:', [round(a, 2) for a in x_n[:5]])



    # reference
    REF, F = [], [fix]
    for d_idx in range(n_predictions):
        aa, _, _, p_i, fix_i, _ = extract_measurement(db, IDX[start_idx+d_idx+1])
        a_i = [round(a, 2) for a in calibration.get_alpha(p_i, version)]
        for idx_a, a in enumerate(a_i):
            if np.isnan(a):
                a_i[idx_a] = aa[idx_a]  # if inv clb not possible, take measured angle
            
        print('ref angle:', a_i)
        print('ref pressure:', [round(p,2) for p in p_i])
        REF.append([a_i, F[d_idx]])
        F.append(fix_i)
    f_l, f_o, f_a = calibration.get_kin_model_params(mode)

    if not np.isnan(fpos[0]).any():
        # init loop
        x_p, fpos_p = x, fpos
        for d_idx in range(n_predictions):
            # predicted next pose
            reference = REF[d_idx]
            x_p, fpos_p, fix_p, constraint, cost = \
                kin_model.predict_next_pose(reference, x_p, fpos_p,
                                            f=[f_l, f_o, f_a],
                                            len_leg=len_leg, len_tor=len_tor)
        plot_pose(x_p, fpos_p, fix_p, col='red')

        # errs
        alp_p, _, eps_p = x_p[0:5], x_p[5:2*5], x_p[-1]
        alp_err = np.array(x_n[:5]) - np.array(alp_p)
        x_err = np.array(fpos_n[0]) - np.array(fpos_p[0])
        y_err = np.array(fpos_n[1]) - np.array(fpos_p[1])
        p_err = np.linalg.norm([x_err, y_err], axis=0)/len_tor
        eps_err = eps_n - eps_p

        return alp_err, p_err, eps_err, cost, (x_err, y_err)
    else:
        return [np.nan]*5, [np.nan]*6, np.nan, np.nan, ([np.nan]*6, [np.nan]*6)


def calc_errors(db, POSE_IDX, version='vS11', nexps=None, predict_poses=1,
                n_runs=None, start_idx=0, mode='1'):
    max_poses = max([len(pidx) for pidx in POSE_IDX]) - predict_poses
    ALPERR = {i: np.empty((max_poses, len(db))) for i in range(5)}
    PERR = {i: np.empty((max_poses, len(db))) for i in range(6)}
    EPSERR = np.empty((max_poses, len(db)))
    for i in range(6):
        if i < 5:
            ALPERR[i][:] = np.nan
        PERR[i][:] = np.nan
    EPSERR[:] = np.nan

    if nexps:
        data = db[:nexps]
    else:
        data = db

    for exp_idx, dset in enumerate(data):
        dset = db[exp_idx]
        if not n_runs:
            n_runs = len(POSE_IDX[exp_idx]) - predict_poses - start_idx - 1
        for pidx in range(start_idx, start_idx+n_runs):
            plt.figure('Prediction'+str(exp_idx)+'_'+str(pidx))
            alp_err, p_err, eps_err, cost, (x_err, y_err) = \
                error_of_prediction(dset, POSE_IDX[exp_idx], pidx, predict_poses,
                                    version, mode)
            for idx in range(6):
                if idx < 5:
                    ALPERR[idx][pidx][exp_idx] = alp_err[idx]
                PERR[idx][pidx][exp_idx] = p_err[idx]
            EPSERR[pidx][exp_idx] = eps_err
            print('p_err: ', [round(p, 2) for p in p_err])
            plt.savefig('Out/Prediction'+str(exp_idx)+'_'+str(pidx)+'png',
                        dpi=300)

    ERR_alp_m = {}
    ERR_p_m = {}
    ERR_alp_sig = {}
    ERR_p_sig = {}
    ERR_eps_m, ERR_eps_sig = calc_mean_stddev(EPSERR)
    for i in range(6):
        if i < 5:
            ERR_alp_m[i], ERR_alp_sig[i] = calc_mean_stddev(ALPERR[i])
        ERR_p_m[i], ERR_p_sig[i] = calc_mean_stddev(PERR[i])

    return ERR_alp_m, ERR_p_m, ERR_eps_m, ERR_alp_sig, ERR_p_sig, ERR_eps_sig


def plot_pose(x, marks, fix, col='k'):
    pose = roboter_repr.GeckoBotPose(x, marks, fix)
    pose.plot_markers(col=col)
    pose.plot(col)


def calc_mean_stddev(mat):
    mu1 = np.nanmean(mat, axis=1)
    sigma1 = np.nanstd(mat, axis=1)
    return mu1, sigma1
