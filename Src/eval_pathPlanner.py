# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:04:03 2019

@author: AmP
"""

import numpy as np
import matplotlib.pyplot as plt


from Src import load
from Src import inverse_kinematics
from Src import kin_model
from Src import calibration
from Src import roboter_repr
from Src import plot_fun_pathPlanner as pf


f_l = 100.      # factor on length objective
f_o = 0.1     # .0003     # factor on orientation objective
f_a = 10        # factor on angle objective



def load_data_pathPlanner(path, sets):
    dataBase = []
#    xscale = 145./1000  # 1000px -> 145cm
#    xshift = -22  # cm
#    yshift = -63  # cm
    xscale = 112./1000  # after changing resolution of RPi
    xshift = -12  # cm
    yshift = -45  # cm

    for exp in sets:
        data = load.read_csv(path+"{}.csv".format(exp))

        try:
            start_idx = data['f0'].index(1)  # upper left foot attached 1sttime
        except ValueError:  # no left foot is fixed
            start_idx = 0
        start_time = data['time'][start_idx]
        data['time'] = \
            [round(data_time - start_time, 3) for data_time in data['time']]
        for key in data:
            if key[0] in ['x', 'y']:
                shift = xshift if key[0] == 'x' else yshift
                data[key] = [i*xscale + shift for i in data[key]]

        dataBase.append(data)

    return dataBase


def find_poses_idx(db, r3_init=.44, neighbors=5):
    IDX = []
    failed = 0
    for exp_idx in range(len(db)):
        pose_idx = []
        start_idx = db[exp_idx]['r3'].index(r3_init)
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
    return (alp, eps, (fposx, fposy), p, fix, (xref, yref))


def error_of_prediction(db, current_pose_idx, next_pose_idx, version='vS11'):
    len_leg, len_tor = calibration.get_len(version)
    ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]

    # current pose
    alp, eps, fpos, _, fix, _ = extract_measurement(db, current_pose_idx)
    x = alp[:3] + alp[-2:] + ell_n + [eps]
    plot_pose(x, fpos, fix, col='black')

#    # corrected current_pose
#    alp_c, eps_c, fpos_c = inverse_kinematics.correct_measurement(
#            alp, eps, fpos, len_leg=len_leg, len_tor=len_tor)
#    x_c = alp_c[:3] + alp_c[-2:] + ell_n + [eps_c]
#    plot_pose(x_c, fpos_c, fix, col='gray')

    # next pose
    alp_n, eps_n, fpos_n, p_n, fix_n, _ = \
        extract_measurement(db, next_pose_idx)
    alpref_n = calibration.get_alpha(p_n, version)
#    if alpref_n[2] < 0:
#        alpref_n[2] += 15
    alp_n = alp_n[:3] + alp_n[-2:]
    x_n = alp_n[:3] + alp_n[-2:] + ell_n + [eps_n]
    plot_pose(x_n, fpos_n, fix_n, col='blue')

    if not np.isnan(fpos[0]).any():
        # predicted next pose
        reference = (alpref_n, fix)
    #    print('reference:', reference)
        x_p, fpos_p, fix_p, constraint, cost = \
            kin_model.predict_next_pose(reference, x, fpos, f=[f_l, f_o, f_a],
                          len_leg=len_leg, len_tor=len_tor)
        alp_p, ell_p, eps_p = x_p[0:5], x_p[5:2*5], x_p[-1]
        plot_pose(x_p, fpos_p, fix_p, col='red')
    
        # errs
        alp_err = np.array(alp_n) - np.array(alp_p)
        x_err = np.array(fpos_n[0]) - np.array(fpos_p[0])
        y_err = np.array(fpos_n[1]) - np.array(fpos_p[1])
        p_err = np.linalg.norm([x_err, y_err], axis=0)/len_tor
        eps_err = eps_n - eps_p

        return alp_err, p_err, eps_err, cost, (x_err, y_err)
    else:
        return [np.nan]*5, [np.nan]*6, np.nan, np.nan, ([np.nan]*6, [np.nan]*6)


def calc_errors(db, POSE_IDX):
    max_poses = max([len(pidx) for pidx in POSE_IDX])
    ALPERR = {i: np.empty((max_poses, len(db))) for i in range(5)}
    PERR = {i: np.empty((max_poses, len(db))) for i in range(6)}
    EPSERR = np.empty((max_poses, len(db)))
    for i in range(6):
        if i < 5:
            ALPERR[i][:] = np.nan
        PERR[i][:] = np.nan
    EPSERR[:] = np.nan

    for exp_idx, dset in enumerate(db):
        dset = db[exp_idx]
        for pidx, pose_idx in enumerate(POSE_IDX[exp_idx][:-1]):
            alp_err, p_err, eps_err, cost, (x_err, y_err) = \
                error_of_prediction(dset, pose_idx, POSE_IDX[exp_idx][pidx+1])
            for idx in range(6):
                if idx < 5:
                    ALPERR[idx][pidx][exp_idx] = alp_err[idx]
                PERR[idx][pidx][exp_idx] = p_err[idx]
            EPSERR[pidx][exp_idx] = eps_err
            print('p_err: ', [round(p, 2) for p in p_err])

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


