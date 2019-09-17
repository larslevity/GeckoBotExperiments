# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:04:03 2019

@author: AmP
"""

import numpy as np

from Src import load
from Src import inverse_kinematics
from Src import kin_model
from Src import calibration


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
                                failed +=1
                                pose_idx.append(idx)  # append ori
                                break
                            else:
                                pose_idx.append(jdx)
                                break
                        elif jdx == idx-neighbors+1:
                            failed += 1
                            pose_idx.append(idx)  # append ori
        #last#
        idx = len(db[exp_idx]['r3'])-1
        for jdx in range(idx, idx-100, -1):  # look the last neigbors
            if not np.isnan(db[exp_idx]['aIMG2'][jdx]):
                # check
                dr = db[exp_idx]['r2'][idx] - db[exp_idx]['r2'][jdx]
                if abs(dr) > .1:
                    failed +=1
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
    p = [measurement['p{}'.format(j)][idx] for j in range(6)]
    fix = [measurement['f{}'.format(j)][idx] for j in range(4)]
    eps = measurement['eps'][idx]
    xref = measurement['x7'][idx]
    yref = measurement['y7'][idx]
    return (alp, eps, (fposx, fposy), p, fix, (xref,yref))



def error_of_prediction(db, current_pose_idx, next_pose_idx, version='vS11'):
    len_leg, len_tor = calibration.get_len(version)
    alp, eps, fpos, _, _, _ = extract_measurement(db, current_pose_idx)

    alp_c, eps_, fpos_c = inverse_kinematics.correct_measurement(
            alp, eps, fpos, len_leg=len_leg, len_tor=len_tor)

    alp_n, eps_n, fpos_n, p_n, fix_n, _ = extract_measurement(db, next_pose_idx)
    alpref_n = calibration.get_alpha(p_n, version)
    alp = alp[:3] + alp[-2:]
    alp_n = alp_n[:3] + alp_n[-2:]
    x = alp + [len_leg, len_leg, len_tor, len_leg, len_leg] + [eps]
    reference = (alpref_n, fix_n)
    
    x, fpos_p, fix, constraint, cost = \
        kin_model.predict_next_pose(reference, x, fpos_n, f=[f_l, f_o, f_a],
                      len_leg=len_leg, len_tor=len_tor)
    alp_p, ell_p, eps_p = x[0:5], x[5:2*5], x[-1]
    
    
    alp_err = np.array(alp_n) - np.array(alp_p)
    print('alperr:', alp_err)
    alp_err = np.linalg.norm(alp_err)
    
    return alp_err


def calc_errors(db, POSE_IDX):
    for exp_idx in [0]:
        dset = db[exp_idx]
        pose_idx = POSE_IDX[exp_idx]
        for idx in  [2]:  # range(len(pose_idx)-1):
            alp_err = error_of_prediction(dset, pose_idx[idx], pose_idx[idx+1])
            print('alp prediction error:', alp_err)
    return alp_err




