# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:42:15 2020

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
from Src import load


def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


# rot: x = cos(a)*x - sin(a)*y
#      y = sin(a)*x + cos(a)*y


def load_data(path, sets, raw=False):
    dataBase = []
#    xscale = 145./1000  # 1000px -> 145cm
#    xshift = -22  # cm
#    yshift = -63  # cm
    xscale = 112./1000  # after changing resolution of RPi
    xshift = -12 - 50  # cm
    yshift = -45 - 20  # cm
    eps_0 = 90  # deg value eps meas is shifted to at start idx

    for exp in sets:
        data = load.read_csv(path+"{}.csv".format(exp))
        if raw:
            dataBase.append(data)
        else:
            try:
                start_idx = data['f0'].index(1)  # upper left foot attached 1sttime
            except ValueError:  # no left foot is fixed
                start_idx = 0

            # correction
            start_time = data['time'][start_idx]

            # shift time acis
            data['time'] = \
                [round(data_time - start_time, 3) for data_time in data['time']]
            for key in data:
                if key[0] in ['x', 'y']:
                    shift = xshift if key[0] == 'x' else yshift
                    data[key] = [i*xscale + shift for i in data[key]]
                if key == 'eps':
                    data['eps'] = [np.mod(e+180, 360)-180+eps_0 for e in data['eps']]

            # shift eps to remove jump
            last_eps = eps_0
            corr_times = 1
            correct_direction = 1
            for idx in range(0, len(data['eps'])):
                eps = data['eps'][idx]
                if not np.isnan(eps):
                    if abs(eps-last_eps) > 200:  # unrealsitic jump in orientation
                        if abs(last_eps - (eps - 360*np.sign(eps)*correct_direction)) > 200:
                            correct_direction = correct_direction*(-1)
                            corr_times += 1
                            print('change eps correction direction\t\t', corr_times)
                        data['eps'][idx] = eps - 360*np.sign(eps)*correct_direction
                    last_eps = data['eps'][idx]

#            # rotate:
            for idx in range(6):
                x = data['x{}'.format(idx)]
                y = data['y{}'.format(idx)]
                X, Y = [], []
                for vec in zip(x, y):
                    xrot, yrot = rotate(vec, np.deg2rad(eps_0))
                    X.append(xrot)
                    Y.append(yrot)
                data['x{}'.format(idx)] = X
                data['y{}'.format(idx)] = Y

            # shift xy coordinates s.t. (x1,y1)(t0) = (0,0)
            start_x1 = (-30, -20)
            if np.isnan(start_x1[0]) or np.isnan(start_x1[1]):
                i = 0
                while np.isnan(start_x1[0]) or np.isnan(start_x1[1]):
                    i -= 1
                    start_x1 = (data['x1'][start_idx+i], data['y1'][start_idx+i])
                    if i < -20:
                        start_x1 = (0, 0)
                        print('can not find start position ...')
            print('Messung startet bei start_x1:  ', start_x1)
            for idx in range(6):
                X = [x - start_x1[0] for x in data['x{}'.format(idx)]]
                Y = [y - start_x1[1] for y in data['y{}'.format(idx)]]
                data['x{}'.format(idx)] = X
                data['y{}'.format(idx)] = Y

            dataBase.append(data)

    return dataBase


def rotate_feet(fpos, theta):
    # rotate:
    x, y = fpos
    X, Y = [], []
    for vec in zip(x, y):
        xrot, yrot = rotate(vec, np.deg2rad(theta))
        X.append(xrot)
        Y.append(yrot)
    return((X, Y))


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
        for jdx in range(idx, idx-100, -1):  # look the last neighbors
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
        if failed > 0:
            print('failed detections of poses:', failed)
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


def plot_pose(x, marks, fix, col='k'):
    pose = roboter_repr.GeckoBotPose(x, marks, fix)
    pose.plot_markers(col=col)
    pose.plot(col)
    plt.axis('equal')


def calc_mean_stddev(mat):
    mu1 = np.nanmean(mat, axis=1)
    sigma1 = np.nanstd(mat, axis=1)
    return mu1, sigma1


def barplot(mu, modes, labels, colors, sig=None, num='errros'):

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

    return ax
