# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:37:21 2019

@author: AmP
"""


import numpy as np
from scipy import signal
from itertools import islice
import traces

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import load


def shift_jump(xlist, jump, minmax=360):
    """ assumes xlist is 'minmax' periodic"""
    x_out = []
    for x in xlist:
        if x < jump:
            x_out.append(x)
        else:
            x_out.append(x-minmax)
    return x_out


def rotate_xy(X, Y, eps):
    s = np.sin(np.deg2rad(eps))
    c = np.cos(np.deg2rad(eps))
    Xout, Yout = [], []
    for x, y in zip(X, Y):
        Xout.append(c*x - s*y)
        Yout.append(s*x + c*y)
    return Xout, Yout


def downsample(rows, proportion=.1):
    return np.array(
            list(islice(list(rows), 0, len(list(rows)), int(1/proportion))))


def resample(t, x, Ts=.03):
    ts = traces.TimeSeries()
    for idx in range(len(t)):
        ts[t[idx]] = x[idx]
    resample = ts.sample(sampling_period=Ts,
                         interpolate='linear')
    t, x = map(list, zip(*resample))
    return t, x


def remove_offset_time_xy(data, Ts):
    # remove time offset
    try:
        start_idx = data['f0'].index(1)  # upper left foot attached 1st time
    except ValueError:  # no left foot is fixed
        start_idx = 0
    start_time = data['time'][start_idx]
    data['time'] = \
        [round(data_time - start_time, 3) for data_time in data['time']]

    # remove offset of xy
    succes, jdx = False, 0
    while not succes:
        if not np.isnan(data['x1'][start_idx-jdx]):
            xstart = data['x1'][start_idx-jdx]
            ystart = data['y1'][start_idx-jdx]
            succes = True
        elif start_idx-jdx < 0:
            xstart, ystart = 0, 0
            break
        else:
            jdx += 1
    sc = 100/640.  # px->cm
    for idx in range(6):
        data['x{}'.format(idx)] = \
            [(x-xstart)*sc for x in data['x{}'.format(idx)]]
        data['y{}'.format(idx)] = \
            [-(y-ystart)*sc for y in data['y{}'.format(idx)]]

    # correction of jump epsilon
    data['eps'] = shift_jump(data['eps'], 180)

    # resample data
    t = data['time']
    for key in data:
        x = data[key]
        _, data[key] = resample(t, x, Ts)

    return data


def scale_alpha(data, scale=1/90.):
    for key in data:
        if key[0] == 'a':
            data[key] = [val*scale for val in data[key]]
    return data


def find_cycle_idx(data):
    # r1 jumps from 0 to some value
    p1 = data['r1']
    idx = [i for i, e in enumerate(p1) if e != 0 and p1[i-1] == 0]

    return idx


def find_dfx_idx(data, foot='f0'):
    # f1 jumps from 1 to some val < 1
    f0 = data[foot]
    idx = [i-1 for i, val in enumerate(f0) if val < 1 and f0[i-1] == 1]

    return idx


def closest_index(lis, val):
    return min(range(len(lis)), key=lambda i: abs(lis[i]-val))


def rm_offset(lis):
    offset = list(lis)[0]
    return [val-offset for val in lis]


def add_offset(lis, offset):
    lis = list(lis)
    return [val+offset for val in lis]


def calc_mean_stddev(mat):
    mu1 = np.nanmean(mat, axis=1)
    sigma1 = np.nanstd(mat, axis=1)
    return mu1, sigma1


def make_matrix(data, cycle_idx):
    start = [cycle_idx[idx]-1 for idx in range(len(cycle_idx)-1)]
    stop = [cycle_idx[idx+1] for idx in range(len(cycle_idx)-1)]
    lens = [sto-sta for sta, sto in zip(start, stop)]
    nSteps = min(lens)
    min_idx = lens.index(nSteps)
    nSets = len(lens)

    mat = np.ndarray((nSteps, nSets))
    for set_ in range(nSets):
        for step in range(nSteps):
            mat[step][set_] = data[start[set_]+step]
    return mat, min_idx


def calc_centerpoint(data_set, cycles, axis='x', marks=range(6)):
    X = []
    min_dist = min([idx[-1] - idx[0] for idx in cycles])
    markers = ['{}{}'.format(axis, idx) for idx in marks]

    for exp in range(len(data_set)):
        start = cycles[exp][0]
        x = []  # list of center in current exp
        for idx in range(start, start+min_dist):
            all_x = [data_set[exp][foot][idx] for foot in markers]  # calc cntr
            x.append(np.nanmean(all_x))
#            x.append(np.mean(all_x))
        X.append(x)     # List of centers in all exp
    t = data_set[exp]['time'][start:start+min_dist]
    return X, t


def make_matrix_plain(data):
    nSets = len(data)
    nSteps = min([len(data[idx]) for idx in range(nSets)])
    mat = np.ndarray((nSteps, nSets))
    for set_ in range(nSets):
        for step in range(nSteps):
            mat[step][set_] = data[set_][step]
    return mat


def calc_foot_mean_of_all_exp(data_set, cycles):
    X, Y, Xstd, Ystd = {}, {}, {}, {}
    min_dist = min([idx[-1] - idx[0] for idx in cycles])
    for foot in range(6):
        x, y, stdx, stdy = [], [], [], []
        for idx in range(min_dist):
            footx = [data_set[exp]['x{}'.format(foot)][cycles[exp][0]+idx]
                     for exp in range(len(data_set))]
            footy = [data_set[exp]['y{}'.format(foot)][cycles[exp][0]+idx]
                     for exp in range(len(data_set))]
            x.append(np.nanmean(footx))
            y.append(np.nanmean(footy))
            stdx.append(np.nanstd(footx))
            stdy.append(np.nanstd(footy))

        X[foot] = x
        Xstd[foot] = stdx
        Y[foot] = y
        Ystd[foot] = stdy
    return X, Y, Xstd, Ystd


def calc_mean_of_axis_in_exp_and_cycle(data_set, cycles, axis='x0'):
    MU, SIG = [], []
    min_len = min([len(cycle) for cycle in cycles])
    cycles = [cycle[0:min_len] for cycle in cycles]  # cut all cycle to min len

    for cycle_idx in range(min_len-1):
        min_dist = min([cycles[exp][cycle_idx+1]-cycles[exp][cycle_idx]
                        for exp in range(len(data_set))])
        mu, sig = [], []
        for idx in range(min_dist):
            x_exp = [data_set[exp][axis][cycles[exp][cycle_idx]+idx]
                     for exp in range(len(data_set))]
            mu.append(np.nanmean(x_exp))
            sig.append(np.nanstd(x_exp))
        MU.append(mu)
        SIG.append(sig)
    return MU, SIG


def calc_mean_of_axis(db, cyc, axis, cyc_index=[1, 2]):
    min_len = min([len(cycle) for cycle in cyc])
    assert (max(cyc_index) < min_len), 'minimal cycle number in given dataset is {}'.format(min_len)
    X = []
    x0 = db[0][axis][cyc[0][cyc_index[0]]]
    for idx in cyc_index:
        for exp in range(len(db)):
            exp_cyc = cyc[exp]
            x_ = db[exp][axis][exp_cyc[idx]:exp_cyc[idx+1]]
#            x_ = add_offset(rm_offset(x_), x0)
            X.append(x_)
    mat = make_matrix_plain(X)
#    print(np.shape(mat))
    xx, sigxx = calc_mean_stddev(mat)
    return xx, sigxx


def calc_mean_of_axis_multi_cyc(db, cyc, axis, skipfirst=1, skiplast=None):
    x, sigx = [], []
    min_len = min([len(cycle) for cycle in cyc])
    last_idx = min_len - skiplast - 1 if skiplast else min_len - 1
    for idx in range(skipfirst, last_idx):
        xx, sigxx = calc_mean_of_axis(db, cyc, axis, [idx])
        x = np.array(list(x)+list(xx))
        sigx = np.array(list(sigx)+list(sigxx))
    return x, sigx


def calc_mean_of_axis_for_all_exp_and_cycles(data, cyc, axis,
                                             skipfirstlast=(0, 0)):
    x, sigx = calc_mean_of_axis_in_exp_and_cycle(data, cyc, axis)
    X = []
    x0 = x[skipfirstlast[0]][0]
    for x_ in x:
        x_ = add_offset(rm_offset(x_), x0)
        X.append(x_)
    X = (X[skipfirstlast[0]:-skipfirstlast[1]]
         if skipfirstlast[1] != 0 else X[skipfirstlast[0]:])
    mat = make_matrix_plain(X)
    print(np.shape(mat))
    xx, sigxx = calc_mean_stddev(mat)
    return xx, sigxx


def load_data(exp_name, exp_idx=['00'], Ts=.03):
    dset, Cycles = [], []
    for exp in exp_idx:
        data = load.read_csv(exp_name+"{}.csv".format(exp))
        data = remove_offset_time_xy(data, Ts)
        cycle = find_cycle_idx(data)
        dset.append(data)

        Cycles.append(cycle)
    return dset, Cycles


def load_data_pathPlanner(exp_name, exp_idx=['00'], Ts=.03):
    dataBase = []
    for exp in exp_idx:
        data = load.read_csv(exp_name+"{}.csv".format(exp))

        try:
            start_idx = data['f0'].index(1)  # upper left foot attached 1sttime
        except ValueError:  # no left foot is fixed
            start_idx = 0
        start_time = data['time'][start_idx]
        data['time'] = \
            [round(data_time - start_time, 3) for data_time in data['time']]

        dataBase.append(data)

    return dataBase



def calc_velocity(db, Ts):
    b, a = signal.butter(3, 0.05)
    for exp in range(len(db)):
        for axis in range(6):
            noisy = np.diff(db[exp]['x{}'.format(axis)])/Ts
#            y = signal.filtfilt(b, a, noisy)
            x = signal.convolve(noisy, [0.1]*10, mode='valid')
            db[exp]['x{}dot'.format(axis)] = np.concatenate(([0], x))

            noisy = np.diff(db[exp]['y{}'.format(axis)])/Ts
#            y = signal.filtfilt(b, a, noisy)
            y = signal.convolve(noisy, [0.1]*10, mode='valid')
            db[exp]['y{}dot'.format(axis)] = np.concatenate(([0], y))
    return db


def get_marker_color():
    return ['red', 'orange', 'darkred', 'blue', 'darkorange', 'darkblue']


def get_actuator_color():
    return ['red', 'darkred', 'darkorange', 'orange', 'blue', 'darkblue']


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data, cyc = load_data('2019_02_07_big_initial_cycle_exp/', ['000'])
    t = data[0]['time']

    tt, ts = resample(t, t)

    plt.plot(t, t)
    plt.plot(ts, ts)

    plt.figure()
    plt.plot(np.diff(t))
    plt.plot(np.diff(ts))
