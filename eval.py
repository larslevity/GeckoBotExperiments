# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:37:21 2019

@author: AmP
"""

import matplotlib.pyplot as plt
import numpy as np
from itertools import islice

import load
import save
import kin_model

SAVE = False


def downsample(rows, proportion=.1):
    return list(islice(rows, 0, len(rows), int(1/proportion)))


def remove_offset_time_xy(data):
    start_idx = data['f0'].index(1)  # upper left foot attached 1st time
    start_time = data['time'][start_idx]
    data['time'] = \
        [round(data_time - start_time, 3) for data_time in data['time']]

    succes, jdx = False, 0
    while not succes:
        if not np.isnan(data['x0'][start_idx-jdx]):
            xstart = data['x0'][start_idx-jdx]
            ystart = data['y0'][start_idx-jdx]
            succes = True
        elif start_idx-jdx < 0:
            xstart, ystart = 0, 0
            break
        else:
            jdx += 1

    sc = 100/640.  # px->cm
    for idx in range(6):
        data['x{}'.format(idx)] = [(x-xstart)*sc for x in data['x{}'.format(idx)]]
        data['y{}'.format(idx)] = [-(y-ystart)*sc for y in data['y{}'.format(idx)]]

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


def rm_offset(lis):
    offset = lis[0]
    return [val-offset for val in lis]


def add_offset(lis, offset):
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
            all_x = [data_set[exp][foot][idx] for foot in markers]  # calc center
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
            footx = [data_set[exp]['x{}'.format(foot)][cycles[exp][0]+idx] for exp in range(len(data_set))]
            footy = [data_set[exp]['y{}'.format(foot)][cycles[exp][0]+idx] for exp in range(len(data_set))]
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
        min_dist = min([cycles[exp][cycle_idx+1]-cycles[exp][cycle_idx] for exp in range(len(data_set))])
        mu, sig = [], []
        for idx in range(min_dist):
            x_exp = [data_set[exp][axis][cycles[exp][cycle_idx]+idx] for exp in range(len(data_set))]
            mu.append(np.nanmean(x_exp))
            sig.append(np.nanstd(x_exp))
        MU.append(mu)
        SIG.append(sig)
    return MU, SIG


def load_data(exp_name, exp_idx=['00']):
    dset, Cycles = [], []
    for exp in exp_idx:
        data = load.read_csv(exp_name+"_{}.csv".format(exp))
        cycle = find_cycle_idx(data)
        data = remove_offset_time_xy(data)
        dset.append(data)

        Cycles.append(cycle)
    return dset, Cycles


"""
________________________________________________
____________________Shift in Position___________
________________________________________________
"""

# exp 0 data qualilty:

#           0 1 2 3 4 5 6 7 8 9 10 11 12
# small     1 0 1 1 1 1 x 1 1 1 x  x  x
# big       1 1 0 0 1 1 1 1 1 0 0  1  1

ggg = 1
sets = ['{}'.format(idx).zfill(2) for idx in [0,2,3,4,5,7,8,9]]
ds, cyc_small = load_data('small_0', sets)

sets = ['{}'.format(idx).zfill(2) for idx in [0,1,4,5,6,7,8,11,12]]
db, cyc_big = load_data('big_0', sets)


color_prs = 'darkslategray'
color_ref = 'lightcoral'
color_alp = 'red'


if 0:

    plt.figure()
    # # small
    centers, t = calc_centerpoint(ds, cyc_small)
    mat = make_matrix_plain(centers)
    mu, sigma = calc_mean_stddev(mat)
    plt.plot(t, mu, '-', lw=2, label='p_{v0}', color=color_prs)
    plt.fill_between(t, mu+sigma, mu-sigma, facecolor=color_prs, alpha=0.2)

    # # big
    centers, t = calc_centerpoint(db, cyc_big)
    mat = make_matrix_plain(centers)
    mu, sigma = calc_mean_stddev(mat)
    plt.plot(t, mu, '-', lw=2, label='p_{v0}', color=color_alp)
    plt.fill_between(t, mu+sigma, mu-sigma, facecolor=color_alp, alpha=0.2)

    plt.xlabel('time (s)')
    plt.ylabel('$\bar{x}$ (cm)')
    plt.grid()

    save.save_as_tikz('pics/Shift.tex')

    if SAVE:
        plt.savefig('pics/Shift.png', dpi=500, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)


###############################################################################
# ################## ALL CYCLE ANALYSE ########################################
###############################################################################

def flat_list(l):
    return [item for sublist in l for item in sublist]


if 0:
    plt.figure()
    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    yshift = -50
    for axis in [4, 0, 1, 3, 2, 5]:
        x, sigx = calc_mean_of_axis_in_exp_and_cycle(db, cyc_big, axis='x{}'.format(axis))
        y, sigy = calc_mean_of_axis_in_exp_and_cycle(db, cyc_big, axis='y{}'.format(axis))
        xxx = x
        x = flat_list(x)
        y = (np.array(flat_list(y)) + yshift)
        sigy = np.array(flat_list(sigy))
        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)

        # Small
        x, sigx = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='x{}'.format(axis))
        y, sigy = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='y{}'.format(axis))
        x = flat_list(x)
        y = (np.array(flat_list(y)))
        sigy = np.array(flat_list(sigy))
        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)


    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axis('equal')


plt.show()


###############################################################################
# ################## SINGLE CYCLE ANALYSE #####################################
###############################################################################


def calc_mean_of_axis_for_all_exp_and_cycles(data, cyc, axis, skipfirstlast=(0, 0)):
    x, sigx = calc_mean_of_axis_in_exp_and_cycle(data, cyc, axis)
    X = []
    x0 = x[skipfirstlast[0]][0]
    for x_ in x:
        x_ = add_offset(rm_offset(x_), x0)
        X.append(x_)
    X = X[skipfirstlast[0]:-skipfirstlast[1]] if skipfirstlast[1]!=0 else X[skipfirstlast[0]:]
    mat = make_matrix_plain(X)
    xx, sigxx = calc_mean_stddev(mat)
    return xx, sigxx


if 1:
    plt.figure()
    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    yshift = -50
    skip_first = 1
    skip_last = 1

    positions = [{}, {}]
    alpha = {}

    for axis in [0, 1, 2, 3, 4, 5]:
        x, sigx = calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc_big, 'x{}'.format(axis), (skip_first, skip_last))
        y, sigy = calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc_big, 'y{}'.format(axis), (skip_first, skip_last))
        a, siga = calc_mean_of_axis_for_all_exp_and_cycles(
                db, cyc_big, 'aIMG{}'.format(axis), (skip_first, skip_last))

        plt.plot(x, y, color=col[axis])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)

        alpha[axis] = a
        positions[0][axis] = x
        positions[1][axis] = y
    eps, sige = calc_mean_of_axis_for_all_exp_and_cycles(
            db, cyc_big, 'eps', (skip_first, skip_last))

    # ####### plot gecko in first, mid and end position:
    for jdx, idx in enumerate([0, len(eps)/2, len(eps)-1]):
        pos = ([positions[0][axis][idx] for axis in range(6)],
               [-positions[1][axis][idx] for axis in range(6)])  # mind minus
        alp = [alpha[axis][idx] for axis in range(6)]
        alp_ = alp[0:2] + [-alp[3]] + alp[4:6]
        eps_ = 3   # cheat

        pose, ell, alp__ = kin_model.extract_pose(alp_, eps_, pos)
        pose = (pose[0], [-val for val in pose[1]])         # flip again
        plt.plot(pose[0], pose[1], '.', color=col[jdx])
        print 'idx: ', ell, [a_ - a__ for a_, a__ in zip(alp_, alp__)]

#        # Small
#        x, sigx = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='x{}'.format(axis))
#        y, sigy = calc_mean_of_axis_in_exp_and_cycle(ds, cyc_small, axis='y{}'.format(axis))
#        x = flat_list(x)
#        y = (np.array(flat_list(y)))
#        sigy = np.array(flat_list(sigy))
#        plt.plot(x, y, color=col[axis])
#        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[axis], alpha=0.5)







    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axis('equal')
    plt.grid()


plt.show()


"""
________________________________________________
____________________Track of feet___________
________________________________________________
"""

if 0:
    plt.figure()
    plt.title('Track')

    axes = range(6)

    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
    X, Y, Xstd, Ystd = calc_foot_mean_of_all_exp(ds, cyc_small)
    for idx in axes:
        x = np.array(X[idx])
        y = np.array(Y[idx])
        sigx = np.array(Xstd[idx])
        sigy = np.array(Ystd[idx])
        plt.plot(x, y, color=col[idx])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[idx], alpha=0.6)

    yshift = -50
    X, Y, Xstd, Ystd = calc_foot_mean_of_all_exp(db, cyc_big)
    for idx in axes:
        x = np.array(X[idx])
        y = (np.array(Y[idx]) + yshift)
        sigx = np.array(Xstd[idx])
        sigy = np.array(Ystd[idx])
        plt.plot(x, y, color=col[idx])
        plt.fill_between(x, y+sigy, y-sigy, facecolor=col[idx], alpha=0.5, label='mark_{}'.format(idx))

    ######## plot gecko in first position:
    #exp = 0
    #cycle = 3
    #idx = cyc_big[exp][cycle]
    #positions = ([db[exp]['x{}'.format(foot)][idx] for foot in range(6)],
    #             [db[exp]['y{}'.format(foot)][idx] for foot in range(6)])
    #alpha = [db[exp]['aIMG{}'.format(foot)][idx] for foot in range(6)]
    #eps = db[exp]['eps'][idx]
    #
    #
    #alpha_ = alpha[0:3] + alpha[4:6]
    #pose, ell, bet = kinematic.extract_pose(alpha_, eps, positions)
    #
    #

    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.axis('equal')


#    save.save_as_tikz('pics/Track.tex')

    if SAVE:
        plt.savefig('pics/Track.png', dpi=500, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)

###############################################################################
# ########################## TRACK MEAN #######################################
###############################################################################

if 0:
    plt.figure()
    plt.title('Track Mean')
    smapleval = .8


    # Big
    yshift = -20
    X, _ = calc_centerpoint(db, cyc_big, axis='x')
    mat = make_matrix_plain(X)
    x, sigx = calc_mean_stddev(mat)

    Y, _ = calc_centerpoint(db, cyc_big, axis='y')
    mat = make_matrix_plain(Y)
    y, sigy = calc_mean_stddev(mat)
    y = (y + yshift)

    x = np.array(downsample(x, smapleval))
    y = np.array(downsample(y, smapleval))
    sigy = np.array(downsample(sigy, smapleval))

    plt.plot(x, y, color=col[idx])
    plt.fill_between(x, y+sigy, y-sigy, facecolor=color_alp, alpha=0.5, label='mean_big')

    # Mean Small
    yshift = 0
    X, _ = calc_centerpoint(ds, cyc_small, axis='x')
    mat = make_matrix_plain(X)
    x, sigx = calc_mean_stddev(mat)

    Y, _ = calc_centerpoint(ds, cyc_small, axis='y')
    mat = make_matrix_plain(Y)
    y, sigy = calc_mean_stddev(mat)
    y = (y + yshift)

    x = np.array(downsample(x, smapleval))
    y = np.array(downsample(y, smapleval))
    sigy = np.array(downsample(sigy, smapleval))

    plt.plot(x, y, color=col[idx])
    plt.fill_between(x, y+sigy, y-sigy, facecolor=color_prs, alpha=0.5, label='mean_small')

    plt.axis('equal')
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    plt.grid()

    save.save_as_tikz('pics/Track_Mean.tex')

    if SAVE:
        save.save_as_tikz('pics/Track_Mean.png')
        plt.savefig('pics/Track_Mean.png', dpi=500, facecolor='w',
                    edgecolor='w', orientation='portrait', papertype=None,
                    format=None, transparent=False, bbox_inches=None,
                    pad_inches=0.1, frameon=None, metadata=None)



