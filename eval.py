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

    for idx in range(6):
        data['x{}'.format(idx)] = [x-xstart for x in data['x{}'.format(idx)]]
        data['y{}'.format(idx)] = [y-ystart for y in data['y{}'.format(idx)]]

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


def calc_centerpoint(data_set, cycles, axis='x'):
    X = []
    min_dist = min([idx[-1] - idx[0] for idx in cycles])
    markers = ['{}{}'.format(axis, idx) for idx in range(6)]

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
    nSteps = len(data[0])
    nSets = len(data)
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
# small     1 0 1 1 1 1 1 1 1 1 x  x  x
# big       1 1 0 0 1 1 1 1 1 0 0  1  1

ggg = 1
sets = ['{}'.format(idx).zfill(2) for idx in [0,2,3,4,5,6,7,8,9]]
ds, cyc_small = load_data('small_0', sets)

sets = ['{}'.format(idx).zfill(2) for idx in [0,1,4,5,6,7,8,11,12]]
db, cyc_big = load_data('big_0', sets)


color_prs = 'darkslategray'
color_ref = 'lightcoral'
color_alp = 'red'




#plt.figure()
#centers, t = calc_centerpoint(ds, cyc_small)
#for row in centers:
#    plt.plot(row)


plt.figure()
plt.title('Shift in position')


## small
centers, t = calc_centerpoint(ds, cyc_small)
mat = make_matrix_plain(centers)
mu, sigma = calc_mean_stddev(mat)
plt.plot(t, mu, '-', lw=2, label='p_{v0}', color=color_prs)
plt.fill_between(t, mu+sigma, mu-sigma, facecolor=color_prs, alpha=0.2)


## big
centers, t = calc_centerpoint(db, cyc_big)
mat = make_matrix_plain(centers)
mu, sigma = calc_mean_stddev(mat)
plt.plot(t, mu, '-', lw=2, label='p_{v0}', color=color_alp)
plt.fill_between(t, mu+sigma, mu-sigma, facecolor=color_alp, alpha=0.2)


save.save_as_tikz('pics/Shift.tex')

if SAVE:
    plt.savefig('pics/Shift.png', dpi=500, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


"""
________________________________________________
____________________Track___________
________________________________________________
"""



plt.figure()
plt.title('Track')

col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']
X, Y, Xstd, Ystd = calc_foot_mean_of_all_exp(ds, cyc_small)
for idx in range(6):
    x = np.array(X[idx])
    y = -np.array(Y[idx])
    sigx = np.array(Xstd[idx])
    sigy = np.array(Ystd[idx])
    plt.plot(x, y, color=col[idx])
    plt.fill_between(x, y+sigy, y-sigy, facecolor=col[idx], alpha=0.6)


yshift = -230
X, Y, Xstd, Ystd = calc_foot_mean_of_all_exp(db, cyc_big)
for idx in range(6):
    x = np.array(X[idx])
    y = -(np.array(Y[idx]) + yshift)
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




plt.axis('equal')
plt.legend(loc='upper right')


save.save_as_tikz('pics/Track.tex')

if SAVE:
    plt.savefig('pics/Track.png', dpi=500, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


###############################################################################
########################### TRACK MEAN ########################################
###############################################################################



plt.figure()
plt.title('Track Mean')

# Big
yshift = -100
X, _ = calc_centerpoint(db, cyc_big, axis='x')
mat = make_matrix_plain(X)
x, sigx = calc_mean_stddev(mat)

Y, _ = calc_centerpoint(db, cyc_big, axis='y')
mat = make_matrix_plain(Y)
y, sigy = calc_mean_stddev(mat)
y = -(y + yshift)

x = np.array(downsample(x, .01))
y = np.array(downsample(y, .01))
sigy = np.array(downsample(sigy, .01))

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
y = -(y + yshift)


x = np.array(downsample(x, .01))
y = np.array(downsample(y, .01))
sigy = np.array(downsample(sigy, .01))

plt.plot(x, y, color=col[idx])
plt.fill_between(x, y+sigy, y-sigy, facecolor=color_prs, alpha=0.5, label='mean_small')

plt.axis('equal')
plt.legend(loc='upper right')






save.save_as_tikz('pics/Track_Mean.tex')

if SAVE:
    save.save_as_tikz('pics/Track_Mean.png')
    plt.savefig('pics/Track_Mean.png', dpi=500, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)





plt.show()
