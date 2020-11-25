#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:51:58 2020

@author: ls
"""


import matplotlib.pyplot as plt
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Src import save as my_save
import plotfun_GaitLawExp as pf
import gait_law_utils as uti
from Src import calibration
from Src import inverse_kinematics
from Src import roboter_repr
from Src import load
from Src import kin_model as model


# %% LOAD
version = 'vS11'
styles = ['-']

exp_path = 'c110_redo_slow/q1_80q2_-05'

# load simulation
RESULT = load.load_data('sim_result.h5')
REF = load.load_data('sim_ref.h5')
TIME = load.load_data('sim_time.h5')


len_leg, len_tor = calibration.get_len(version)
ell_n = [len_leg, len_leg, len_tor, len_leg, len_leg]

DRAW_SHOTS = False


def find_poses_idx(db):
    POSE_IDX = []
    for data in db:
        # f1 or f2 jumps from 0 to 1
        f1 = data['f0']
        f2 = data['f1']
        idx = [i for i, (v1, v2) in enumerate(zip(f1, f2))
               if (v1 == 1 and f1[i-1] == 0) or (v2 == 1 and f2[i-1] == 0)]
        POSE_IDX.append(idx)
    return POSE_IDX


def load_data(path, sets, start_cycle=0, raw=0, eps_0=90):
    dataBase = []
#    xscale = 145./1000  # 1000px -> 145cm
#    xshift = -22  # cm
#    yshift = -63  # cm
    xscale = 112./1000  # after changing resolution of RPi
    xshift = -12 - 50  # cm
    yshift = -45 - 20  # cm
    eps_0 = eps_0  # deg value eps meas is shifted to at start idx

    for exp in sets:
        data = load.read_csv(path+"{}.csv".format(exp))
        if raw:
            dataBase.append(data)
            continue

        try:
            p1 = data['f1']
            idx = [i for i, v in enumerate(p1) if v == 1 and p1[i-1] == 0]
            start_idx = idx[start_cycle]
#            start_idx = data['f0'].index(1)  # upper left foot attached 1sttime
        except ValueError:  # no left foot is fixed
            start_idx = 0

        # correction
        start_time = data['time'][start_idx]
        start_eps = data['eps'][start_idx]
        start_eps_idx = start_idx
        if np.isnan(start_eps):
            i = 0
            print('eps measurment is corrupt at start_idx')
            while np.isnan(start_eps):
                i += 1
                start_eps = data['eps'][start_idx+i]
                start_eps_idx = start_idx+i
            print('took start_eps from idx', start_idx + i,
                  '(start_idx: ', start_idx, ')')
            if i > 10:
                print('THAT ARE MORE THAN 10 MEASUREMENTS!!\n\n')

        # shift time acis
        data['time'] = \
            [round(data_time - start_time, 3) for data_time in data['time']]
        for key in data:
            if key[0] in ['x', 'y']:
                shift = xshift if key[0] == 'x' else yshift
                data[key] = [i*xscale + shift for i in data[key]]
            if key == 'eps':
                data['eps'] = \
                 [np.mod(e+180-start_eps, 360)-180+eps_0 for e in data['eps']]

        # shift eps to remove jump
        last_eps = eps_0
        corr_times = 1
        correct_direction = 1
        for idx in range(start_eps_idx+1, len(data['eps'])):
            eps = data['eps'][idx]
            if not np.isnan(eps):
                if abs(eps-last_eps) > 200:  # unrealsitic jump in orientation
                    if abs(last_eps -
                           (eps - 360*np.sign(eps)*correct_direction)) > 200:
                        correct_direction = correct_direction*(-1)
                        corr_times += 1
                        print('change eps correction direction\t\t',
                              corr_times)
                    data['eps'][idx] = eps - 360*np.sign(eps)*correct_direction
                last_eps = data['eps'][idx]

        # rotate:
        for idx in range(6):
            x = data['x{}'.format(idx)]
            y = data['y{}'.format(idx)]
            X, Y = [], []
            for vec in zip(x, y):
                xrot, yrot = uti.rotate(vec, np.deg2rad(-start_eps+eps_0))
                X.append(xrot)
                Y.append(yrot)
            data['x{}'.format(idx)] = X
            data['y{}'.format(idx)] = Y

        # shift xy coordinates s.t. (x1,y1)(t0) = (0,0)
        start_x1 = (data['x1'][start_idx], data['y1'][start_idx])
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



# %% LETS GO



dirpath = exp_path + '/'
sets = load.get_csv_set(dirpath)
start_cycle = 1
db = load_data(dirpath, sets, start_cycle, raw=0)

# %% Plot track for first impression

db_ = db  #[db[0]]

POSE_IDX = find_poses_idx(db_)
POSE_IDX = [idx[start_cycle*2+1:start_cycle*2+4] for idx in POSE_IDX]
print('plot track....')
pf.plot_track(db_, POSE_IDX, '', version,
              save_as_tikz='Out/'+exp_path+'/02.tex')
pf.plot_eps(db_, POSE_IDX, '', version,
            save_as_tikz='Out/'+exp_path+'/03.tex')


# %% # mean of epsilon

def make_matrix_plain(data):
    nSets = len(data)
    nSteps = min([len(data[idx]) for idx in range(nSets)])
    mat = np.ndarray((nSteps, nSets))
    for set_ in range(nSets):
        for step in range(nSteps):
            mat[step][set_] = data[set_][step]
    return mat

def calc_mean_stddev(mat, axis=1):
    mu1 = np.nanmean(mat, axis=axis)
    sigma1 = np.nanstd(mat, axis=axis)
    return mu1, sigma1

def calc_mean_of_axis(db, POSE_IDX, axis, startend=[0, 2]):
    X = []
    for exp in range(len(db)):
        start = POSE_IDX[exp][startend[0]]
        end = POSE_IDX[exp][startend[1]]
        x_ = db[exp][axis][start:end]
        X.append(x_)
    mat = make_matrix_plain(X)
#    print(mat.shape)
    xx, sigxx = calc_mean_stddev(mat)
    return xx, sigxx

def rm_offset(lis):
    offset = list(lis)[0]
    return [val-offset for val in lis]

def calc_prop(db):
    len_exp = min([len(dset['pr1']) for dset in db])
    prop = 400./len_exp
    prop = .99 if prop >= 1. else prop
    return prop

prop = calc_prop(db)

# ### eps during cycle
eps, sige = calc_mean_of_axis(db, POSE_IDX, 'eps', [0, 2])
eps = eps-90
t_, sigt = calc_mean_of_axis(db, POSE_IDX, 'time', [0, 2])
t_ = np.array(rm_offset(t_))
t = t_/t_[-1]  # scale to length of 1

plt.figure('Epsilon mean')
plt.plot(t, eps, '-', color='green')
plt.fill_between(t, eps+sige, eps-sige,
                 facecolor='green', alpha=0.5)
plt.grid()
plt.xlabel('time in cycle [1]')
plt.ylabel('mean of varepsilon [deg]')
my_save.save_plt_as_tikz('Out/'+exp_path+'/04.tex')

# %% Bending angle plot
opa = .4  # opacity of simulated data

alp, sigalp = [], []
for i in [0, 1, 2, 4, 5]:
    a, sig = calc_mean_of_axis(db, POSE_IDX, 'aIMG{}'.format(i), [0, 2])
    alp.append(a)
    sigalp.append(sig)

ref, sigref = [], []
for i in range(6):
    r, sigr = calc_mean_of_axis(db, POSE_IDX, 'pr{}'.format(i), [0, 2])
    ref.append(r)
    sigref.append(sigr)








# %%

ref_alp = [[], [], [], [], []]
for j in range(len(ref[0])):
    p = [ref[i][j] for i in range(6)]
    ref_a = calibration.get_alpha(p, 'vS11')
    for ii in range(5):
        ref_alp[ii].append(ref_a[ii])


fpos = [([], []) for i in range(len(t))]
sigfpos = [([], []) for i in range(len(t))]
for i in range(6):
    x, sigx = calc_mean_of_axis(db, POSE_IDX, 'x{}'.format(i), [0, 2])
    y, sigy = calc_mean_of_axis(db, POSE_IDX, 'y{}'.format(i), [0, 2])
    for tidx, (xi, yi, sigxi, sigyi) in enumerate(zip(x, y, sigx, sigy)):
        fpos[tidx][0].append(xi)
        fpos[tidx][1].append(yi)
        sigfpos[tidx][0].append(sigxi)
        sigfpos[tidx][1].append(sigyi)

fix = []
for i in range(4):
    a, sig = calc_mean_of_axis(db, POSE_IDX, 'f{}'.format(i), [0, 2])
    fix.append(a)


def get_actuator_color():
    return ['red', 'darkred', 'orange', 'blue', 'darkblue']


color = get_actuator_color()


# %%

opa = .8  # opacity of simulated data

fig = plt.figure('Bending angle')

ax = fig.subplots(5,1, sharex='all')


for i in range(5):
    alp_ds = pf.downsample(alp[i], proportion=prop)
    sig = pf.downsample(sigalp[i], proportion=prop)
    x = pf.downsample(t, proportion=prop)
    ref_alp_ds = pf.downsample(ref_alp[i], proportion=prop)
    ref_ds = pf.downsample(ref[i], proportion=prop)

    j = i + 1 if i > 2 else i
    idx = int(j/2) % 3
    ax[idx].plot(x, alp_ds, color=color[i])
    ax[idx].fill_between(x, alp_ds+sig, alp_ds-sig,
                     facecolor=color[i], alpha=0.5)

# add simulation
ax[1].plot(TIME, RESULT['alp'][2], 'orange', label='a2', alpha=opa)
ax[1].plot(TIME, REF['alp'][2], ':', color='orange', alpha=opa)

# for normal start:
switch_idx = REF['f1'].index(0)
t1 = TIME[:switch_idx]
t2 = TIME[switch_idx-1:]

ax[0].plot(t1, RESULT['alp'][0][:switch_idx], '-', color='red', label='a0', alpha=opa)
ax[0].plot(t2, RESULT['alp'][0][switch_idx-1:], '-', color='red', alpha=opa)
ax[0].plot(TIME, REF['alp'][0], ':', color='red')

ax[2].plot(t1, RESULT['alp'][4][:switch_idx], '-', color='darkblue', label='a4', alpha=opa)
ax[2].plot(t2, RESULT['alp'][4][switch_idx-1:], '-', color='darkblue', alpha=opa)
ax[2].plot(TIME, REF['alp'][4], ':', color='darkblue', alpha=opa)

ax[2].plot(t1, RESULT['alp'][3][:switch_idx], '-', color='blue', label='a3', alpha=opa)
ax[2].plot(t2, RESULT['alp'][3][switch_idx-1:], '-', color='blue', alpha=opa)
ax[2].plot(TIME, REF['alp'][3], ':', color='blue', alpha=opa)


ax[0].plot(t1, RESULT['alp'][1][:switch_idx], '-', color='darkred', label='a1', alpha=opa)
ax[0].plot(t2, RESULT['alp'][1][switch_idx-1:], '-', color='darkred', alpha=opa)
ax[0].plot(TIME, REF['alp'][1], ':', color='darkred', alpha=opa)

# save
pih = '$\\frac{\\pi}{2}$'

    
ax[0].set_yticks([0, 45, 90, 135])
ax[0].set_yticklabels(['0', '', pih, ''])
ax[0].set_ylabel('$\\alpha_0, \\alpha_1$ (rad)')
ax[1].set_yticks([-90, -45, 0])
ax[1].set_yticklabels(['$-\\frac{\\pi}{2}$', '', '0'])
ax[1].set_ylabel('$\\alpha_2$ (rad)')
ax[2].set_yticks([0, -45, 90, 135])
ax[2].set_yticklabels(['0', '', pih, ''])
ax[2].set_ylabel('$\\alpha_3, \\alpha_4$ (rad)')



def get_feet_color():
    return ['red', 'darkred', 'blue', 'darkblue']

def calc_phi(alp, eps):
    phi_ = []
    for idx in range(len(alp[0])):
        alp_i = [alp[i][idx] for i in range(5)]
        phi_.append(model._calc_phi(alp_i, eps[idx]))
    phi = []
    for i in range(4):
        phi.append([phi_[idx][i] for idx in range(len(alp[0]))])
    return phi

def mean_of_phi(db, POSE_IDX):
    PHI = {i: [] for i in range(4)}
    startend = [0, 2]
    for exp in range(len(db)):
        start = POSE_IDX[exp][startend[0]]
        end = POSE_IDX[exp][startend[1]]
        alp = [db[exp]['aIMG{}'.format(i)][start:end]
               for i in [0, 1, 2, 4, 5]]
        eps = db[exp]['eps'][start:end]
#        print('len alp: ', len(alp[0]))
        phi = calc_phi(alp, eps)
        for i in range(4):
            PHI[i].append(phi[i])
#    print(np.size(PHI[3]))
    phi_mean = []
    sigphi = []
    for i in range(4):
        X = list(PHI[i])
        mat = make_matrix_plain(X)
        xx, sigxx = calc_mean_stddev(mat)
#        print(type(xx))

        phi_mean.append(xx)
        sigphi.append(sigxx)
    return phi_mean, sigphi

# calc phi with already meaned data
phi_1 = calc_phi(alp, eps)
# calc phi with raw data and mean over all calculated phi
phi_2, sigphi_2 = mean_of_phi(db, POSE_IDX)

for i in range(4):
#        plt.plot(t, phi_1[i], color=get_feet_color()[i])
    y = pf.downsample(phi_2[i], proportion=prop)
    sig = pf.downsample(sigphi_2[i], proportion=prop)
    x = pf.downsample(t, proportion=prop)

    fix_ = pf.downsample(fix[1][10:], prop)
    switch_exp = list(fix_).index(0) + 5  # kleiner hack ;)

    ls_ = ['-', '--'] if i in [1, 2] else ['--', '-']

    ax[3].plot(x[:switch_exp], y[:switch_exp], ls_[0],
             color=get_feet_color()[i])
    ax[3].plot(x[switch_exp-1:], y[switch_exp-1:], ls_[1],
             color=get_feet_color()[i])
    ax[3].fill_between(x, y+sig, y-sig,
                     facecolor=get_feet_color()[i], alpha=0.5)

y = pf.downsample(eps, proportion=prop)
sig = pf.downsample(sige, proportion=prop)
ax[4].plot(x, y, '-', color='green')
ax[4].fill_between(x, y+sig, y-sig,
                 facecolor='green', alpha=0.5)

# add sim
ax[4].plot(TIME, np.array(RESULT['eps'][0])-90, 'green', label='eps', alpha=opa)
ax[4].plot(TIME, REF['eps'][0], ':', color='green', label='eps', alpha=opa)

ax[4].set_ylabel('$\\varepsilon$ (rad)')


ax[3].plot(t1, RESULT['phi'][1][:switch_idx], '-', color='darkred', label='phi1', alpha=opa)
ax[3].plot(t2, RESULT['phi'][1][switch_idx-1:], '--', color='darkred', alpha=opa)
ax[3].plot(TIME, REF['phi'][1], ':', color='darkred', alpha=opa)

ax[3].plot(t1, RESULT['phi'][0][:switch_idx], '--', color='red', label='phi0', alpha=opa)
ax[3].plot(t2, RESULT['phi'][0][switch_idx-1:], '-', color='red', alpha=opa)
ax[3].plot(TIME, REF['phi'][0], ':', color='red', alpha=opa)

ax[3].plot(t1, RESULT['phi'][3][:switch_idx], '--', color='darkblue', label='phi3', alpha=opa)
ax[3].plot(t2, RESULT['phi'][3][switch_idx-1:], '-', color='darkblue', alpha=opa)
ax[3].plot(TIME, REF['phi'][3], ':', color='darkblue', alpha=opa)

ax[3].plot(t1, RESULT['phi'][2][:switch_idx], '-', color='blue', label='phi2', alpha=opa)
ax[3].plot(t2, RESULT['phi'][2][switch_idx-1:], '--', color='blue', alpha=opa)
ax[3].plot(TIME, REF['phi'][2], ':', color='blue', alpha=opa)

ax[3].set_ylabel('$\\varphi$ (rad)')



def pifrac(a, b):
    return '$\\frac{' + str(a) + '\\pi}{' + str(b) + '}$'


ax[3].set_yticks([0, 90, 180, 270, 360])
ax[3].set_yticklabels(['0', pifrac('', 2), '$\\pi$', pifrac(3, 2), '$2\\pi$'])


ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
ax[4].grid()


ax[4].set_xticks([0, .25, .5, .75, 1], ['0', '', '0.5', '', '1'])
ax[4].set_xlabel('time in cycle (1)')

ax[4].set_yticks([0, 45, 90])
ax[4].set_yticklabels(['0', '', pih])

ax[3].set_xlim([0, 1])
ax[4].set_ylim([-2, 92])
ax[3].set_ylim([-30, 380])
#ax[3].set_ylim([-135, 160.43])

plt.minorticks_off()

#plt.axis('tight')

kwargs = {
    'extra_axis_parameters': {
#            'anchor=origin',
            'width=15cm', 'height=4cm'},
    'extra_groupstyle_parameters': {'vertical sep={3pt}'}
}

#my_save.save_plt_as_tikz('Out/'+exp_path+'/orientations.tex')
my_save.save_plt_as_tikz('Out/'+exp_path+'/bending_angle.tex', **kwargs)



# %% Extract Poses

def get_actuator_tikzcolor():
    return ['red', 'red!50!black', 'orange', 'blue', 'blue!50!black']

def find_closest(val1, val2, target):
    return val2 if target - val1 >= val2 - target else val1

def get_closest_idx(arr, target):
    n = len(arr)
    left = 0
    right = n - 1
    mid = 0
    if target >= arr[n - 1]:
        return n-1
    if target <= arr[0]:
        return 0

    # BSearch solution: Time & Space: Log(N)
    while left < right:
        mid = (left + right) // 2  # find the mid
        if target < arr[mid]:
            right = mid
        elif target > arr[mid]:
            left = mid + 1

    return mid

plt.figure('Extrated Poses')

for ti in [0.01, .15, .52, .6, .99]:
    tidx = get_closest_idx(t, ti)
    fix_i = [round(fix[i][tidx]) for i in range(4)]
    fpos_i = fpos[tidx]
    alp_i = [alp[i][tidx] for i in range(5)]
    eps_i = eps[tidx]
    x_i = alp_i + ell_n + [eps_i]
    pose_raw = roboter_repr.GeckoBotPose(x_i, fpos_i, fix_i)
    print('time in cyc: ', ti, '\tfeet: ', fix_i)
#        pose_raw.plot()

    def correct(alp, eps, fpos, fix):
        alp_c, eps_c, fpos_c = inverse_kinematics.correct_measurement(
                alp_i, eps, fpos, len_leg, len_tor)
        x_c = alp_c + ell_n + [eps_c]
        pose_cor = roboter_repr.GeckoBotPose(x_c, fpos_c, fix)
        return pose_cor

    if pose_raw.complete():
        pose_cor = correct(alp_i, eps_i, fpos_i, fix_i)
    pose_cor.plot()
    pose_cor.save_as_tikz2('Out/'+exp_path+'/pose_'+str(ti).replace('.','')+'.tex',
                           col=get_actuator_tikzcolor(), linewidth='1mm')

plt.axis('scaled')




plt.show()
