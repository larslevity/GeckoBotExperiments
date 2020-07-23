#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:50:56 2020

@author: ls
"""

import matplotlib.pyplot as plt
import numpy as np

import load as my_load
import compute_utils as utils

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from Src import save as my_save


# %% LOAD
exp = 'pattern0_0g/PPID.csv'
exp = 'pattern0_0g/PBooster_1_0.csv'
data = my_load.read_csv(exp)

# %% find pattern start and end

smargin = 40  # number of samples before pattern
emargin = 100  # number of samples after pattern


alpha_marica = np.array(data['aIMU0'])

time = np.array(data['time'])
Ts = np.mean(np.diff(time))  # calc mean sampling time
gamma = .07

accA = np.array([np.array(data['accx0']), np.array(data['accy0']), np.array(data['accz0'])])
accA_filt = np.array(
        [utils.lowpass_array(data['accx0'], Ts, gamma),
         utils.lowpass_array(data['accy0'], Ts, gamma),
         utils.lowpass_array(data['accz0'], Ts, gamma)])

accB = np.array([np.array(data['accx1']), np.array(data['accy1']), np.array(data['accz1'])])
accB_filt = np.array(
        [utils.lowpass_array(data['accx1'], Ts, gamma),
         utils.lowpass_array(data['accy1'], Ts, gamma),
         utils.lowpass_array(data['accz1'], Ts, gamma)])

gyrA = np.array([np.array(data['gyrx0']), np.array(data['gyry0']), np.array(data['gyrz0'])])
gyrA_filt = np.array(
        [utils.lowpass_array(data['gyrx0'], Ts, gamma),
         utils.lowpass_array(data['gyry0'], Ts, gamma),
         utils.lowpass_array(data['gyrz0'], Ts, gamma)])

gyrB = np.array([np.array(data['gyrx1']), np.array(data['gyry1']), np.array(data['gyrz1'])])
gyrB_filt = np.array(
        [utils.lowpass_array(data['gyrx1'], Ts, gamma),
         utils.lowpass_array(data['gyry1'], Ts, gamma),
         utils.lowpass_array(data['gyrz1'], Ts, gamma)])


aref = np.array(data['aref0'])
pressure_ref = np.array(data['u0'])/100.

N = len(aref)
start = [i-smargin for i in range(N-1) if (aref[i]==0 and aref[i+1]==60)]
end = [i+emargin for i in range(N-1) if (aref[i]==40 and aref[i+1]==0)]


which_cyc = 2
s = start[which_cyc]
e = end[which_cyc]

col = ['red', 'orange', 'gray']

# %% Check lowpass filt
fig = plt.figure(2)
for i in range(3):
    plt.plot(time[s:e]-time[s], accB[i][s:e], ':', color=col[i])
for i in range(3):
    plt.plot(time[s:e]-time[s], accB_filt[i][s:e], color=col[i])
plt.ylabel('acc$_B$')
plt.grid()
fig.set_size_inches(18.5, 10.5)
plt.savefig('Out/sensorfusion/check_filt.png', dpi=300)


# %% compute alphas

alpha_a = []
for (aA, aB) in zip(accA.T, accB.T):
    alpha_a.append(utils.calc_angle(aA, aB))
alpha_acc = np.array(alpha_a)

alpha_a = []
for (aA, aB) in zip(accA_filt.T, accB_filt.T):
    alpha_a.append(utils.calc_angle(aA, aB))
alpha_acc_filt = np.array(alpha_a)


# %% a dyn
domegaB_z = np.diff(np.insert(gyrB_filt[2], 0, 0))/800


accB_static_x = [0]
accB_static_y = [0]
alpha_dyn_filt = [0]
ell = 11.2

for (aA, aB, dwB) in zip(accA_filt.T[1:], accB_filt.T[1:], domegaB_z):
    adynx, adyny = utils.a_dyn(dwB, alpha_dyn_filt[-1], ell)
    aB_static = [aB[0]+adynx, aB[1]+adyny, aB[2]]
    accB_static_x.append(aB_static[0])
    accB_static_y.append(aB_static[1])
    alpha_dyn_filt.append(utils.calc_angle(aA, aB_static))
alpha_dyn_filt = np.array(alpha_dyn_filt)
accB_static_x = np.array(accB_static_x)
accB_static_y = np.array(accB_static_y)


    
## %% plot only alpha

fig = plt.figure(3)
fig.set_size_inches(18.5, 10.5)
plt.plot(time[s:e]-time[s], alpha_acc[s:e], '-', color='orange',  label='$\\alpha(a_{raw})$')
plt.plot(time[s:e]-time[s], alpha_marica[s:e], '--', color='blue',  label='$\\alpha(marica)$')
plt.plot(time[s:e]-time[s], alpha_acc_filt[s:e], ':', color='red',  label='$\\alpha(a_{filt})$')
plt.plot(time[s:e]-time[s], alpha_dyn_filt[s:e], '-o', color='purple',  label='$\\alpha(a_{dyn, filt})$')


plt.ylabel('bending angle $\\alpha$ ($^\\circ$)')
plt.xlabel('time (s)')
plt.grid()
plt.legend()
plt.savefig('Out/sensorfusion/alpha_course.png', dpi=300)

# %% plot tripple axis

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True,
     gridspec_kw={'height_ratios': [.1, .1, 2, 2, 3]})

limacc = [np.min(accB)-.3, np.max(accB)+.3]
limgyr = [np.min(gyrB)-10, np.max(gyrB)+10]

# IMU A
for i in range(3):
    ax1.plot(time[s:e]-time[s], gyrA[i][s:e], '--', color=col[i])
for i in range(3):
    ax1.plot(time[s:e]-time[s], gyrA_filt[i][s:e], color=col[i])
ax1.set_ylabel('gyr$_A$')
ax1.grid()
ax1.set_ylim(*limgyr)

for i in range(3):
    ax2.plot(time[s:e]-time[s], accA[i][s:e], '--', color=col[i])
for i in range(3):
    ax2.plot(time[s:e]-time[s], accA_filt[i][s:e], color=col[i])
ax2.set_ylabel('acc$_A$')
ax2.grid()
ax2.set_ylim(*limacc)


# IMU B
for i in range(3):
    ax3.plot(time[s:e]-time[s], gyrB[i][s:e], '--', color=col[i])
for i in range(3):
    ax3.plot(time[s:e]-time[s], gyrB_filt[i][s:e], color=col[i])
ax3.set_ylabel('gyr$_B$')
ax3.grid()
ax3.set_ylim(*limgyr)

for i in range(3):
    ax4.plot(time[s:e]-time[s], accB[i][s:e], '--', color=col[i])
for i in range(3):
    ax4.plot(time[s:e]-time[s], accB_filt[i][s:e], color=col[i])
ax4.set_ylabel('acc$_B$')
ax4.grid()
ax4.set_ylim(*limacc)

# alpha
ax5.plot(time[s:e]-time[s], alpha_acc[s:e], '--', color='orange',  label='$\\alpha(a_{raw})$')
ax5.plot(time[s:e]-time[s], alpha_acc_filt[s:e], '-', color='red',  label='$\\alpha(a_{filt})$')
ax5.plot(time[s:e]-time[s], alpha_marica[s:e], '-', color='blue',  label='$\\alpha(marica)$')
ax5.set_yticks([0, 20, 60, 90])
ax5.set_xticks([0, 1.85, 3.1, 6.9, 11.75, 13.7, 17, 21.6])
ax5.set_ylim(-10, 110)
ax5.set_xlim(0, 25)
ax5.set_ylabel(r'$\alpha$ ($^\circ$)')
ax5.set_xlabel('time (s)')
ax5.grid()
ax5.legend()

fig.set_size_inches(18.5, 10.5)
plt.savefig('Out/sensorfusion/alpha_course_2.png', dpi=300)

# %%

## %% Resample and mean
#    def resample_and_mean(x, start, end):
#        mean_len = int(np.mean([e-s for (s, e) in zip(start, end)]))
#        x_resample = np.zeros((len(start), mean_len))
#        for i, (start_idx, end_idx) in enumerate(zip(start, end)):
#            x_resample[i][:] = np.interp(
#                    np.arange(0, mean_len, 1),
#                    np.arange(0, end_idx-start_idx),
#                    x[start_idx:end_idx])
#        mean = np.mean(x_resample, axis=0)
#        std = np.std(x_resample, axis=0)
#        return x_resample, mean, std
##    ## Counter Check
##    resam, alp, alp_std = resample_and_mean(alpha, start, end)
##    plt.plot(alp)
##    for sample in resam:
##        plt.plot(sample, ':')
#
#    _, t, _ = resample_and_mean(time, start, end)
#    t = t - t[0]  # remove offset
#    _, aref_m, _ = resample_and_mean(aref, start, end)
#    _, alp, alp_std = resample_and_mean(alpha, start, end)
#    _, pref, pref_std = resample_and_mean(pressure_ref, start, end)
#
##    plt.figure('Slices'+str(idx))
##    ax1 = plt.gca()  # alp axis
##    ax2 = ax1.twinx()  # pref axis
#
#
#    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
#         gridspec_kw={'height_ratios': [3, 1]})
#    
#    ax1.plot(t, aref_m, '--', color='blue')
#    ax1.plot(t, alp, color='blue')
#    ax1.fill_between(t, alp-alp_std, alp+alp_std, alpha=.2, color='blue')
#    
#    ax2.plot(t, pref, color='red')
#    ax2.fill_between(t, pref-pref_std, pref+pref_std, alpha=.2, color='blue')
#    
#    ax1.set_yticks([0, 20, 60, 90])
#    ax1.set_ylim(-10, 110)
#    ax1.set_xlim(0, 25)
#    ax1.set_ylabel(r'$\alpha$ ($^\circ$)', color='blue')
#    ax1.tick_params('y', colors='blue')
#    ax1.grid()
#
#    ax2.set_ylim(-.1, 1.1)
#    ax2.set_yticks([0, .5, 1])
#    ax2.set_ylabel(r'$\bar{p}$ (bar)', color='red')
#    ax2.tick_params('y', colors='red')
#    ax2.grid()
#    ax2.set_xlabel('time (s)')
#    
#    sep = r"2cm"
#    kwargs = {
#        'strict': 1,
#        'extra_tikzpicture_parameters': {},
#        'extra_axis_parameters': {'height={4cm}', 'width={12cm}'},
#        'extra_groupstyle_parameters': {'vertical sep={0pt}',
#                                        'x descriptions at=edge bottom'}
#            }
#
#    my_save.save_plt_as_tikz('Out/sensorfusion/test.tex', **kwargs)
#
#
#    plt.show()
#    