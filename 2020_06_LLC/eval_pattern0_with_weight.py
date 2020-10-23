#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 08:15:29 2020

@author: ls
"""

import matplotlib.pyplot as plt
import numpy as np

import load as my_load

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from Src import save as my_save


# %% LOAD
exp = 'pattern0_31_2g'
filenames = my_load.get_csv_set(exp)

db = []
for filename in filenames:
    data = my_load.read_csv(exp+'/'+filename+'.csv')
    db.append(data)

# %% find pattern start and end

smargin = 40  # number of samples before pattern
emargin = 100  # number of samples after pattern

for idx, data in enumerate(db):
    ctrname = filenames[idx]
    alpha = np.array(data['aIMU0'])
    aref = np.array(data['aref0'])
    pressure_ref = np.array(data['u0'])/100.
    time = np.array(data['time'])
    N = len(aref)
    print(ctrname, '(', N, ')')
    start = [i-smargin for i in range(N-1) if (aref[i]==0 and aref[i+1]==60)]
    end = [i+emargin for i in range(N-1) if (aref[i]==40 and aref[i+1]==0)]
    
    if start[0] < 0:
        start = start[1:]
        end = end[1:]
    
    ## Counter Check
#    plt.figure('Index'+str(idx))
#    plt.plot(alpha)
#    plt.plot(aref)
#    plt.plot(start, np.zeros((len(start))), 'ko')
#    plt.plot(end, np.zeros((len(end))), 'kx')
    


# %% Resample and mean
    def resample_and_mean(x, start, end):
        mean_len = int(np.mean([e-s for (s, e) in zip(start, end)]))
        x_resample = np.zeros((len(start), mean_len))
        for i, (start_idx, end_idx) in enumerate(zip(start, end)):
            x_resample[i][:] = np.interp(
                    np.arange(0, mean_len, 1),
                    np.arange(0, end_idx-start_idx),
                    x[start_idx:end_idx])
        mean = np.mean(x_resample, axis=0)
        std = np.std(x_resample, axis=0)
        return x_resample, mean, std
#    ## Counter Check
#    resam, alp, alp_std = resample_and_mean(alpha, start, end)
#    plt.plot(alp)
#    for sample in resam:
#        plt.plot(sample, ':')





# %% evaluation criteria

    _, t, _ = resample_and_mean(time, start, end)
    t = t - t[0]  # remove offset
    _, aref_m, _ = resample_and_mean(aref, start, end)
    _, alp, alp_std = resample_and_mean(alpha, start, end)
    _, pref, pref_std = resample_and_mean(pressure_ref, start, end)



    def find_nearest_idx(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    tchange = [1.35 ,6.35 ,11.35, 16.35]
    idxchange = [find_nearest_idx(t, timestep) for timestep in tchange]
    refchange = [aref_m[i] for i in idxchange]

    # rise time
    rise_time = []
    for step in [(0, 1), (2, 3)]:
        start = alp[idxchange[step[0]]]
        final = alp[idxchange[step[1]]]
        angle_change = final-start
        threshold1 = idxchange[step[0]]+1 + np.argmax(alp[idxchange[step[0]]:] > start+angle_change*.1)
        threshold2 = idxchange[step[0]]+1 + np.argmax(alp[idxchange[step[0]]:] > start+angle_change*.9)
        rise_time.append(t[threshold2] - t[threshold1])
        plt.plot([t[threshold2], t[threshold2]], [0, 90], ':k')
        plt.plot([t[threshold1], t[threshold1]], [0, 90], ':k')
    print('rise time (mean): [{:3.2f}, {:3.2f}] ({:3.2f})'.format(rise_time[0], rise_time[1], np.mean(rise_time)))
    
    # settling time    
    settling_time = []
    for step in [(0, 1), (2, 3)]:
        start = alp[idxchange[step[0]]]
        final = alp[idxchange[step[1]]]
        angle_change = final-start
        threshold1 = idxchange[step[0]]+1 + np.argmax(alp[idxchange[step[0]]:] > start+angle_change*.95)
        settling_time.append(t[threshold1] - t[idxchange[step[0]]])
        plt.plot([t[threshold1], t[threshold1]], [-10, 100], ':r')
        plt.plot([t[idxchange[step[0]]], t[idxchange[step[0]]]], [-10, 100], ':r')
    print('rise time (mean): [{:3.2f}, {:3.2f}] ({:3.2f})'.format(settling_time[0], settling_time[1], np.mean(settling_time)))
    
    
    plt.plot(t, alp)
    plt.plot(t, aref_m)
    etrack = np.nanmean(np.abs(alp-aref_m))
    print('tracking error:', etrack)

# %% plot

#    plt.figure('Slices'+str(idx))
#    ax1 = plt.gca()  # alp axis
#    ax2 = ax1.twinx()  # pref axis
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
         gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(t, aref_m, '--', color='blue')
    ax1.plot(t, alp, color='blue')
    ax1.fill_between(t, alp-alp_std, alp+alp_std, alpha=.2, color='blue')
    
    ax2.plot(t, pref, color='red')
    ax2.fill_between(t, pref-pref_std, pref+pref_std, alpha=.2, color='blue')
    
    ax1.set_yticks([0, 20, 60, 90])
    ax1.set_ylim(-10, 110)
    ax1.set_xlim(0, 25)
    ax1.set_ylabel(r'$\alpha$ ($^\circ$)', color='blue')
    ax1.tick_params('y', colors='blue')
    ax1.grid()

    ax2.set_ylim(-.1, 1.1)
    ax2.set_yticks([0, .5, 1])
    ax2.set_ylabel(r'$\bar{p}$ (bar)', color='red')
    ax2.tick_params('y', colors='red')
    ax2.grid()
    ax2.set_xlabel('time (s)')
    
    sep = r"2cm"
    kwargs = {
        'strict': 1,
        'extra_tikzpicture_parameters': {},
        'extra_axis_parameters': {'height={3cm}', 'width={9cm}'},
        'extra_groupstyle_parameters': {'vertical sep={0pt}',
                                        'x descriptions at=edge bottom'}
            }

    my_save.save_plt_as_tikz('Out/31_2g/'+ctrname+'.tex', **kwargs)


    plt.show()
    